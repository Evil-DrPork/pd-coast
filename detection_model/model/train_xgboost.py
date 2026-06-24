from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

from .dataset import ChunkSample, augment_chunk_windows, load_public_benchmark
from .inference import Poker44BotDetector
from .train_hierarchical import find_best_threshold


try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tree head on top of the hierarchical PyTorch chunk encoder."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--torch-model", required=True)
    parser.add_argument("--out", default="artifacts/p44_xgb_detector.joblib")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--augment-windows", action="store_true")
    parser.add_argument("--augment-validation-windows", action="store_true")
    parser.add_argument("--window-hands", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--keep-short-window-chunks", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--no-auto-threshold", action="store_true")
    return parser.parse_args()


def apply_window_augmentation(
    samples: List[ChunkSample],
    *,
    enabled: bool,
    window_hands: int,
    window_stride: int,
    keep_short: bool,
    name: str,
) -> List[ChunkSample]:
    if not enabled:
        return samples

    before = len(samples)
    out = augment_chunk_windows(
        samples,
        window_hands=window_hands,
        stride=window_stride,
        keep_short_chunks=keep_short,
    )
    print(
        f"{name} window chunks: {before} -> {len(out)} "
        f"(window_hands={window_hands}, stride={window_stride})"
    )
    return out


def extract_features(
    detector: Poker44BotDetector,
    samples: List[ChunkSample],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    labels: List[int] = []

    detector.model.eval()

    chunks = [sample.chunk for sample in samples]
    labels = [int(sample.label) for sample in samples]

    for start in tqdm(range(0, len(chunks), batch_size), desc="extract embeddings"):
        batch_chunks = chunks[start:start + batch_size]
        batch = detector._make_inference_batch(batch_chunks)

        chunk_embedding = detector.model.extract_chunk_embedding(
            action_cat=batch["action_cat"],
            action_num=batch["action_num"],
            action_mask=batch["action_mask"],
            hand_mask=batch["hand_mask"],
            hand_num=batch.get("hand_num"),
        )

        emb_np = chunk_embedding.detach().cpu().numpy().astype(np.float32)
        feat_np = batch["features"].detach().cpu().numpy().astype(np.float32)
        rows.append(np.concatenate([emb_np, feat_np], axis=1))

    if not rows:
        raise RuntimeError("No samples available for XGBoost training.")

    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.int32)


def make_classifier(args: argparse.Namespace, y_train: np.ndarray) -> Any:
    if XGBClassifier is not None:
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        scale_pos_weight = neg / max(pos, 1.0)
        return XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=args.seed,
            n_jobs=0,
            scale_pos_weight=scale_pos_weight,
        )

    print("xgboost is not installed; using sklearn HistGradientBoostingClassifier fallback.")
    return HistGradientBoostingClassifier(
        max_iter=args.n_estimators,
        max_leaf_nodes=max(2, 2 ** int(args.max_depth)),
        learning_rate=args.learning_rate,
        l2_regularization=args.reg_lambda,
        random_state=args.seed,
    )


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=np.float32)
        return np.asarray(proba, dtype=np.float32).reshape(-1)

    raw = np.asarray(model.predict(x), dtype=np.float32).reshape(-1)
    if raw.min() < 0.0 or raw.max() > 1.0:
        raw = 1.0 / (1.0 + np.exp(-raw))
    return raw


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float32).clip(1e-6, 1 - 1e-6)
    labels = np.asarray(labels, dtype=np.int32)
    preds = (scores >= threshold).astype(np.int32)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metrics: Dict[str, Any] = {
        "count": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else 0.0,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "confusion_matrix": {
            "tn_human_pred_human": int(cm[0, 0]),
            "fp_human_pred_bot": int(cm[0, 1]),
            "fn_bot_pred_human": int(cm[1, 0]),
            "tp_bot_pred_bot": int(cm[1, 1]),
        },
    }
    metrics.update(find_best_threshold(labels, scores))

    if len(set(labels.tolist())) > 1:
        metrics["log_loss"] = float(log_loss(labels, scores, labels=[0, 1]))
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        metrics["log_loss"] = 0.0
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    return metrics


def main() -> None:
    args = parse_args()

    train_samples, val_samples = load_public_benchmark(args.data, seed=args.seed)
    train_samples = apply_window_augmentation(
        train_samples,
        enabled=args.augment_windows,
        window_hands=args.window_hands,
        window_stride=args.window_stride,
        keep_short=args.keep_short_window_chunks,
        name="Train",
    )
    val_samples = apply_window_augmentation(
        val_samples,
        enabled=args.augment_validation_windows,
        window_hands=args.window_hands,
        window_stride=args.window_stride,
        keep_short=args.keep_short_window_chunks,
        name="Validation",
    )

    print(f"Loading hierarchical torch model: {args.torch_model}")
    detector = Poker44BotDetector.load(args.torch_model, device=args.device)

    x_train, y_train = extract_features(detector, train_samples, args.batch_size)
    x_val, y_val = extract_features(detector, val_samples, args.batch_size)

    print(f"Train feature matrix: {x_train.shape}")
    print(f"Validation feature matrix: {x_val.shape}")

    model = make_classifier(args, y_train)
    model.fit(x_train, y_train)

    train_scores = predict_proba(model, x_train)
    val_scores = predict_proba(model, x_val)

    val_threshold = float(args.threshold)
    val_metrics_at_fixed = compute_metrics(y_val, val_scores, threshold=val_threshold)
    selected_threshold = (
        float(val_metrics_at_fixed.get("best_threshold", args.threshold))
        if not args.no_auto_threshold
        else float(args.threshold)
    )

    train_metrics = compute_metrics(y_train, train_scores, threshold=selected_threshold)
    val_metrics = compute_metrics(y_val, val_scores, threshold=selected_threshold)

    print("Train metrics:")
    print(json.dumps(train_metrics, indent=2))
    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print(f"Selected threshold: {selected_threshold:.6f}")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
    except Exception as exc:  # pragma: no cover
        raise ImportError("joblib is required to save the trained tree head.") from exc

    payload = {
        "xgb_model": model,
        "model": model,
        "threshold": selected_threshold,
        "feature_dim": int(x_train.shape[1]),
        "backend": "xgboost" if XGBClassifier is not None else "sklearn_hist_gradient_boosting",
        "args": vars(args),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }
    joblib.dump(payload, out_path)
    print(f"Saved tree head to: {out_path}")


if __name__ == "__main__":
    main()
