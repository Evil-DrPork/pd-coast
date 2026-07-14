"""Train Poker44 V4 on real coherent chunks with chronological validation.

The final two non-empty dates are a locked promotion holdout.  Model/mapper
selection uses only expanding walk-forward predictions from earlier dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np

from detection_model.model_v3.calibration import fit_fixed_mapper
from detection_model.model_v3.schema import Chunk, canonicalize_chunks, chunk_multiset_hash, load_chunks
from poker44.score.scoring import reward as validator_reward

from .features import (
    FEATURE_IMPLEMENTATION_SHA256,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    matrix_for_chunks,
)
from .mapping import chunk_tie_key, exact_rank_map, mapping_metadata
from .model import (
    BRANCH_NAMES,
    RAW_BRANCH_COUNT,
    CoherentEnsemble,
    ModelConfig,
    blend_branches,
    percentile_feature_matrix,
)
from .provenance import DETECTOR_RUNTIME_SHA256, RUNTIME_LIBRARY_VERSIONS


PUBLIC_TREE_WEIGHTS = np.asarray(
    [0.45, 0.25, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="detection_model/artifacts/p44_v4_coherent.joblib")
    parser.add_argument("--report", default="detection_model/artifacts/p44_v4_coherent_train_report.json")
    parser.add_argument("--feature-cache", default="detection_model/artifacts/p44_v4_features.npz")
    parser.add_argument("--canonicalize", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--selection-days", type=int, default=6)
    parser.add_argument("--holdout-days", type=int, default=2)
    parser.add_argument("--windows-per-date", type=int, default=30)
    parser.add_argument("--weight-step", type=float, default=0.10)
    parser.add_argument("--finalists", type=int, default=24)
    parser.add_argument("--fractions", default="0.10,0.125,0.15")
    parser.add_argument("--sample-date-power", type=float, default=0.5)
    parser.add_argument("--cv-trees", type=int, default=300)
    parser.add_argument("--cv-hist-iterations", type=int, default=300)
    parser.add_argument("--final-trees", type=int, default=700)
    parser.add_argument("--final-hist-iterations", type=int, default=700)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--no-final-fit", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_schema_sha256() -> str:
    return FEATURE_SCHEMA_SHA256


def _feature_implementation_sha256() -> str:
    return FEATURE_IMPLEMENTATION_SHA256


def _feature_matrix(
    chunks: Sequence[Chunk],
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    cache_path: Path,
    data_sha256: str,
    canonicalize_mode: str,
) -> np.ndarray:
    runtime_versions_json = json.dumps(RUNTIME_LIBRARY_VERSIONS, sort_keys=True)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            cache_data_sha = str(cached["data_sha256"].item())
            cache_schema_sha = str(cached["feature_schema_sha256"].item())
            cache_implementation_sha = (
                str(cached["feature_implementation_sha256"].item())
                if "feature_implementation_sha256" in cached.files
                else ""
            )
            matrix = np.asarray(cached["x"], dtype=np.float64)
            cached_labels = np.asarray(cached["y"], dtype=int)
            cached_dates = np.asarray(cached["dates"], dtype=str)
            cached_canonicalize_mode = (
                str(cached["canonicalize_mode"].item())
                if "canonicalize_mode" in cached.files
                else ""
            )
            cached_runtime_versions_json = (
                str(cached["runtime_library_versions"].item())
                if "runtime_library_versions" in cached.files
                else ""
            )
        if (
            cache_data_sha == data_sha256
            and cache_schema_sha == _feature_schema_sha256()
            and cache_implementation_sha == _feature_implementation_sha256()
            and cached_canonicalize_mode == canonicalize_mode
            and cached_runtime_versions_json == runtime_versions_json
            and matrix.shape == (len(chunks), len(FEATURE_NAMES))
            and np.array_equal(cached_labels, labels)
            and np.array_equal(cached_dates, dates)
        ):
            print(f"Loaded validated feature cache: {cache_path}")
            return matrix
        print(f"Ignoring stale feature cache: {cache_path}")
    matrix = matrix_for_chunks([chunk.hands for chunk in chunks])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=matrix,
        y=labels,
        dates=dates,
        data_sha256=np.asarray(data_sha256),
        feature_schema_sha256=np.asarray(_feature_schema_sha256()),
        feature_implementation_sha256=np.asarray(_feature_implementation_sha256()),
        canonicalize_mode=np.asarray(canonicalize_mode),
        runtime_library_versions=np.asarray(runtime_versions_json),
    )
    print(f"Saved feature cache: {cache_path}")
    return matrix


def _deduplicate(chunks: Sequence[Chunk]) -> tuple[list[Chunk], int]:
    seen: dict[str, int | None] = {}
    output: list[Chunk] = []
    conflicts = 0
    for chunk in chunks:
        digest = chunk_multiset_hash(chunk)
        if digest in seen:
            conflicts += int(seen[digest] != chunk.label)
            continue
        seen[digest] = chunk.label
        output.append(chunk)
    if conflicts:
        raise ValueError(f"found {conflicts} duplicate chunks with conflicting labels")
    return output, len(chunks) - len(output)


def _date_weights(chunks: Sequence[Chunk], power: float) -> np.ndarray:
    counts = Counter(chunk.source_date or "" for chunk in chunks)
    weights = np.asarray(
        [(1.0 / max(counts[chunk.source_date or ""], 1)) ** float(power) for chunk in chunks],
        dtype=float,
    )
    return weights / max(float(weights.mean()), 1e-12)


def _metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    value, details = validator_reward(np.asarray(scores, dtype=float), np.asarray(labels, dtype=int))
    return {key: float(item) for key, item in details.items()} | {"reward": float(value)}


def _simplex(total: int, parts: int, prefix: tuple[int, ...] = ()) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (*prefix, total)
        return
    for value in range(total + 1):
        yield from _simplex(total - value, parts - 1, (*prefix, value))


def _candidate_weights(step: float) -> list[np.ndarray]:
    units = int(round(1.0 / float(step)))
    if units < 2 or not np.isclose(units * float(step), 1.0):
        raise ValueError("weight-step must evenly divide 1.0")
    if len(BRANCH_NAMES) != 9:
        raise RuntimeError("update the bounded V4 candidate search for the branch schema")
    candidates: list[np.ndarray] = []
    # Preserve the fine-grained search over the six established raw branches.
    for parts in _simplex(units, 6):
        candidates.append(np.r_[np.asarray(parts, dtype=float) / units, np.zeros(3)])
    # Search the three scale-invariant branches at the same fine resolution.
    for parts in _simplex(units, 3):
        candidates.append(np.r_[np.zeros(6), np.asarray(parts, dtype=float) / units])
    # Explore raw/rank mixtures on a bounded 20% grid rather than exploding the
    # nine-dimensional 10% simplex and over-selecting six validation dates.
    coarse_units = 5
    candidates.extend(
        np.asarray(parts, dtype=float) / coarse_units
        for parts in _simplex(coarse_units, len(BRANCH_NAMES))
    )
    candidates.extend(
        [
            PUBLIC_TREE_WEIGHTS.copy(),
            np.asarray([0.60, 0.25, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.35, 0.25, 0.20, 0.10, 0.10, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.2, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.2]),
            np.asarray([0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.1]),
        ]
    )
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for value in candidates:
        value = np.clip(value, 0.0, None)
        value = value / value.sum()
        unique[tuple(np.round(value, 8))] = value
    return list(unique.values())


def _balanced_windows(labels: np.ndarray, count: int, repeats: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    left = count // 2
    windows = []
    for _ in range(max(1, repeats)):
        pos = rng.choice(positive, left, replace=len(positive) < left)
        neg = rng.choice(negative, count - left, replace=len(negative) < count - left)
        index = np.concatenate([pos, neg])
        rng.shuffle(index)
        windows.append(index)
    return windows


def _mapped_metrics(
    branches: np.ndarray,
    labels: np.ndarray,
    keys: Sequence[str],
    weights: np.ndarray,
    mode: str,
    fraction: float,
) -> dict[str, float]:
    raw = blend_branches(branches, weights, mode, tie_keys=keys)
    mapped = exact_rank_map(raw, fraction, tie_keys=keys)
    return _metric(labels, mapped)


def _request_branches(fold: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    """Recompute request-relative model branches without refitting raw models."""
    subset = np.asarray(fold["branches"][indices], dtype=float)
    model = fold.get("model")
    matrix = fold.get("matrix")
    if model is None or matrix is None:
        return subset
    return model.request_rebased_branches(np.asarray(matrix)[indices], subset)


def _batched_request_branches(
    fold: dict[str, Any],
    windows: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Rebase many request windows with one estimator call per rank branch."""
    if not windows:
        return []
    model = fold.get("model")
    matrix = fold.get("matrix")
    if model is None or matrix is None:
        return [np.asarray(fold["branches"][indices], dtype=float) for indices in windows]

    matrix = np.asarray(matrix, dtype=float)
    ranked_requests = [percentile_feature_matrix(matrix[indices]) for indices in windows]
    lengths = [len(value) for value in ranked_requests]
    combined = np.vstack(ranked_requests)
    _, _, coherent = model._views(combined)
    rank_scores = np.clip(
        np.column_stack(
            (
                model.rank_coherent_hist.predict_proba(coherent)[:, 1],
                model.rank_combined_extra.predict_proba(combined)[:, 1],
                model.rank_combined_logistic.predict_proba(combined)[:, 1],
            )
        ),
        1e-6,
        1.0 - 1e-6,
    )
    output: list[np.ndarray] = []
    offset = 0
    for indices, length in zip(windows, lengths):
        branches = np.asarray(fold["branches"][indices], dtype=float).copy()
        branches[:, RAW_BRANCH_COUNT:] = rank_scores[offset : offset + length]
        output.append(branches)
        offset += length
    return output


def _robust_objective(rewards: Sequence[float]) -> tuple[float, float, float, float]:
    values = np.asarray(rewards, dtype=float)
    mean = float(values.mean())
    std = float(values.std())
    p10 = float(np.quantile(values, 0.10))
    return mean - 0.5 * std, mean, p10, std


def _select_configuration(
    folds: list[dict[str, Any]],
    *,
    fractions: Sequence[float],
    weight_step: float,
    finalists: int,
    windows_per_date: int,
    seed: int,
) -> dict[str, Any]:
    weights = _candidate_weights(weight_step)
    shortlist: list[dict[str, Any]] = []
    for mode in ("probability", "rank"):
        for fraction in fractions:
            rows = []
            for candidate in weights:
                rewards = [
                    _mapped_metrics(
                        fold["branches"], fold["labels"], fold["keys"], candidate, mode, fraction
                    )["reward"]
                    for fold in folds
                ]
                robust, mean, p10, std = _robust_objective(rewards)
                rows.append(
                    {
                        "weights": candidate,
                        "mode": mode,
                        "fraction": float(fraction),
                        "date_objective": robust,
                        "date_mean": mean,
                        "date_p10": p10,
                        "date_std": std,
                    }
                )
            rows.sort(key=lambda row: (row["date_objective"], row["date_mean"]), reverse=True)
            shortlist.extend(rows[: max(1, finalists)])

    request_windows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        windows = _balanced_windows(
            fold["labels"], 40, windows_per_date, seed + 1000 * fold_index
        )
        for indices, branches in zip(windows, _batched_request_branches(fold, windows)):
            request_windows.append(
                {
                    "branches": branches,
                    "labels": fold["labels"][indices],
                    "keys": [fold["keys"][index] for index in indices],
                }
            )

    for row_index, row in enumerate(shortlist):
        rewards: list[float] = []
        for request in request_windows:
            reward = _mapped_metrics(
                request["branches"],
                request["labels"],
                request["keys"],
                row["weights"],
                row["mode"],
                row["fraction"],
            )["reward"]
            rewards.append(reward)
        robust, mean, p10, std = _robust_objective(rewards)
        row.update(
            {
                "window_objective": robust,
                "window_mean": mean,
                "window_p10": p10,
                "window_std": std,
                "window_count": len(rewards),
                "shortlist_index": row_index,
            }
        )
    return max(shortlist, key=lambda row: (row["window_objective"], row["window_mean"]))


def _evaluate_predictions(
    folds: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    window_size: int,
    windows_per_date: int,
    seed: int,
) -> dict[str, Any]:
    per_date = []
    window_rewards: list[float] = []
    all_labels = []
    all_scores = []
    for fold_index, fold in enumerate(folds):
        raw = blend_branches(
            fold["branches"], config["weights"], config["mode"], tie_keys=fold["keys"]
        )
        mapped = exact_rank_map(raw, config["fraction"], tie_keys=fold["keys"])
        metrics = _metric(fold["labels"], mapped)
        per_date.append({"date": fold["date"], **metrics})
        all_labels.append(fold["labels"])
        all_scores.append(mapped)
        windows = _balanced_windows(
            fold["labels"], window_size, windows_per_date, seed + 1000 * fold_index
        )
        for indices, branches in zip(windows, _batched_request_branches(fold, windows)):
            labels = fold["labels"][indices]
            keys = [fold["keys"][index] for index in indices]
            window_rewards.append(
                _mapped_metrics(
                    branches,
                    labels,
                    keys,
                    config["weights"],
                    config["mode"],
                    config["fraction"],
                )["reward"]
            )
    objective, mean, p10, std = _robust_objective(window_rewards)
    return {
        "window_size": int(window_size),
        "window_count": len(window_rewards),
        "window_objective": objective,
        "window_mean": mean,
        "window_p10": p10,
        "window_std": std,
        "per_date": per_date,
        "pooled": _metric(np.concatenate(all_labels), np.concatenate(all_scores)),
    }


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    chunks: Sequence[Chunk],
    indices: np.ndarray,
    *,
    seed: int,
    config: ModelConfig,
    date_power: float,
) -> CoherentEnsemble:
    selected_chunks = [chunks[index] for index in indices]
    sample_weight = _date_weights(selected_chunks, date_power)
    return CoherentEnsemble(seed, config).fit(
        x[indices],
        y[indices],
        sample_weight=sample_weight,
        groups=[chunk.source_date or "" for chunk in selected_chunks],
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_path = Path(args.data)
    data_sha256 = _sha256(data_path)
    repo_root = Path(__file__).resolve().parents[2]
    chunks = canonicalize_chunks(load_chunks(data_path), repo_root, args.canonicalize)
    chunks, duplicate_count = _deduplicate(chunks)
    if any(chunk.label is None or not chunk.source_date for chunk in chunks):
        raise SystemExit("V4 training requires a label and source_date on every chunk")
    y = np.asarray([int(chunk.label) for chunk in chunks], dtype=int)
    dates = np.asarray([str(chunk.source_date) for chunk in chunks])
    unique_dates = sorted(set(dates))
    required_days = args.selection_days + args.holdout_days
    if len(unique_dates) <= required_days:
        raise SystemExit("not enough dated history for selection and holdout")
    holdout_dates = unique_dates[-args.holdout_days :]
    selection_dates = unique_dates[-required_days : -args.holdout_days]

    print(
        f"Loaded {len(chunks)} unique real chunks ({duplicate_count} duplicates removed) | "
        f"bot={int(y.sum())} human={int(np.sum(y == 0))} dates={len(unique_dates)}"
    )
    print(f"Selection dates: {selection_dates}")
    print(f"Locked holdout:  {holdout_dates}")
    x = _feature_matrix(
        chunks,
        y,
        dates,
        cache_path=Path(args.feature_cache),
        data_sha256=data_sha256,
        canonicalize_mode=args.canonicalize,
    )
    print(f"V4 feature matrix: {x.shape}")

    cv_model_config = ModelConfig(
        trees=args.cv_trees,
        hist_iterations=args.cv_hist_iterations,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    selection_folds: list[dict[str, Any]] = []
    for fold_index, date in enumerate(selection_dates):
        train_indices = np.flatnonzero(dates < date)
        validation_indices = np.flatnonzero(dates == date)
        model = _fit_model(
            x,
            y,
            chunks,
            train_indices,
            seed=args.seed + fold_index,
            config=cv_model_config,
            date_power=args.sample_date_power,
        )
        branches = model.branch_scores(x[validation_indices])
        keys = [chunk_tie_key(chunks[index].hands) for index in validation_indices]
        default_metrics = _mapped_metrics(
            branches,
            y[validation_indices],
            keys,
            PUBLIC_TREE_WEIGHTS,
            "probability",
            0.125,
        )
        print(
            f"[{date}] train={len(train_indices)} validation={len(validation_indices)} "
            f"public-tree reward={default_metrics['reward']:.4f} "
            f"ap={default_metrics['ap_score']:.4f} recall={default_metrics['bot_recall']:.4f}"
        )
        selection_folds.append(
            {
                "date": date,
                "indices": validation_indices,
                "branches": branches,
                "matrix": x[validation_indices],
                "model": model,
                "labels": y[validation_indices],
                "keys": keys,
            }
        )

    fractions = [float(value.strip()) for value in args.fractions.split(",") if value.strip()]
    selected = _select_configuration(
        selection_folds,
        fractions=fractions,
        weight_step=args.weight_step,
        finalists=args.finalists,
        windows_per_date=args.windows_per_date,
        seed=args.seed,
    )
    print(
        "Selected: "
        f"mode={selected['mode']} fraction={selected['fraction']} "
        f"weights={np.round(selected['weights'], 3).tolist()} "
        f"window_mean={selected['window_mean']:.4f} p10={selected['window_p10']:.4f}"
    )
    selection_eval_40 = _evaluate_predictions(
        selection_folds,
        selected,
        window_size=40,
        windows_per_date=args.windows_per_date,
        seed=args.seed + 20000,
    )
    selection_eval_100 = _evaluate_predictions(
        selection_folds,
        selected,
        window_size=100,
        windows_per_date=max(10, args.windows_per_date // 2),
        seed=args.seed + 30000,
    )

    # Freeze configuration, then fit once through the last selection date and
    # evaluate both future dates without updating on either holdout label.
    final_model_config = ModelConfig(
        trees=args.final_trees,
        hist_iterations=args.final_hist_iterations,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    pre_holdout = np.flatnonzero(dates < holdout_dates[0])
    holdout_indices = np.flatnonzero(np.isin(dates, holdout_dates))
    holdout_model = _fit_model(
        x,
        y,
        chunks,
        pre_holdout,
        seed=args.seed,
        config=final_model_config,
        date_power=args.sample_date_power,
    )
    holdout_model.branch_weights_ = np.asarray(selected["weights"], dtype=float)
    holdout_folds = []
    for date in holdout_dates:
        local = np.flatnonzero(dates[holdout_indices] == date)
        global_indices = holdout_indices[local]
        date_matrix = x[global_indices]
        holdout_folds.append(
            {
                "date": date,
                "indices": global_indices,
                "branches": holdout_model.branch_scores(date_matrix),
                "matrix": date_matrix,
                "model": holdout_model,
                "labels": y[global_indices],
                "keys": [chunk_tie_key(chunks[index].hands) for index in global_indices],
            }
        )
    holdout_eval_40 = _evaluate_predictions(
        holdout_folds,
        selected,
        window_size=40,
        windows_per_date=max(100, args.windows_per_date),
        seed=args.seed + 40000,
    )
    holdout_eval_100 = _evaluate_predictions(
        holdout_folds,
        selected,
        window_size=100,
        windows_per_date=max(50, args.windows_per_date // 2),
        seed=args.seed + 50000,
    )
    print(
        f"LOCKED holdout | n=40 mean={holdout_eval_40['window_mean']:.4f} "
        f"p10={holdout_eval_40['window_p10']:.4f} | "
        f"n=100 mean={holdout_eval_100['window_mean']:.4f} "
        f"p10={holdout_eval_100['window_p10']:.4f}"
    )

    selection_oof_raw = np.concatenate(
        [
            blend_branches(fold["branches"], selected["weights"], selected["mode"], tie_keys=fold["keys"])
            for fold in selection_folds
        ]
    )
    selection_oof_y = np.concatenate([fold["labels"] for fold in selection_folds])
    mapper = fit_fixed_mapper(selection_oof_raw, selection_oof_y, target_human_fpr=0.05)

    artifact_path = Path(args.out)
    artifact_written = False
    if not args.no_final_fit:
        all_indices = np.arange(len(chunks), dtype=int)
        final_model = _fit_model(
            x,
            y,
            chunks,
            all_indices,
            seed=args.seed,
            config=final_model_config,
            date_power=args.sample_date_power,
        )
        final_model.branch_weights_ = np.asarray(selected["weights"], dtype=float)
        feature_reference = {
            "q01": np.quantile(x, 0.01, axis=0),
            "q25": np.quantile(x, 0.25, axis=0),
            "median": np.quantile(x, 0.50, axis=0),
            "q75": np.quantile(x, 0.75, axis=0),
            "q99": np.quantile(x, 0.99, axis=0),
        }
        artifact = {
            "artifact_version": 4,
            "architecture": "coherent_real_rank_robust_v2",
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "feature_schema_sha256": _feature_schema_sha256(),
            "feature_implementation_sha256": _feature_implementation_sha256(),
            "detector_runtime_sha256": DETECTOR_RUNTIME_SHA256,
            "runtime_library_versions": dict(RUNTIME_LIBRARY_VERSIONS),
            "branch_names": list(BRANCH_NAMES),
            "feature_reference": feature_reference,
            "blend_mode": selected["mode"],
            "mapper": mapper,
            "batch_top_fraction": float(selected["fraction"]),
            "mapping": mapping_metadata(selected["fraction"]),
            "training_data": str(data_path),
            "training_data_sha256": data_sha256,
            "training_count": int(len(chunks)),
            "synthetic_training_count": 0,
            "label_counts": {"bot": int(y.sum()), "human": int(np.sum(y == 0))},
            "dates": unique_dates,
            "model_config": asdict(final_model_config),
            "sample_date_power": float(args.sample_date_power),
            "canonicalize_mode": args.canonicalize,
            "selection_dates": selection_dates,
            "locked_holdout_dates": holdout_dates,
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, artifact_path, compress=3)
        artifact_written = True
        print(f"Saved V4 artifact: {artifact_path}")

    def json_selected(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in row.items()
            if key != "shortlist_index"
        }

    report = {
        "artifact": str(artifact_path) if artifact_written else None,
        "artifact_written": artifact_written,
        "data": str(data_path),
        "data_sha256": data_sha256,
        "training_count": len(chunks),
        "duplicate_count": duplicate_count,
        "feature_count": len(FEATURE_NAMES),
        "feature_schema_sha256": _feature_schema_sha256(),
        "feature_implementation_sha256": _feature_implementation_sha256(),
        "detector_runtime_sha256": DETECTOR_RUNTIME_SHA256,
        "runtime_library_versions": dict(RUNTIME_LIBRARY_VERSIONS),
        "branch_names": list(BRANCH_NAMES),
        "selection_dates": selection_dates,
        "locked_holdout_dates": holdout_dates,
        "selected": json_selected(selected),
        "mapper": mapper,
        "cv_model_config": asdict(cv_model_config),
        "final_model_config": asdict(final_model_config),
        "sample_date_power": float(args.sample_date_power),
        "canonicalize_mode": args.canonicalize,
        "selection_eval_40": selection_eval_40,
        "selection_eval_100": selection_eval_100,
        "locked_holdout_eval_40": holdout_eval_40,
        "locked_holdout_eval_100": holdout_eval_100,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved V4 report:   {report_path}")


if __name__ == "__main__":
    main()
