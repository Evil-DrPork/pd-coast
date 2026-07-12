"""Train Poker44 v3 with date-blocked out-of-fold model selection.

Run from the repository root, for example:
    python -m detection_model.model_v3.train --data detection_model/data_v3/benchmark_chunks_2026-07-06_to_2026-07-12.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .calibration import apply_batch_mapper, apply_mapper, fit_fixed_mapper
from .features import FEATURE_NAMES, matrix_for_chunks
from .metrics import format_metrics, metrics, simulate_windows
from .model import V3Ensemble, choose_weights
from .schema import Chunk, canonicalize_chunks, chunk_multiset_hash, load_chunks


def _deduplicate(chunks: List[Chunk]) -> Tuple[List[Chunk], int]:
    seen = {}
    out = []
    conflicts = 0
    for chunk in chunks:
        digest = chunk_multiset_hash(chunk)
        if digest in seen:
            if seen[digest] != chunk.label:
                conflicts += 1
            continue
        seen[digest] = chunk.label
        out.append(chunk)
    if conflicts:
        raise ValueError(f"found {conflicts} duplicate chunks with conflicting labels")
    return out, len(chunks) - len(out)


def _folds(chunks: List[Chunk], y: np.ndarray, seed: int):
    dates = np.asarray([c.source_date or "" for c in chunks])
    unique_dates = sorted(set(dates) - {""})
    if len(unique_dates) >= 3:
        for date in unique_dates:
            va = np.flatnonzero(dates == date)
            tr = np.flatnonzero(dates != date)
            if len(np.unique(y[va])) == 2 and len(np.unique(y[tr])) == 2:
                yield f"date={date}", tr, va
        return
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        yield f"stratified={i}", tr, va


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="detection_model/artifacts/p44_v3.joblib")
    ap.add_argument("--report", default="detection_model/artifacts/p44_v3_train_report.json")
    ap.add_argument("--canonicalize", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--max-date", default=None, help="Train only source_date <= YYYY-MM-DD (walk-forward holdout).")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--window-simulations", type=int, default=3000)
    ap.add_argument("--reward-window", type=int, default=100, help="Observed live scoring batch/window size.")
    ap.add_argument("--batch-top-fraction", type=float, default=0.20)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    repo_root = Path(__file__).resolve().parents[2]
    chunks = canonicalize_chunks(load_chunks(args.data), repo_root, args.canonicalize)
    if args.max_date:
        chunks = [c for c in chunks if c.source_date and c.source_date <= args.max_date]
        if not chunks:
            raise SystemExit(f"no chunks remain at or before --max-date {args.max_date}")
    if any(c.label is None for c in chunks):
        raise SystemExit("training requires is_bot on every chunk")
    chunks, duplicate_count = _deduplicate(chunks)
    y = np.asarray([int(c.label) for c in chunks], dtype=int)
    if len(np.unique(y)) != 2:
        raise SystemExit("training requires both bot and human chunks")

    print(f"Loaded {len(chunks)} unique chunks ({duplicate_count} duplicates removed) | "
          f"bot={int(y.sum())} human={int((y == 0).sum())}")
    print("Dates:", dict(Counter(c.source_date or "unknown" for c in chunks)))
    x = matrix_for_chunks([c.hands for c in chunks])
    print(f"Feature matrix: {x.shape}")

    oof_sum = np.zeros((len(y), 4), dtype=float)
    oof_count = np.zeros(len(y), dtype=int)
    fold_rows = []
    for fold_i, (name, tr, va) in enumerate(_folds(chunks, y, args.seed), 1):
        model = V3Ensemble(args.seed + fold_i).fit(x[tr], y[tr])
        branch = model.branch_scores(x[va])
        oof_sum[va] += branch; oof_count[va] += 1
        default = branch @ model.branch_weights_
        row = {"fold": name, "train": len(tr), "validation": len(va), **metrics(y[va], default)}
        fold_rows.append(row)
        print(f"[{name}] {format_metrics(row)}")
    if np.any(oof_count == 0):
        raise RuntimeError("OOF split failed to cover every chunk")
    oof = oof_sum / oof_count[:, None]
    weights = choose_weights(oof, y, metrics)
    oof_raw = oof @ weights
    mapper = fit_fixed_mapper(oof_raw, y, target_human_fpr=0.05)
    oof_score = apply_mapper(oof_raw, mapper)
    overall = metrics(y, oof_score)
    windows = simulate_windows(
        y, oof_raw, window=args.reward_window,
        repetitions=args.window_simulations, seed=args.seed,
        score_transform=lambda s: apply_batch_mapper(s, args.batch_top_fraction),
    )
    print("OOF weights:", np.round(weights, 3).tolist())
    print("OOF:", format_metrics(overall))
    print(f"{args.reward_window}-chunk window reward:", {k: round(v, 4) for k, v in windows.items()})

    final_model = V3Ensemble(args.seed).fit(x, y)
    final_model.branch_weights_ = weights
    artifact = {
        "artifact_version": 3,
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "mapper": mapper,
        "batch_top_fraction": float(args.batch_top_fraction),
        "reward_window": int(args.reward_window),
        "canonicalizer_mode": args.canonicalize,
        "training_data": str(args.data),
        "training_count": int(len(y)),
        "label_counts": {"bot": int(y.sum()), "human": int((y == 0).sum())},
        "dates": sorted(set(c.source_date for c in chunks if c.source_date)),
        "oof_metrics": overall,
        "window_metrics": windows,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)

    report = {
        "artifact": str(out), "feature_count": int(x.shape[1]),
        "duplicate_count": duplicate_count, "weights": weights.tolist(),
        "mapper": mapper, "oof_metrics": overall,
        "window_metrics": windows, "folds": fold_rows,
    }
    report_path = Path(args.report); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved artifact: {out}")
    print(f"Saved report:   {report_path}")


if __name__ == "__main__":
    main()
