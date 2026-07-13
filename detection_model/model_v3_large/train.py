"""Train the separate large-chunk tabular challenger.

The stable ``detection_model.model_v3`` implementation is imported but never
modified.  Artifacts remain compatible with its production detector/miner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from detection_model.model_v3.calibration import apply_batch_mapper, apply_mapper, fit_fixed_mapper
from detection_model.model_v3.features import FEATURE_NAMES, matrix_for_chunks
from detection_model.model_v3.metrics import format_metrics, metrics, simulate_windows
from detection_model.model_v3.schema import Chunk, canonicalize_chunks, chunk_multiset_hash, load_chunks

from .augmentation import LargeAugmentationConfig, generate_merged_chunks
from .model import LargeChunkTabularEnsemble, choose_large_weights


def _deduplicate(chunks: List[Chunk]) -> Tuple[List[Chunk], int]:
    seen = {}
    out = []
    conflicts = 0
    for chunk in chunks:
        digest = chunk_multiset_hash(chunk)
        if digest in seen:
            conflicts += int(seen[digest] != chunk.label)
            continue
        seen[digest] = chunk.label
        out.append(chunk)
    if conflicts:
        raise ValueError(f"found {conflicts} duplicate chunks with conflicting labels")
    return out, len(chunks) - len(out)


def _folds(chunks: List[Chunk], y: np.ndarray, seed: int):
    dates = np.asarray([chunk.source_date or "" for chunk in chunks])
    unique_dates = sorted(set(dates) - {""})
    if len(unique_dates) >= 3:
        for date in unique_dates:
            validation = np.flatnonzero(dates == date)
            training = np.flatnonzero(dates != date)
            if len(np.unique(y[validation])) == 2 and len(np.unique(y[training])) == 2:
                yield f"date={date}", training, validation
        return
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (training, validation) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        yield f"stratified={fold}", training, validation


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="detection_model/artifacts/p44_v3_large.joblib")
    ap.add_argument("--report", default="detection_model/artifacts/p44_v3_large_train_report.json")
    ap.add_argument("--canonicalize", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--seed", type=int, default=144)
    ap.add_argument("--batch-top-fraction", type=float, default=0.10)
    ap.add_argument("--reward-window", type=int, default=100)
    ap.add_argument("--window-simulations", type=int, default=3000)
    ap.add_argument("--large-ratio", type=float, default=1.25)
    ap.add_argument("--medium-ratio", type=float, default=0.40)
    ap.add_argument("--sources-per-merge", type=int, default=3)
    ap.add_argument("--large-min-hands", type=int, default=80)
    ap.add_argument("--large-max-hands", type=int, default=100)
    ap.add_argument("--medium-min-hands", type=int, default=50)
    ap.add_argument("--medium-max-hands", type=int, default=75)
    ap.add_argument("--subbag-target-hands", type=int, default=34)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.batch_top_fraction < 1.0):
        raise SystemExit("batch-top-fraction must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = LargeAugmentationConfig(
        large_ratio=args.large_ratio,
        medium_ratio=args.medium_ratio,
        sources_per_merge=args.sources_per_merge,
        large_min_hands=args.large_min_hands,
        large_max_hands=args.large_max_hands,
        medium_min_hands=args.medium_min_hands,
        medium_max_hands=args.medium_max_hands,
        subbag_target_hands=args.subbag_target_hands,
    )

    repo_root = Path(__file__).resolve().parents[2]
    chunks = canonicalize_chunks(load_chunks(args.data), repo_root, args.canonicalize)
    if args.max_date:
        chunks = [chunk for chunk in chunks if chunk.source_date and chunk.source_date <= args.max_date]
    if not chunks or any(chunk.label is None for chunk in chunks):
        raise SystemExit("large-chunk training requires labeled chunks")
    chunks, duplicate_count = _deduplicate(chunks)
    y = np.asarray([int(chunk.label) for chunk in chunks], dtype=int)
    if len(np.unique(y)) != 2:
        raise SystemExit("large-chunk training requires both labels")
    hands = [chunk.hands for chunk in chunks]
    x = matrix_for_chunks(hands)
    print(
        f"Loaded {len(chunks)} unique chunks ({duplicate_count} duplicates removed) | "
        f"bot={int(y.sum())} human={int((y == 0).sum())} features={x.shape[1]}",
        flush=True,
    )
    print("Dates:", dict(Counter(chunk.source_date or "unknown" for chunk in chunks)), flush=True)
    print("Large augmentation:", config.as_dict(), flush=True)

    channel_count = len(LargeChunkTabularEnsemble.channel_names)
    oof = np.zeros((len(y), channel_count), dtype=float)
    oof_count = np.zeros(len(y), dtype=int)
    merged_oof_branches = []
    merged_oof_labels = []
    fold_rows = []

    for fold_index, (name, training, validation) in enumerate(_folds(chunks, y, args.seed), 1):
        print(
            f"[{name}] fitting large challenger | train={len(training)} validation={len(validation)}",
            flush=True,
        )
        model = LargeChunkTabularEnsemble(args.seed + fold_index, config).fit(
            [hands[i] for i in training], y[training]
        )
        original_branches = model.branch_scores(x[validation], [hands[i] for i in validation])
        oof[validation] = original_branches
        oof_count[validation] += 1

        rng = np.random.default_rng(args.seed + 10000 + fold_index)
        merged_chunks, merged_labels = generate_merged_chunks(
            [hands[i] for i in validation],
            y[validation],
            count=max(100, len(validation)),
            min_hands=config.large_min_hands,
            max_hands=config.large_max_hands,
            sources_per_merge=config.sources_per_merge,
            rng=rng,
        )
        merged_x = matrix_for_chunks(merged_chunks)
        merged_oof_branches.append(model.branch_scores(merged_x, merged_chunks))
        merged_oof_labels.append(merged_labels)

        default_scores = original_branches @ model.branch_weights_
        row = {
            "fold": name,
            "train": len(training),
            "validation": len(validation),
            "augmentation": model.augmentation_stats_,
            **metrics(y[validation], default_scores),
        }
        fold_rows.append(row)
        print(f"[{name}] {format_metrics(row)} | views={model.augmentation_stats_}", flush=True)

    if np.any(oof_count != 1):
        raise RuntimeError("OOF split failed to cover each chunk exactly once")
    merged_branches = np.concatenate(merged_oof_branches)
    merged_y = np.concatenate(merged_oof_labels)
    weights, selection = choose_large_weights(
        oof,
        y,
        metrics,
        merged_branches=merged_branches,
        merged_y=merged_y,
    )
    original_raw = oof @ weights
    merged_raw = merged_branches @ weights
    mapper = fit_fixed_mapper(original_raw, y, target_human_fpr=0.05)
    original_metrics = metrics(y, apply_mapper(original_raw, mapper))
    merged_rank_metrics = metrics(merged_y, merged_raw)
    original_windows = simulate_windows(
        y,
        original_raw,
        window=args.reward_window,
        repetitions=args.window_simulations,
        seed=args.seed,
        score_transform=lambda score: apply_batch_mapper(score, args.batch_top_fraction),
    )
    merged_windows = simulate_windows(
        merged_y,
        merged_raw,
        window=args.reward_window,
        repetitions=args.window_simulations,
        seed=args.seed + 1,
        score_transform=lambda score: apply_batch_mapper(score, args.batch_top_fraction),
    )
    print("OOF weights:", np.round(weights, 3).tolist(), flush=True)
    print("View selection:", selection.as_dict(), flush=True)
    print("Original OOF:", format_metrics(original_metrics), flush=True)
    print("Merged OOF ranking:", format_metrics(merged_rank_metrics), flush=True)
    print("Original 100-window:", {k: round(v, 4) for k, v in original_windows.items()}, flush=True)
    print("Merged 100-window:", {k: round(v, 4) for k, v in merged_windows.items()}, flush=True)

    print("Fitting final large-chunk challenger...", flush=True)
    final_model = LargeChunkTabularEnsemble(args.seed, config).fit(hands, y)
    final_model.branch_weights_ = weights
    artifact = {
        "artifact_version": 3,
        "architecture": "tabular_large_multiscale_v1",
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "mapper": mapper,
        "batch_top_fraction": float(args.batch_top_fraction),
        "reward_window": int(args.reward_window),
        "canonicalizer_mode": args.canonicalize,
        "training_data": str(args.data),
        "training_count": int(len(y)),
        "label_counts": {"bot": int(y.sum()), "human": int((y == 0).sum())},
        "dates": sorted(set(chunk.source_date for chunk in chunks if chunk.source_date)),
        "augmentation_config": config.as_dict(),
        "augmentation_stats": final_model.augmentation_stats_,
        "channel_names": list(final_model.channel_names),
        "weight_selection": selection.as_dict(),
        "oof_metrics": original_metrics,
        "merged_oof_metrics": merged_rank_metrics,
        "window_metrics": original_windows,
        "merged_window_metrics": merged_windows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)

    report = {
        "artifact": str(out),
        "architecture": artifact["architecture"],
        "feature_count": int(x.shape[1]),
        "duplicate_count": duplicate_count,
        "weights": weights.tolist(),
        "channel_names": list(final_model.channel_names),
        "weight_selection": selection.as_dict(),
        "augmentation_config": config.as_dict(),
        "augmentation_stats": final_model.augmentation_stats_,
        "mapper": mapper,
        "oof_metrics": original_metrics,
        "merged_oof_metrics": merged_rank_metrics,
        "window_metrics": original_windows,
        "merged_window_metrics": merged_windows,
        "folds": fold_rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"Saved challenger: {out}", flush=True)
    print(f"Saved report:     {report_path}", flush=True)
    print(f"Artifact SHA256:  {sha}", flush=True)


if __name__ == "__main__":
    main()
