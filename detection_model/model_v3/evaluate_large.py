"""Evaluate v3 on synthetic 80–100-hand chunks made from held-out chunks.

This is a length-shift stress test. Merged chunks contain only same-label source
chunks, but source identity is unavailable, so the result must not be presented
as a perfect reconstruction of a real single-player chunk.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .inference import Poker44V3Detector
from .metrics import format_metrics, metrics
from .schema import Chunk, canonicalize_chunks, load_chunks


def build_merged_batch(
    chunks: Sequence[Chunk],
    *,
    batch_size: int,
    bot_fraction: float,
    sources_per_chunk: int,
    min_hands: int,
    max_hands: int,
    rng: np.random.Generator,
) -> Tuple[List[List[dict]], np.ndarray, Dict[str, float]]:
    """Build one balanced batch without reusing a source chunk in that batch."""
    n_bot = int(round(batch_size * bot_fraction))
    n_human = batch_size - n_bot
    wanted = {0: n_human, 1: n_bot}
    by_label = {
        label: np.asarray([i for i, c in enumerate(chunks) if c.label == label], dtype=int)
        for label in (0, 1)
    }
    for label in (0, 1):
        required = wanted[label] * sources_per_chunk
        if len(by_label[label]) < required:
            raise ValueError(
                f"need {required} unique label={label} source chunks for one batch, "
                f"but only {len(by_label[label])} are available"
            )
        rng.shuffle(by_label[label])

    merged: List[List[dict]] = []
    labels: List[int] = []
    target_lengths: List[int] = []
    cursors = {0: 0, 1: 0}
    label_order = np.asarray([0] * n_human + [1] * n_bot, dtype=int)
    rng.shuffle(label_order)

    for label in label_order:
        start = cursors[int(label)]
        selected = by_label[int(label)][start : start + sources_per_chunk]
        cursors[int(label)] += sources_per_chunk
        pool = [hand for idx in selected for hand in chunks[int(idx)].hands]
        if len(pool) < min_hands:
            raise ValueError(
                f"selected {sources_per_chunk} chunks contain only {len(pool)} hands; "
                f"need at least {min_hands}"
            )
        requested = int(rng.integers(min_hands, max_hands + 1))
        target = min(requested, len(pool))
        chosen = rng.choice(len(pool), size=target, replace=False)
        hands = [pool[int(i)] for i in chosen]
        rng.shuffle(hands)
        merged.append(hands)
        labels.append(int(label))
        target_lengths.append(target)

    return merged, np.asarray(labels, int), {
        "min_hands": float(min(target_lengths)),
        "mean_hands": float(np.mean(target_lengths)),
        "max_hands": float(max(target_lengths)),
        "unique_sources_used": float(batch_size * sources_per_chunk),
    }


def build_small_batch(
    chunks: Sequence[Chunk], *, batch_size: int, bot_fraction: float, rng: np.random.Generator
) -> Tuple[List[List[dict]], np.ndarray]:
    n_bot = int(round(batch_size * bot_fraction)); n_human = batch_size - n_bot
    selected = []
    for label, count in ((0, n_human), (1, n_bot)):
        candidates = np.asarray([i for i, c in enumerate(chunks) if c.label == label], dtype=int)
        if len(candidates) < count:
            raise ValueError(f"not enough label={label} chunks for small baseline")
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return [chunks[i].hands for i in selected], np.asarray([chunks[i].label for i in selected], int)


def _summary(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, float)
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)), "p90": float(np.quantile(a, 0.90)),
        "min": float(a.min()), "max": float(a.max()), "std": float(a.std()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Combined labeled benchmark containing held-out dates.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="detection_model/artifacts/p44_v3_large_eval.json")
    ap.add_argument("--min-date", default=None)
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--canonicalize", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--bot-fraction", type=float, default=0.50)
    ap.add_argument("--sources-per-chunk", type=int, default=3)
    ap.add_argument("--min-hands", type=int, default=80)
    ap.add_argument("--max-hands", type=int, default=100)
    ap.add_argument("--repetitions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--allow-date-overlap", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.sources_per_chunk < 2:
        raise SystemExit("batch-size must be positive and sources-per-chunk must be >=2")
    if not (0 < args.bot_fraction < 1) or args.min_hands > args.max_hands:
        raise SystemExit("invalid bot fraction or hand range")

    repo_root = Path(__file__).resolve().parents[2]
    chunks = canonicalize_chunks(load_chunks(args.data), repo_root, args.canonicalize)
    chunks = [c for c in chunks if c.label is not None]
    if args.min_date:
        chunks = [c for c in chunks if c.source_date and c.source_date >= args.min_date]
    if args.max_date:
        chunks = [c for c in chunks if c.source_date and c.source_date <= args.max_date]
    if not chunks:
        raise SystemExit("no labeled chunks remain after date filtering")

    detector = Poker44V3Detector.load(args.model)
    train_dates = set(detector.artifact.get("dates") or [])
    eval_dates = set(c.source_date for c in chunks if c.source_date)
    overlap = sorted(train_dates & eval_dates)
    if overlap and not args.allow_date_overlap:
        raise SystemExit(
            f"training/evaluation date overlap would leak: {overlap}; use a walk-forward artifact"
        )

    print(f"Evaluation sources: {len(chunks)} | labels={dict(Counter(c.label for c in chunks))} "
          f"| dates={sorted(eval_dates)}")
    rng = np.random.default_rng(args.seed)
    large_rewards: List[float] = []
    small_rewards: List[float] = []
    large_ap: List[float] = []
    large_low_fpr: List[float] = []
    length_means: List[float] = []
    first_large = None

    for rep in range(args.repetitions):
        merged, y_large, size_stats = build_merged_batch(
            chunks, batch_size=args.batch_size, bot_fraction=args.bot_fraction,
            sources_per_chunk=args.sources_per_chunk, min_hands=args.min_hands,
            max_hands=args.max_hands, rng=rng,
        )
        large_scores = np.asarray(detector.predict_chunks(merged, batch_map=True), float)
        large_m = metrics(y_large, large_scores)
        large_rewards.append(large_m["reward"]); large_ap.append(large_m["ap_score"])
        large_low_fpr.append(large_m["bot_recall"]); length_means.append(size_stats["mean_hands"])
        if first_large is None:
            first_large = {"metrics": large_m, "size_stats": size_stats}

        small, y_small = build_small_batch(
            chunks, batch_size=args.batch_size, bot_fraction=args.bot_fraction, rng=rng
        )
        small_scores = np.asarray(detector.predict_chunks(small, batch_map=True), float)
        small_rewards.append(metrics(y_small, small_scores)["reward"])

    report = {
        "model": str(args.model), "data": str(args.data),
        "evaluation_dates": sorted(eval_dates), "training_dates": sorted(train_dates),
        "batch_size": args.batch_size, "bot_fraction": args.bot_fraction,
        "sources_per_merged_chunk": args.sources_per_chunk,
        "target_hand_range": [args.min_hands, args.max_hands],
        "repetitions": args.repetitions,
        "first_large_batch": first_large,
        "large_chunk_reward": _summary(large_rewards),
        "small_chunk_reward": _summary(small_rewards),
        "large_chunk_ap": _summary(large_ap),
        "large_chunk_recall_at_fpr05": _summary(large_low_fpr),
        "realized_mean_hands": _summary(length_means),
        "mean_reward_delta_large_minus_small": float(np.mean(large_rewards) - np.mean(small_rewards)),
        "limitation": (
            "Each merged chunk combines same-label sources because player/source identity is unavailable; "
            "this is a length/domain stress test, not proof of real single-source performance."
        ),
    }
    print("First merged batch:", format_metrics(first_large["metrics"]))
    print("Large reward:", {k: round(v, 4) for k, v in report["large_chunk_reward"].items()})
    print("Small reward:", {k: round(v, 4) for k, v in report["small_chunk_reward"].items()})
    print("Mean delta large-small:", round(report["mean_reward_delta_large_minus_small"], 4))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()

