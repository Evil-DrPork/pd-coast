"""Evaluate v3 on labeled data or produce diagnostics for hidden-label data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .inference import Poker44V3Detector
from .metrics import format_metrics, metrics, simulate_windows
from .schema import canonicalize_chunks, load_chunks


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="Optional JSON report/predictions.")
    ap.add_argument("--canonicalize", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--window-simulations", type=int, default=3000)
    ap.add_argument("--stability-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--no-batch-map", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    chunks = canonicalize_chunks(load_chunks(args.data), repo_root, args.canonicalize)
    detector = Poker44V3Detector.load(args.model)
    hands = [c.hands for c in chunks]
    scores, diagnostics = detector.predict_chunks(
        hands, batch_map=not args.no_batch_map, return_diagnostics=True
    )
    score_arr = np.asarray(scores, float)
    report = {"count": len(chunks), "diagnostics": diagnostics, "predictions": scores}

    # Exact permutation-invariance check on a deterministic sample.
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(chunks), min(args.stability_samples, len(chunks)), replace=False)
    original = [hands[i] for i in sample_idx]
    shuffled = []
    for chunk in original:
        order = rng.permutation(len(chunk))
        shuffled.append([chunk[i] for i in order])
    p1 = np.asarray(detector.predict_chunks(original, batch_map=False))
    p2 = np.asarray(detector.predict_chunks(shuffled, batch_map=False))
    report["permutation_max_abs_delta"] = float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0

    labels = [c.label for c in chunks]
    if all(v is not None for v in labels):
        y = np.asarray(labels, int)
        m = metrics(y, score_arr)
        windows = simulate_windows(y, score_arr, repetitions=args.window_simulations, seed=args.seed)
        report["metrics"] = m; report["window_metrics"] = windows
        print(format_metrics(m))
        print("40-window reward:", {k: round(v, 4) for k, v in windows.items()})
    else:
        print("Unlabeled dataset: accuracy/reward cannot be computed.")
        print("Prediction summary:", {
            "min": round(float(score_arr.min()), 4), "mean": round(float(score_arr.mean()), 4),
            "max": round(float(score_arr.max()), 4), "above_0.5": int((score_arr >= 0.5).sum()),
        })
    print(f"Permutation max |delta|: {report['permutation_max_abs_delta']:.12g}")
    if args.out:
        path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {path}")


if __name__ == "__main__":
    main()

