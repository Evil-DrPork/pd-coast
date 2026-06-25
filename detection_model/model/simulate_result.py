from __future__ import annotations

import argparse
import time
import csv
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dataset import ChunkSample
from .inference import Poker44BotDetector


# =========================================================
# Args
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on chunk_dump.json (with optional label check)."
    )

    parser.add_argument("--data", required=True, help="Path to chunk_dump.json")
    parser.add_argument("--model", required=True, help="Path to .pt model")

    parser.add_argument("--xgb-model", default=None)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--show", type=int, default=40)

    # window inference
    parser.add_argument("--window-inference", action="store_true")
    parser.add_argument("--window-hands", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument(
        "--window-agg",
        choices=["mean", "max", "topk_mean"],
        default="mean",
    )
    parser.add_argument("--keep-short-window-chunks", action="store_true")
    parser.add_argument("--out-json", default=None)

    return parser.parse_args()


# =========================================================
# Loader
# =========================================================

def load_chunk_dump(path: str) -> List[ChunkSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[ChunkSample] = []
    if data and isinstance(data, dict):
        if "labeled_chunks" in data:
            data = data["labeled_chunks"]

    for i, chunk in enumerate(data):
        samples.append(
            ChunkSample(
                chunk=chunk["hands"] if "hands" in chunk else chunk,
                label=chunk.get("is_bot", -1) if isinstance(chunk, dict) else -1,  # true label
            )
        )

    return samples


# =========================================================
# Model loader
# =========================================================

def load_detector(
    model_path: str,
    device: str,
    xgb_model_path: Optional[str],
) -> Poker44BotDetector:

    load_fn = Poker44BotDetector.load
    sig = inspect.signature(load_fn)

    kwargs: Dict[str, Any] = {"device": device}

    if xgb_model_path and "xgb_path" in sig.parameters:
        kwargs["xgb_path"] = xgb_model_path

    detector = load_fn(model_path, **kwargs)

    if xgb_model_path and not getattr(detector, "xgb_model", None):
        import joblib
        payload = joblib.load(xgb_model_path)
        detector.xgb_model = payload["xgb_model"]

    return detector


# =========================================================
# Window logic
# =========================================================

def make_windows_for_chunk(
    chunk: List[Dict[str, Any]],
    window_hands: int,
    window_stride: int,
    keep_short: bool,
) -> List[List[Dict[str, Any]]]:

    if not chunk:
        return []

    n = len(chunk)

    if n < window_hands:
        return [chunk] if keep_short else []

    windows = []
    for start in range(0, n - window_hands + 1, window_stride):
        windows.append(chunk[start:start + window_hands])

    # last_start = n - window_hands
    # if len(windows) == 0 or (start != last_start):
    #     windows.append(chunk[last_start:last_start + window_hands])

    return windows


def aggregate_scores(scores: List[float], method: str) -> float:
    if not scores:
        return 0.5

    scores = [float(s) for s in scores]

    if method == "max":
        return max(scores)

    if method == "topk_mean":
        k = min(3, len(scores))
        return sum(sorted(scores, reverse=True)[:k]) / k

    return sum(scores) / len(scores)


def predict_with_windows(
    detector: Poker44BotDetector,
    samples: List[ChunkSample],
    batch_size: int,
    window_hands: int,
    window_stride: int,
    window_agg: str,
    keep_short: bool,
) -> Tuple[List[float], List[int]]:

    all_windows = []
    ranges = []
    counts = []

    cursor = 0

    for sample in samples:
        windows = make_windows_for_chunk(
            sample.chunk,
            window_hands,
            window_stride,
            keep_short,
        )

        if not windows:
            windows = [sample.chunk]

        start = cursor
        all_windows.extend(windows)
        cursor += len(windows)
        end = cursor

        ranges.append((start, end))
        counts.append(len(windows))

    window_scores = detector.predict_chunks(all_windows, batch_size=batch_size)

    final_scores = []
    for s, e in ranges:
        final_scores.append(
            aggregate_scores(window_scores[s:e], window_agg)
        )

    return final_scores, counts

def save_mismatches_json(
    path: str,
    samples: List[ChunkSample],
    rows: List[Dict[str, Any]],
    min_chunks: int = 40,
) -> None:
    chunks = []

    for sample, row in zip(samples, rows):
        true_label = int(sample.label)

        if true_label < 0:
            true_label = row["prediction"]

        score = float(row["score"])
        diff = abs(score - true_label)

        chunks.append({
            "hands": sample.chunk,
            "is_bot": true_label,
            "score": score,
            "prediction": row["prediction"],
            "diff": diff,
        })

    # hardest first
    chunks.sort(key=lambda x: x["diff"], reverse=True)

    half = min_chunks // 2

    bot_chunks = [x for x in chunks if x["is_bot"] == 1][:half]
    human_chunks = [x for x in chunks if x["is_bot"] == 0][:half]

    selected_chunks = bot_chunks + human_chunks

    # optional: sort again by difficulty
    selected_chunks.sort(key=lambda x: x["diff"], reverse=True)

    # remove the chunks with diff less than 1e-5
    selected_chunks = [chunk for chunk in selected_chunks if chunk["diff"] > 1e-5]

    print(f"saved {len(selected_chunks)} chunks in mismatched json file")

    output = {
        "labeled_chunks": selected_chunks
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

# =========================================================
# Output
# =========================================================

def build_rows(
    samples: List[ChunkSample],
    scores: List[float],
    threshold: float,
    window_counts: Optional[List[int]],
) -> List[Dict[str, Any]]:

    rows = []

    for i, (sample, score) in enumerate(zip(samples, scores)):
        pred = int(score >= threshold)
        true = int(sample.label)

        if true < 0:
            true = pred  # if no true label, treat as correct prediction for metrics and sorting

        row = {
            "idx": i,
            "score": float(score),
            "prediction": pred,
            "prediction_name": "bot" if pred else "human",
            "true_label": true,
            "true_name": "bot" if true else "human",
            "is_mismatch": int(pred != true),
            "chunk_size_hands": len(sample.chunk),
        }

        if window_counts:
            row["num_windows"] = window_counts[i]

        rows.append(row)

    return rows


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = parse_args()

    print(f"Loading chunk dump: {args.data}")
    samples = load_chunk_dump(args.data)

    print(f"Loaded chunks: {len(samples)}")

    detector = load_detector(
        args.model,
        args.device,
        args.xgb_model,
    )

    threshold = args.threshold

    if args.window_inference:
        scores, window_counts = predict_with_windows(
            detector,
            samples,
            args.batch_size,
            args.window_hands,
            args.window_stride,
            args.window_agg,
            args.keep_short_window_chunks,
        )
    else:
        chunks = [s.chunk for s in samples]
        scores = detector.predict_chunks(chunks, batch_size=args.batch_size)
        window_counts = None

    rows = build_rows(samples, scores, threshold, window_counts)

    if args.out_json:
        save_mismatches_json(args.out_json, samples, rows)
        print(f"Saved mismatches JSON: {args.out_json}")

    # sort mismatches first, then by score
    rows = sorted(
        rows,
        key=lambda r: (r["is_mismatch"], r["score"]),
        reverse=True,
    )

    # mismatch summary
    total = len(rows)
    diff_count = sum(r["is_mismatch"] for r in rows)

    print(f"\n=== Mismatch Summary ===")
    print(f"{diff_count} / {total} mismatches ({diff_count/total:.2%})")

    print(f"\n=== Top {args.show} rows (mismatches first) ===")
    for r in rows[:args.show]:
        extra = f" windows={r['num_windows']}" if "num_windows" in r else ""
        mismatch = "❌" if r["is_mismatch"] else "✔"

        print(
            f"{r['idx']} "
            f"{mismatch} "
            f"score={r['score']:.6f} "
            f"pred={r['prediction_name']} "
            f"true={r['true_name']} "
            f"hands={r['chunk_size_hands']}"
            f"{extra}"
        )

    if args.out_csv:
        save_csv(args.out_csv, rows)
        print(f"Saved CSV: {args.out_csv}")
    
    max_diff = max(abs(r["score"] - r["true_label"]) for r in rows)
    print(f"\n=== Max Diff ===")
    print(f"{max_diff:.6f}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Execution time: {time.perf_counter() - start:.3f}s")