"""Paired chronological promotion evaluation for V4 versus V3 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from detection_model.model_v3.inference import Poker44V3Detector
from detection_model.model_v3.schema import canonicalize_chunks, load_chunks
from detection_model.model_v3_large.inference import Poker44V3LargeDetector

from .features import FEATURE_IMPLEMENTATION_SHA256, FEATURE_NAMES, FEATURE_SCHEMA_SHA256
from .mapping import chunk_tie_key, exact_rank_map
from .model import CoherentEnsemble, ModelConfig, blend_branches
from .provenance import DETECTOR_RUNTIME_SHA256, RUNTIME_LIBRARY_VERSIONS
from .train import (
    _balanced_windows,
    _batched_request_branches,
    _date_weights,
    _deduplicate,
    _metric,
    _sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--v4-report", required=True)
    parser.add_argument("--stable-010", required=True)
    parser.add_argument("--stable-015", required=True)
    parser.add_argument("--large", required=True)
    parser.add_argument("--out", default="detection_model/artifacts/p44_v4_holdout_comparison.json")
    parser.add_argument("--windows-40", type=int, default=300)
    parser.add_argument("--windows-100", type=int, default=150)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "std": float(array.std()),
        "zero_rate": float(np.mean(array <= 0.0)),
    }


def _paired(candidate: Sequence[float], champion: Sequence[float]) -> dict[str, float]:
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(champion, dtype=float)
    delta = left - right
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "p10_delta": float(np.quantile(delta, 0.10)),
        "win_rate": float(np.mean(delta > 1e-12)),
        "tie_rate": float(np.mean(np.abs(delta) <= 1e-12)),
    }


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.v4_report).read_text(encoding="utf-8"))
    selected = report["selected"]
    selected_weights = np.asarray(selected["weights"], dtype=float)
    selected_mode = str(selected["mode"])
    selected_fraction = float(selected["fraction"])
    holdout_dates = [str(value) for value in report["locked_holdout_dates"]]
    cutoff = min(holdout_dates)
    canonicalize_mode = str(report.get("canonicalize_mode", "auto"))

    repo_root = Path(__file__).resolve().parents[2]
    chunks = canonicalize_chunks(load_chunks(args.data), repo_root, canonicalize_mode)
    chunks, duplicate_count = _deduplicate(chunks)
    labels = np.asarray([int(chunk.label) for chunk in chunks], dtype=int)
    dates = np.asarray([str(chunk.source_date) for chunk in chunks])
    with np.load(args.feature_cache, allow_pickle=False) as cache:
        x = np.asarray(cache["x"], dtype=np.float64)
        cached_labels = np.asarray(cache["y"], dtype=int)
        cached_dates = np.asarray(cache["dates"], dtype=str)
        cache_data_sha256 = str(cache["data_sha256"].item())
        cache_schema_sha256 = str(cache["feature_schema_sha256"].item())
        cache_implementation_sha256 = str(cache["feature_implementation_sha256"].item())
        cache_canonicalize_mode = str(cache["canonicalize_mode"].item())
        cache_runtime_library_versions = json.loads(
            str(cache["runtime_library_versions"].item())
        )
    data_sha256 = _sha256(Path(args.data))
    metadata_matches = (
        data_sha256 == str(report["data_sha256"])
        and cache_data_sha256 == data_sha256
        and cache_schema_sha256 == FEATURE_SCHEMA_SHA256
        and cache_schema_sha256 == str(report["feature_schema_sha256"])
        and cache_implementation_sha256 == FEATURE_IMPLEMENTATION_SHA256
        and cache_implementation_sha256 == str(report["feature_implementation_sha256"])
        and cache_canonicalize_mode == canonicalize_mode
        and cache_runtime_library_versions == RUNTIME_LIBRARY_VERSIONS
        and duplicate_count == int(report.get("duplicate_count", duplicate_count))
        and str(report.get("detector_runtime_sha256", "")) == DETECTOR_RUNTIME_SHA256
        and report.get("runtime_library_versions") == RUNTIME_LIBRARY_VERSIONS
        and list(report.get("branch_names") or []) == list(CoherentEnsemble.branch_names)
        and int(report.get("feature_count", -1)) == len(FEATURE_NAMES)
    )
    matrix_matches = (
        x.shape == (len(chunks), len(FEATURE_NAMES))
        and np.array_equal(labels, cached_labels)
        and np.array_equal(dates, cached_dates)
    )
    if not metadata_matches or not matrix_matches:
        raise SystemExit("feature cache does not match evaluation dataset")

    model_config = ModelConfig(**report["final_model_config"])
    train_indices = np.flatnonzero(dates < cutoff)
    training_chunks = [chunks[index] for index in train_indices]
    v4_model = CoherentEnsemble(args.seed, model_config).fit(
        x[train_indices],
        labels[train_indices],
        sample_weight=_date_weights(training_chunks, float(report["sample_date_power"])),
        groups=dates[train_indices],
    )
    v4_model.branch_weights_ = selected_weights

    detectors: dict[str, Any] = {
        "stable_010": Poker44V3Detector.load(args.stable_010),
        "stable_015": Poker44V3Detector.load(args.stable_015),
        "large_010": Poker44V3LargeDetector.load(args.large),
    }
    all_results: dict[str, Any] = {}
    for window_size, repeats in ((40, args.windows_40), (100, args.windows_100)):
        rewards: dict[str, list[float]] = {"v4": [], **{name: [] for name in detectors}}
        per_date: dict[str, dict[str, dict[str, float]]] = {}
        for date_index, date in enumerate(holdout_dates):
            indices = np.flatnonzero(dates == date)
            date_labels = labels[indices]
            date_chunks = [chunks[index].hands for index in indices]
            keys = [chunk_tie_key(chunk) for chunk in date_chunks]
            v4_branches = v4_model.branch_scores(x[indices])
            baseline_raw = {name: detector.raw_scores(date_chunks) for name, detector in detectors.items()}

            v4_full_raw = blend_branches(
                v4_branches, selected_weights, selected_mode, tie_keys=keys
            )
            v4_full_scores = exact_rank_map(v4_full_raw, selected_fraction, tie_keys=keys)
            date_metrics: dict[str, dict[str, float]] = {"v4": _metric(date_labels, v4_full_scores)}
            for name, detector in detectors.items():
                date_metrics[name] = _metric(
                    date_labels,
                    detector.map_scores(baseline_raw[name], batch_map=True),
                )
            per_date[date] = date_metrics

            windows = _balanced_windows(
                date_labels, window_size, repeats, args.seed + 1000 * date_index + window_size
            )
            window_branches = _batched_request_branches(
                {
                    "branches": v4_branches,
                    "matrix": x[indices],
                    "model": v4_model,
                },
                windows,
            )
            for window, request_branches in zip(windows, window_branches):
                window_labels = date_labels[window]
                window_keys = [keys[index] for index in window]
                v4_raw = blend_branches(
                    request_branches,
                    selected_weights,
                    selected_mode,
                    tie_keys=window_keys,
                )
                rewards["v4"].append(
                    _metric(
                        window_labels,
                        exact_rank_map(v4_raw, selected_fraction, tie_keys=window_keys),
                    )["reward"]
                )
                for name, detector in detectors.items():
                    rewards[name].append(
                        _metric(
                            window_labels,
                            detector.map_scores(baseline_raw[name][window], batch_map=True),
                        )["reward"]
                    )

        summaries = {name: _summary(values) for name, values in rewards.items()}
        all_results[str(window_size)] = {
            "per_date": per_date,
            "summaries": summaries,
            "paired_vs_stable_010": _paired(rewards["v4"], rewards["stable_010"]),
            "paired_vs_stable_015": _paired(rewards["v4"], rewards["stable_015"]),
            "paired_vs_large_010": _paired(rewards["v4"], rewards["large_010"]),
        }
        print(
            f"n={window_size} V4={summaries['v4']['mean']:.4f} "
            f"stable.10={summaries['stable_010']['mean']:.4f} "
            f"stable.15={summaries['stable_015']['mean']:.4f} "
            f"large={summaries['large_010']['mean']:.4f}"
        )

    output = {
        "data": args.data,
        "training_cutoff_exclusive": cutoff,
        "holdout_dates": holdout_dates,
        "v4_selected": selected,
        "results": all_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved paired comparison: {out_path}")


if __name__ == "__main__":
    main()
