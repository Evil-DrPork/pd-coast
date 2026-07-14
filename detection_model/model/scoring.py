"""Compatibility metrics backed by the current authoritative Poker44 reward.

Historically this module copied an older validator formula. Keeping two reward
implementations allowed training and evaluation to optimize a stale objective,
so all current scoring is now delegated to :mod:`poker44.score.scoring`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from poker44.score.scoring import (
    AP_WEIGHT,
    BOT_RECALL_WEIGHT,
    CALIBRATION_WEIGHT,
    HUMAN_SAFETY_WEIGHT,
    LATENCY_WEIGHT,
    _threshold_metrics,
    reward as authoritative_reward,
)

# Retained for older calibration imports. Under the current scorer 0.10 is the
# start of a gradual threshold-sanity penalty, not the former all-or-zero cliff.
VALIDATOR_FPR_CLIFF = 0.10


def validator_reward(
    scores: np.ndarray,
    labels: np.ndarray,
    boundary: Optional[float] = None,
) -> tuple[float, Dict[str, float]]:
    """Return the current validator reward, optionally simulating a boundary."""
    values = np.asarray(scores, dtype=float)
    truth = np.asarray(labels, dtype=int)
    reward_value, details = authoritative_reward(values, truth)
    if boundary is None or float(boundary) == 0.5:
        return float(reward_value), {key: float(value) for key, value in details.items()}

    threshold = _threshold_metrics(values, truth, threshold=float(boundary))
    safety = float(threshold["threshold_sanity_quality"])
    calibration = safety
    if safety <= 0.0:
        base_score = 0.0
        reward_value = 0.0
    else:
        base_score = (
            AP_WEIGHT * float(details["ap_score"])
            + BOT_RECALL_WEIGHT * float(details["bot_recall"])
            + HUMAN_SAFETY_WEIGHT * safety
            + CALIBRATION_WEIGHT * calibration
            + LATENCY_WEIGHT
        )
        reward_value = float(np.clip(base_score, 0.0, 1.0))
    updated = dict(details)
    updated.update(threshold)
    updated.update(
        human_safety_penalty=safety,
        calibration_quality=calibration,
        base_score=base_score,
        reward=reward_value,
    )
    return reward_value, {key: float(value) for key, value in updated.items()}


def reward_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    boundary: Optional[float] = None,
) -> Dict[str, float]:
    """Return current reward components plus score-separation diagnostics."""
    truth = np.asarray([int(value) for value in labels], dtype=int)
    values = np.clip(np.asarray([float(value) for value in scores]), 0.0, 1.0)
    reward_value, details = validator_reward(values, truth, boundary=boundary)
    metrics: Dict[str, float] = {
        "validator_reward": reward_value,
        "validator_fpr": details["fpr"],
        "validator_bot_recall": details["bot_recall"],
        "validator_ap_score": details["ap_score"],
        "validator_base_score": details["base_score"],
        "human_safety_penalty": details["human_safety_penalty"],
        "hard_fpr": details["hard_fpr"],
        "hard_bot_recall": details["hard_bot_recall"],
        "positive_prediction_rate": details["positive_prediction_rate"],
    }
    humans = values[truth == 0]
    bots = values[truth == 1]
    metrics["human_prob_max"] = float(humans.max()) if humans.size else 0.0
    metrics["bot_prob_min"] = float(bots.min()) if bots.size else 1.0
    metrics["score_gap_at_0_5"] = metrics["bot_prob_min"] - metrics["human_prob_max"]
    metrics["prob_min"] = float(values.min()) if values.size else 0.0
    metrics["prob_max"] = float(values.max()) if values.size else 0.0
    metrics["prob_mean"] = float(values.mean()) if values.size else 0.0
    return metrics


def format_reward_line(metrics: Dict[str, float]) -> str:
    return (
        f"reward={metrics.get('validator_reward', 0.0):.4f} "
        f"ap={metrics.get('validator_ap_score', 0.0):.4f} "
        f"recall@5%fpr={metrics.get('validator_bot_recall', 0.0):.4f} "
        f"fpr={metrics.get('validator_fpr', 0.0):.4f} "
        f"hard_recall={metrics.get('hard_bot_recall', 0.0):.4f} "
        f"hard_fpr={metrics.get('hard_fpr', 0.0):.4f}"
    )


def print_reward_diagnostics(
    title: str,
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    indent: str = "  ",
) -> Dict[str, float]:
    metrics = reward_metrics(labels, scores)
    print(f"{indent}{title}: {format_reward_line(metrics)}")
    return metrics
