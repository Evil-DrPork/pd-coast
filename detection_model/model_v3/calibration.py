"""Monotone score mapping that preserves validator ranking components."""

from __future__ import annotations

from typing import Dict

import numpy as np


def fit_fixed_mapper(scores: np.ndarray, labels: np.ndarray, target_human_fpr: float = 0.05) -> Dict[str, float]:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    human = s[y == 0]
    if human.size == 0:
        cut = float(np.median(s))
    else:
        cut = float(np.quantile(human, 1.0 - target_human_fpr, method="higher"))
    spread = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
    return {"cut": cut, "scale": max(spread, 0.03), "target_human_fpr": target_human_fpr}


def apply_mapper(scores: np.ndarray, mapper: Dict[str, float]) -> np.ndarray:
    z = (np.asarray(scores, float) - float(mapper["cut"])) / float(mapper["scale"])
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))), 0.01, 0.99)


def apply_batch_mapper(scores: np.ndarray, top_fraction: float = 0.20) -> np.ndarray:
    """Monotone mapping placing the selected batch fraction above 0.5."""
    s = np.asarray(scores, float)
    if s.size < 8 or not (0 < top_fraction < 1):
        return s
    cut = float(np.quantile(s, 1.0 - top_fraction, method="linear"))
    spread = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
    z = (s - cut) / max(spread, 0.03)
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))), 0.01, 0.99)

