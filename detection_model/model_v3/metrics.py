"""Metrics imported from the validator oracle plus production-window simulation."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

try:
    from poker44.score.scoring import reward as validator_reward
except ImportError:  # running from detection_model/
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from poker44.score.scoring import reward as validator_reward


def metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    reward, detail = validator_reward(np.asarray(scores, float), np.asarray(labels, int))
    return {k: float(v) for k, v in detail.items()} | {"reward": float(reward)}


def format_metrics(m: Dict[str, float]) -> str:
    return (
        f"reward={m['reward']:.4f} ap={m['ap_score']:.4f} "
        f"recall@fpr05={m['bot_recall']:.4f} hard_recall={m['hard_bot_recall']:.4f} "
        f"hard_fpr={m['hard_fpr']:.4f} sanity={m['threshold_sanity_quality']:.3f}"
    )


def simulate_windows(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    window: int = 100,
    repetitions: int = 3000,
    seed: int = 44,
    score_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, float]:
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    if len(y) < window:
        evaluated = score_transform(s.copy()) if score_transform else s
        m = metrics(y, evaluated)
        return {"mean": m["reward"], "median": m["reward"], "p10": m["reward"], "std": 0.0}
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rewards = []
    for _ in range(repetitions):
        # Balanced windows mirror current public material. Replacement is used
        # only when a held-out fold has fewer than window/2 of one class.
        p = rng.choice(pos, window // 2, replace=len(pos) < window // 2)
        n = rng.choice(neg, window - window // 2, replace=len(neg) < window - window // 2)
        idx = np.concatenate([p, n])
        rng.shuffle(idx)
        window_scores = s[idx]
        if score_transform is not None:
            window_scores = score_transform(window_scores.copy())
        rewards.append(metrics(y[idx], window_scores)["reward"])
    a = np.asarray(rewards)
    return {
        "mean": float(a.mean()), "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)), "std": float(a.std()),
    }
