"""Validator-aligned metrics — mirrors poker44/score/scoring.py (2026-07 formula).

    reward = 0.35*AP + 0.30*recall@(fpr<=0.05) + 0.20*S + 0.10*S + 0.05*latency
      S = threshold_sanity_quality at the 0.5 boundary;  reward = 0 if S <= 0.

* **AP** and **recall@fpr<=0.05** are *rank-based* (threshold swept) — they do NOT
  depend on the absolute score scale, only the ordering. Together = 65% of reward.
* **S** (threshold sanity, 30%) is the only calibration-sensitive term: the model
  MUST cross 0.5 on the labeled window (>=1 true positive) and keep hard fpr@0.5
  <= 0.10 for full credit; it decays linearly above 0.10 and hard-zeros the whole
  reward if nothing crosses 0.5.
* **latency** (5%) is a constant 1.0 placeholder here.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

AP_WEIGHT = 0.35
BOT_RECALL_WEIGHT = 0.30
HUMAN_SAFETY_WEIGHT = 0.20
CALIBRATION_WEIGHT = 0.10
LATENCY_WEIGHT = 0.05
SANITY_FPR = 0.10          # hard fpr@0.5 giving full threshold-sanity credit


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.05) -> Tuple[float, float]:
    """Best bot recall reachable while sweeping the threshold with fpr <= max_fpr."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos <= 0 or neg <= 0 or scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    tp = np.cumsum(sl == 1)
    fp = np.cumsum(sl == 0)
    recall = tp / max(pos, 1)
    fpr = fp / max(neg, 1)
    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0
    ai = np.flatnonzero(allowed)
    best = int(ai[np.argmax(recall[allowed])])
    return float(recall[best]), float(fpr[best])


def _threshold_sanity(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    hp = scores >= float(threshold)
    ppr = float(hp.mean())
    tp = int((hp & (labels == 1)).sum())
    fp = int((hp & (labels == 0)).sum())
    hard_recall = tp / max(pos, 1) if pos > 0 else 0.0
    hard_fpr = fp / max(neg, 1) if neg > 0 else 0.0
    if pos <= 0 or neg <= 0:
        sanity = 1.0
    elif tp <= 0:
        sanity = 0.0
    elif hard_fpr <= SANITY_FPR:
        sanity = 1.0
    else:
        sanity = max(0.0, 1.0 - (hard_fpr - SANITY_FPR) / (1.0 - SANITY_FPR))
    return hard_recall, hard_fpr, ppr, sanity


def reward_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    both = len(set(y.tolist())) > 1
    ap = float(average_precision_score(y, s)) if (s.size and (y == 1).any()) else 0.0
    bot_recall, best_fpr = recall_at_fpr(s, y, max_fpr=0.05)
    hard_recall, hard_fpr, ppr, sanity = _threshold_sanity(s, y, 0.5)
    latency = 1.0

    if sanity <= 0.0:
        base = 0.0
        reward = 0.0
    else:
        base = (
            AP_WEIGHT * ap
            + BOT_RECALL_WEIGHT * bot_recall
            + HUMAN_SAFETY_WEIGHT * sanity
            + CALIBRATION_WEIGHT * sanity
            + LATENCY_WEIGHT * latency
        )
        reward = float(np.clip(base, 0.0, 1.0))

    return {
        "reward": reward,
        "ap": ap,
        "bot_recall_at_fpr005": bot_recall,
        "fpr_at_best": best_fpr,
        "threshold_sanity": sanity,
        "recall_at_0.5": hard_recall,
        "fpr_at_0.5": hard_fpr,
        "positive_rate": ppr,
        "roc_auc": float(roc_auc_score(np.clip(y, 0, 1), s)) if both else 0.0,
        "base_score": float(base),
    }


def format_metrics(m: Dict[str, float]) -> str:
    return (
        f"reward={m['reward']:.4f} ap={m['ap']:.4f} "
        f"recall@fpr05={m['bot_recall_at_fpr005']:.4f} sanity={m['threshold_sanity']:.3f} "
        f"recall@0.5={m['recall_at_0.5']:.4f} fpr@0.5={m['fpr_at_0.5']:.4f}"
    )
