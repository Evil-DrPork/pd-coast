"""Exact v3 chunk features with a challenger-local per-hand cache."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from detection_model.model_v3.features import (
    ACTIONS,
    FEATURE_NAMES,
    QUANTILES,
    STREETS,
    _entropy,
    hand_features,
)


class CachedChunkFeaturizer:
    """Cache hand summaries while repeated augmented views share hand objects."""

    def __init__(self) -> None:
        self._hand_cache: Dict[int, Tuple[Dict[str, float], Counter, Counter]] = {}

    def _hand_features(self, hand: Dict[str, Any]):
        key = id(hand)
        value = self._hand_cache.get(key)
        if value is None:
            value = hand_features(hand)
            self._hand_cache[key] = value
        return value

    def chunk_feature_dict(self, hands: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        rows: List[Dict[str, float]] = []
        bigram_total: Counter = Counter()
        joint_total: Counter = Counter()
        templates: Counter = Counter()
        for hand in hands:
            row, bigrams, joints = self._hand_features(hand)
            rows.append(row)
            bigram_total.update(bigrams)
            joint_total.update(joints)
            template = tuple(
                round(row[key], 2) for key in sorted(row) if key.startswith("action_")
            )
            templates[template] += 1

        rows.sort(
            key=lambda row: hashlib.sha256(
                repr(tuple((key, round(row[key], 10)) for key in sorted(row))).encode()
            ).digest()
        )
        out: Dict[str, float] = {}
        keys = sorted({key for row in rows for key in row})
        for key in keys:
            values = np.asarray([row.get(key, 0.0) for row in rows], dtype=float)
            if values.size == 0:
                values = np.zeros(1)
            out[f"{key}__mean"] = float(values.mean())
            out[f"{key}__std"] = float(values.std())
            out[f"{key}__mad"] = float(
                np.median(np.abs(values - np.median(values)))
            )
            # The stable implementation invokes np.quantile once per q. One
            # vector invocation is bit-identical for a 1-D input and avoids
            # four extra Python/NumPy crossings per hand feature.
            quantiles = np.quantile(values, QUANTILES)
            for q, value in zip(QUANTILES, quantiles):
                out[f"{key}__q{int(q * 100):02d}"] = float(value)

        bigram_denominator = max(1, sum(bigram_total.values()))
        for first in ACTIONS:
            for second in ACTIONS:
                out[f"bigram__{first}>{second}"] = (
                    bigram_total[f"{first}>{second}"] / bigram_denominator
                )
        joint_denominator = max(1, sum(joint_total.values()))
        for street in STREETS:
            for action in ACTIONS:
                out[f"joint__{street}:{action}"] = (
                    joint_total[f"{street}:{action}"] / joint_denominator
                )

        count = max(1, len(rows))
        out["template_concentration"] = max(templates.values(), default=0) / count
        out["template_unique_rate"] = len(templates) / count
        out["template_entropy"] = _entropy(map(str, templates.elements()))

        if len(rows) >= 4:
            matrix = np.asarray(
                [[row.get(key, 0.0) for key in keys] for row in rows], dtype=float
            )
            signatures = np.asarray(
                [
                    int.from_bytes(
                        hashlib.sha256(np.round(row, 4).tobytes()).digest()[:8], "big"
                    )
                    for row in matrix
                ],
                dtype=np.uint64,
            )
            order = np.argsort(signatures, kind="stable")
            left = matrix[order[::2]]
            right = matrix[order[1::2]]
            paired = min(len(left), len(right))
            out["half_disagreement"] = float(
                np.mean(np.abs(left[:paired].mean(axis=0) - right[:paired].mean(axis=0)))
            )
        else:
            out["half_disagreement"] = 0.0
        return out

    def matrix_for_chunks(self, chunks: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
        rows = []
        for hands in chunks:
            features = self.chunk_feature_dict(hands)
            rows.append([features.get(name, 0.0) for name in FEATURE_NAMES])
        matrix = np.asarray(rows, dtype=np.float64)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
