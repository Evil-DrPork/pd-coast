"""Faster detector path for the separate large-chunk challenger."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from detection_model.model_v3.inference import Poker44V3Detector
from detection_model.model_v3.schema import clean_hand

from .features import CachedChunkFeaturizer


class Poker44V3LargeDetector(Poker44V3Detector):
    """Use exact cached features and share their hand cache with micro-bags."""

    def _prepare(self, chunks: List[List[Dict[str, Any]]]):
        clean = [
            [clean_hand(hand) for hand in (chunk or []) if isinstance(hand, dict)]
            for chunk in chunks
        ]
        featurizer = CachedChunkFeaturizer()
        features = featurizer.matrix_for_chunks(clean)
        return clean, features, featurizer

    def _branches(self, clean, features, featurizer):
        if getattr(self.model, "requires_chunks", False):
            try:
                return self.model.branch_scores(
                    features, clean, featurizer=featurizer
                )
            except TypeError:
                return self.model.branch_scores(features, clean)
        return self.model.branch_scores(features)

    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        *,
        batch_map: bool = True,
        return_diagnostics: bool = False,
    ):
        if not chunks:
            return ([], {}) if return_diagnostics else []
        clean, features, featurizer = self._prepare(chunks)
        branches = self._branches(clean, features, featurizer)
        raw = np.clip(branches @ self.model.branch_weights_, 1e-5, 1.0 - 1e-5)
        scores = self.map_scores(raw, batch_map=batch_map)
        scores = np.nan_to_num(scores, nan=0.5, posinf=0.99, neginf=0.01)
        values = [round(float(np.clip(value, 0.01, 0.99)), 8) for value in scores]
        if not return_diagnostics:
            return values
        diagnostics = {
            "chunk_count": len(chunks),
            "hand_counts": [len(chunk) for chunk in clean],
            "branch_disagreement_mean": float(np.std(branches, axis=1).mean()),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "batch_mapping": bool(batch_map and len(chunks) >= 8),
            "positive_rate": float(np.mean(scores >= 0.5)),
            "large_cached_inference": True,
        }
        return values, diagnostics

    def raw_scores(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        clean, features, featurizer = self._prepare(chunks)
        branches = self._branches(clean, features, featurizer)
        return np.clip(branches @ self.model.branch_weights_, 1e-5, 1.0 - 1e-5)
