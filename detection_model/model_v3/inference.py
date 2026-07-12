"""Production inference wrapper for Poker44 v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from .calibration import apply_batch_mapper, apply_mapper
from .features import FEATURE_NAMES, matrix_for_chunks
from .schema import clean_hand


class Poker44V3Detector:
    def __init__(self, artifact: Dict[str, Any]) -> None:
        if artifact.get("artifact_version") != 3:
            raise ValueError("not a Poker44 v3 artifact")
        if list(artifact.get("feature_names") or []) != FEATURE_NAMES:
            raise ValueError("v3 feature schema mismatch; retrain the artifact")
        self.artifact = artifact
        self.model = artifact["model"]
        self.mapper = artifact["mapper"]
        self.batch_top_fraction = float(artifact.get("batch_top_fraction", 0.20))

    @classmethod
    def load(cls, path: str | Path) -> "Poker44V3Detector":
        return cls(joblib.load(Path(path).expanduser()))

    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        *,
        batch_map: bool = True,
        return_diagnostics: bool = False,
    ):
        if not chunks:
            return ([], {}) if return_diagnostics else []
        clean = [[clean_hand(h) for h in (chunk or []) if isinstance(h, dict)] for chunk in chunks]
        x = matrix_for_chunks(clean)
        branches = self.model.branch_scores(x, clean) if getattr(self.model, "requires_chunks", False) else self.model.branch_scores(x)
        raw = np.clip(branches @ self.model.branch_weights_, 1e-5, 1 - 1e-5)
        scores = self.map_scores(raw, batch_map=batch_map)
        scores = np.nan_to_num(scores, nan=0.5, posinf=0.99, neginf=0.01)
        values = [round(float(np.clip(v, 0.01, 0.99)), 8) for v in scores]
        if not return_diagnostics:
            return values
        diagnostics = {
            "chunk_count": len(chunks),
            "hand_counts": [len(c) for c in clean],
            "branch_disagreement_mean": float(np.std(branches, axis=1).mean()),
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "batch_mapping": bool(batch_map and len(chunks) >= 8),
            "positive_rate": float(np.mean(scores >= 0.5)),
        }
        return values, diagnostics

    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        return self.predict_chunks([chunk], batch_map=False)[0]

    def raw_scores(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        """Return the uncalibrated ensemble scores for reusable window analysis."""
        clean = [[clean_hand(h) for h in (chunk or []) if isinstance(h, dict)] for chunk in chunks]
        x = matrix_for_chunks(clean)
        branches = self.model.branch_scores(x, clean) if getattr(self.model, "requires_chunks", False) else self.model.branch_scores(x)
        return np.clip(branches @ self.model.branch_weights_, 1e-5, 1 - 1e-5)

    def map_scores(self, raw: np.ndarray, *, batch_map: bool = True) -> np.ndarray:
        raw = np.asarray(raw, float)
        fixed = apply_mapper(raw, self.mapper)
        if batch_map and len(raw) >= 8:
            return apply_batch_mapper(raw, self.batch_top_fraction)
        return fixed
