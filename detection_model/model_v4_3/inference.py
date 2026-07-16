"""V4.3 serving wrapper over the qualified V4.1 artifact."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np

from detection_model.model_v3.calibration import apply_mapper
from detection_model.model_v4.features import matrix_for_chunks
from detection_model.model_v4.inference import Poker44V4Detector
from detection_model.model_v4.mapping import chunk_tie_key, exact_rank_map
from detection_model.model_v4.model import (
    BRANCH_NAMES,
    percentile_columns,
    percentile_feature_matrix,
)


ACTIVE_BRANCH_INDICES = (2, 4, 6, 7, 8)
ACTIVE_BRANCH_NAMES = tuple(BRANCH_NAMES[index] for index in ACTIVE_BRANCH_INDICES)
QUALIFIED_WEIGHTS = np.asarray(
    [0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2, 0.2],
    dtype=float,
)
DEFAULT_CONSENSUS_ALPHA = 0.20
MAX_CONSENSUS_ALPHA = 0.25


def _configured_alpha() -> float:
    raw = os.getenv("P44_V43_CONSENSUS_ALPHA", str(DEFAULT_CONSENSUS_ALPHA)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("P44_V43_CONSENSUS_ALPHA must be numeric") from exc
    if not 0.0 <= value <= MAX_CONSENSUS_ALPHA:
        raise ValueError(
            f"P44_V43_CONSENSUS_ALPHA must be between 0 and {MAX_CONSENSUS_ALPHA}"
        )
    return value


class Poker44V43Detector(Poker44V4Detector):
    """Run the active V4.1 branches with an optional guarded consensus."""

    detector_version = "4.3.0"

    def __init__(self, artifact: Dict[str, Any]) -> None:
        super().__init__(artifact)
        weights = np.asarray(getattr(self.model, "branch_weights_", ()), dtype=float)
        self.fast_path_enabled = bool(
            self.blend_mode == "probability"
            and weights.shape == QUALIFIED_WEIGHTS.shape
            and np.array_equal(weights, QUALIFIED_WEIGHTS)
            and tuple(getattr(self.model, "branch_names", ())) == BRANCH_NAMES
        )
        self.consensus_alpha = _configured_alpha()

    @staticmethod
    def _effective_alpha(size: int, maximum: float) -> float:
        return float(maximum) * float(np.clip((int(size) - 40) / 60.0, 0.0, 1.0))

    def _active_branch_scores(self, matrix: np.ndarray) -> np.ndarray:
        combined, _, coherent = self.model._views(matrix)
        rank_combined = percentile_feature_matrix(combined)
        _, _, rank_coherent = self.model._views(rank_combined)
        columns = (
            self.model.coherent_hist.predict_proba(coherent)[:, 1],
            self.model.combined_logistic.predict_proba(combined)[:, 1],
            self.model.rank_coherent_hist.predict_proba(rank_coherent)[:, 1],
            self.model.rank_combined_extra.predict_proba(rank_combined)[:, 1],
            self.model.rank_combined_logistic.predict_proba(rank_combined)[:, 1],
        )
        return np.clip(np.column_stack(columns), 1e-6, 1.0 - 1e-6)

    @staticmethod
    def _keys_if_tied(
        raw: np.ndarray,
        clean: Sequence[Sequence[Dict[str, Any]]],
    ) -> list[str] | None:
        if len(np.unique(np.asarray(raw, dtype=float))) == len(raw):
            return None
        return [chunk_tie_key(chunk) for chunk in clean]

    def _fast_raw(
        self,
        matrix: np.ndarray,
        clean: Sequence[Sequence[Dict[str, Any]]],
    ) -> tuple[np.ndarray, np.ndarray, list[str] | None, float]:
        branches = self._active_branch_scores(matrix)
        active_weights = QUALIFIED_WEIGHTS[np.asarray(ACTIVE_BRANCH_INDICES, dtype=int)]
        active_weights = active_weights / active_weights.sum()
        baseline = np.clip(branches @ active_weights, 1e-6, 1.0 - 1e-6)
        alpha = self._effective_alpha(len(clean), self.consensus_alpha)
        if alpha <= 0.0:
            return baseline, branches, self._keys_if_tied(baseline, clean), alpha

        keys = [chunk_tie_key(chunk) for chunk in clean]
        base_rank = percentile_columns(baseline[:, None], tie_keys=keys)[:, 0]
        branch_ranks = percentile_columns(branches, tie_keys=keys)
        consensus = np.mean(branch_ranks, axis=1)
        raw = (1.0 - alpha) * base_rank + alpha * consensus
        return raw, branches, keys, alpha

    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        *,
        batch_map: bool = True,
        return_diagnostics: bool = False,
    ):
        if not chunks:
            return ([], {}) if return_diagnostics else []
        if not self.fast_path_enabled:
            result = super().predict_chunks(
                chunks,
                batch_map=batch_map,
                return_diagnostics=return_diagnostics,
            )
            if not return_diagnostics:
                return result
            scores, diagnostics = result
            diagnostics = dict(diagnostics)
            diagnostics.update(
                {
                    "detector_version": self.detector_version,
                    "v43_fast_path": False,
                    "v43_consensus_alpha": 0.0,
                }
            )
            return scores, diagnostics

        clean = self._clean(chunks)
        matrix = matrix_for_chunks(clean)
        raw, branches, keys, alpha = self._fast_raw(matrix, clean)
        if batch_map and len(raw) >= 8:
            scores = exact_rank_map(raw, self.batch_top_fraction, tie_keys=keys)
        else:
            scores = apply_mapper(raw, self.mapper)
        scores = np.nan_to_num(scores, nan=0.5, posinf=0.99, neginf=0.01)
        output = [round(float(np.clip(value, 0.01, 0.99)), 8) for value in scores]
        if not return_diagnostics:
            return output

        lower = self.feature_reference["q01"]
        upper = self.feature_reference["q99"]
        outside = (matrix < lower) | (matrix > upper)
        training_iqr = np.maximum(
            self.feature_reference["q75"] - self.feature_reference["q25"],
            1e-9,
        )
        median_shift = np.abs(
            np.median(matrix, axis=0) - self.feature_reference["median"]
        ) / training_iqr
        diagnostics = {
            "chunk_count": len(clean),
            "hand_counts": [len(chunk) for chunk in clean],
            "blend_mode": self.blend_mode,
            "branch_names": list(ACTIVE_BRANCH_NAMES),
            "branch_disagreement_mean": float(np.std(branches, axis=1).mean()),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "raw_std": float(raw.std()),
            "feature_outside_q01_q99_rate": float(outside.mean()),
            "features_majority_outside_q01_q99": int(
                np.sum(outside.mean(axis=0) > 0.5)
            ),
            "median_feature_shift_iqr": float(np.median(median_shift)),
            "p90_feature_shift_iqr": float(np.quantile(median_shift, 0.90)),
            "batch_mapping": bool(batch_map and len(clean) >= 8),
            "positive_count": int(np.sum(scores >= 0.5)),
            "positive_rate": float(np.mean(scores >= 0.5)),
            "detector_version": self.detector_version,
            "v43_fast_path": True,
            "v43_consensus_alpha": alpha,
        }
        return output, diagnostics

    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        return self.predict_chunks([chunk], batch_map=False)[0]

    def raw_scores(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        if not self.fast_path_enabled:
            return super().raw_scores(chunks)
        clean = self._clean(chunks)
        matrix = matrix_for_chunks(clean)
        raw, _, _, _ = self._fast_raw(matrix, clean)
        return raw
