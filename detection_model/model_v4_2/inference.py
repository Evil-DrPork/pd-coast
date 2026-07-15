"""Poker44 V4.2 inference wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from detection_model.model_v4.inference import Poker44V4Detector

from .selection import HAND_CAP, SELECTION_ALGORITHM, prepare_chunks


class Poker44V42Detector(Poker44V4Detector):
    """Apply the V4.2 request policy before the qualified detector."""

    detector_version = "4.2.0"
    hand_cap = HAND_CAP
    selection_algorithm = SELECTION_ALGORITHM

    @staticmethod
    def _clean(
        chunks: Sequence[Sequence[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        selected, _ = prepare_chunks(chunks)
        return selected

    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        *,
        batch_map: bool = True,
        return_diagnostics: bool = False,
    ):
        result = super().predict_chunks(
            chunks,
            batch_map=batch_map,
            return_diagnostics=return_diagnostics,
        )
        if not return_diagnostics:
            return result

        scores, diagnostics = result
        _, selection_diagnostics = prepare_chunks(chunks)
        diagnostics = dict(diagnostics)
        diagnostics["length_balance"] = selection_diagnostics
        return scores, diagnostics
