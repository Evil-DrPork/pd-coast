from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

import numpy as np

from detection_model.model_v4.features import (
    BASE_FEATURE_COUNT,
    FEATURE_IMPLEMENTATION_SHA256,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
)
from detection_model.model_v4.inference import Poker44V4Detector
from detection_model.model_v4.model import BRANCH_NAMES, percentile_feature_matrix
from detection_model.model_v4.provenance import DETECTOR_RUNTIME_SHA256, RUNTIME_LIBRARY_VERSIONS
from detection_model.model_v4_3.inference import (
    ACTIVE_BRANCH_INDICES,
    Poker44V43Detector,
    QUALIFIED_WEIGHTS,
)


class _ColumnEstimator:
    def __init__(self, column: int, scale: float, offset: float = 0.0) -> None:
        self.column = column
        self.scale = scale
        self.offset = offset

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        column = self.column % values.shape[1]
        raw = self.offset + self.scale * values[:, column]
        raw = raw - np.median(raw)
        probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))
        return np.column_stack((1.0 - probability, probability))


class _TinyBranchModel:
    branch_names = BRANCH_NAMES

    def __init__(self) -> None:
        self.branch_weights_ = QUALIFIED_WEIGHTS.copy()
        self.coherent_hist = _ColumnEstimator(3, 0.7, 0.2)
        self.combined_logistic = _ColumnEstimator(11, -0.9, -0.1)
        self.rank_coherent_hist = _ColumnEstimator(7, 1.2)
        self.rank_combined_extra = _ColumnEstimator(17, -1.1, 0.3)
        self.rank_combined_logistic = _ColumnEstimator(23, 0.8, -0.2)

    @staticmethod
    def _views(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        combined = np.asarray(matrix, dtype=float)
        return combined, combined[:, :BASE_FEATURE_COUNT], combined[:, BASE_FEATURE_COUNT:]

    def branch_scores(self, matrix: np.ndarray) -> np.ndarray:
        combined, _, coherent = self._views(matrix)
        ranked = percentile_feature_matrix(combined)
        _, _, rank_coherent = self._views(ranked)
        active = np.column_stack(
            (
                self.coherent_hist.predict_proba(coherent)[:, 1],
                self.combined_logistic.predict_proba(combined)[:, 1],
                self.rank_coherent_hist.predict_proba(rank_coherent)[:, 1],
                self.rank_combined_extra.predict_proba(ranked)[:, 1],
                self.rank_combined_logistic.predict_proba(ranked)[:, 1],
            )
        )
        output = np.full((len(combined), len(BRANCH_NAMES)), 0.37, dtype=float)
        output[:, np.asarray(ACTIVE_BRANCH_INDICES)] = active
        return np.clip(output, 1e-6, 1.0 - 1e-6)


def _artifact() -> dict:
    zeros = np.zeros(len(FEATURE_NAMES), dtype=float)
    ones = np.ones(len(FEATURE_NAMES), dtype=float)
    return {
        "artifact_version": 4,
        "architecture": "coherent_real_rank_robust_v2",
        "model": _TinyBranchModel(),
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_implementation_sha256": FEATURE_IMPLEMENTATION_SHA256,
        "detector_runtime_sha256": DETECTOR_RUNTIME_SHA256,
        "runtime_library_versions": dict(RUNTIME_LIBRARY_VERSIONS),
        "branch_names": list(BRANCH_NAMES),
        "feature_reference": {
            "q01": zeros - 100.0,
            "q25": zeros - 1.0,
            "median": zeros,
            "q75": ones,
            "q99": ones * 100.0,
        },
        "blend_mode": "probability",
        "mapper": {"cut": 0.5, "scale": 1.0},
        "batch_top_fraction": 0.10,
        "training_count": 8,
    }


def _chunks(count: int) -> list[list[dict]]:
    chunks = []
    for index in range(count):
        amount = 0.5 + (index % 19) * 0.17
        actor = 1 + (index % 5)
        chunks.append(
            [
                {
                    "metadata": {
                        "bb": 0.02,
                        "sb": 0.01,
                        "hero_seat": actor,
                        "button_seat": 1 + ((index + 2) % 5),
                        "max_seats": 5,
                    },
                    "players": [
                        {"seat": seat, "starting_stack": 2.0 + 0.03 * index + 0.1 * seat}
                        for seat in range(1, 6)
                    ],
                    "actions": [
                        {
                            "action_id": 1,
                            "action_type": ("raise", "call", "check", "fold")[index % 4],
                            "actor_seat": actor,
                            "amount": amount * 0.02,
                            "normalized_amount_bb": amount,
                            "pot_before": 0.03 + index * 0.0001,
                            "pot_after": 0.03 + amount * 0.02 + index * 0.0001,
                            "street": "preflop",
                        }
                    ],
                    "streets": [{"street": "preflop"}],
                }
            ]
        )
    return chunks


class ModelV43Tests(unittest.TestCase):
    def test_zero_consensus_is_exact_v41_fast_path(self) -> None:
        chunks = _chunks(100)
        control = Poker44V4Detector(_artifact())
        with patch.dict(os.environ, {"P44_V43_CONSENSUS_ALPHA": "0"}):
            challenger = Poker44V43Detector(_artifact())
        expected = control.predict_chunks(chunks)
        actual, diagnostics = challenger.predict_chunks(chunks, return_diagnostics=True)
        self.assertEqual(actual, expected)
        self.assertTrue(diagnostics["v43_fast_path"])
        self.assertEqual(diagnostics["v43_consensus_alpha"], 0.0)

    def test_consensus_is_size_guarded_and_permutation_invariant(self) -> None:
        with patch.dict(os.environ, {"P44_V43_CONSENSUS_ALPHA": "0.2"}):
            challenger = Poker44V43Detector(_artifact())
        control = Poker44V4Detector(_artifact())
        short = _chunks(40)
        self.assertEqual(challenger.predict_chunks(short), control.predict_chunks(short))

        chunks = _chunks(100)
        scores, diagnostics = challenger.predict_chunks(chunks, return_diagnostics=True)
        self.assertEqual(sum(value >= 0.5 for value in scores), 10)
        self.assertAlmostEqual(diagnostics["v43_consensus_alpha"], 0.2)

        permutation = np.random.default_rng(44).permutation(len(chunks))
        permuted = challenger.predict_chunks([chunks[index] for index in permutation])
        restored = np.empty(len(permuted), dtype=float)
        restored[permutation] = permuted
        self.assertTrue(np.array_equal(np.asarray(scores), restored))

    def test_invalid_consensus_alpha_is_rejected(self) -> None:
        with patch.dict(os.environ, {"P44_V43_CONSENSUS_ALPHA": "0.4"}):
            with self.assertRaisesRegex(ValueError, "between 0 and"):
                Poker44V43Detector(_artifact())

    def test_unqualified_artifact_falls_back_to_v41(self) -> None:
        artifact = _artifact()
        artifact["model"].branch_weights_ = np.ones(len(BRANCH_NAMES), dtype=float) / len(
            BRANCH_NAMES
        )
        with patch.dict(os.environ, {"P44_V43_CONSENSUS_ALPHA": "0.2"}):
            challenger = Poker44V43Detector(copy.deepcopy(artifact))
        control = Poker44V4Detector(copy.deepcopy(artifact))
        expected = control.predict_chunks(_chunks(20))
        actual, diagnostics = challenger.predict_chunks(_chunks(20), return_diagnostics=True)
        self.assertEqual(actual, expected)
        self.assertFalse(diagnostics["v43_fast_path"])
        self.assertEqual(diagnostics["v43_consensus_alpha"], 0.0)


if __name__ == "__main__":
    unittest.main()
