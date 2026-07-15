from __future__ import annotations

import copy
import unittest

import numpy as np

from detection_model.model_v3.calibration import fit_fixed_mapper
from detection_model.model_v4.features import (
    FEATURE_IMPLEMENTATION_SHA256,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    matrix_for_chunks,
)
from detection_model.model_v4.inference import Poker44V4Detector
from detection_model.model_v4.mapping import chunk_tie_key
from detection_model.model_v4.model import BRANCH_NAMES
from detection_model.model_v4.provenance import (
    DETECTOR_RUNTIME_SHA256,
    RUNTIME_LIBRARY_VERSIONS,
)
from detection_model.model_v4_2.inference import Poker44V42Detector
from detection_model.model_v4_2.selection import (
    HAND_CAP,
    SELECTION_ALGORITHM,
    hand_behavior_key,
    prepare_chunks,
)


def _hand(index: int, *, chunk_seed: int = 0) -> dict:
    hero = 1 + ((index + chunk_seed) % 6)
    button = 1 + ((index * 3 + chunk_seed) % 6)
    amount = round(0.75 + ((index * 7 + chunk_seed) % 29) / 20.0, 4)
    return {
        "hand_id": f"private-{chunk_seed}-{index}",
        "metadata": {
            "ante": 0.0,
            "bb": 0.02,
            "button_seat": button,
            "game_type": "holdem",
            "hand_ended_on_street": "",
            "hero_seat": hero,
            "limit_type": "nl",
            "max_seats": 6,
            "sb": 0.01,
        },
        "players": [
            {"seat": seat, "starting_stack": 2.0 + seat * 0.01}
            for seat in range(1, 7)
        ],
        # Deliberately received out of order; clean_hand must order by action_id
        # before hashing or feature extraction.
        "actions": [
            {
                "action_id": "2",
                "action_type": "fold" if index % 3 == 0 else "check",
                "actor_seat": 1 + ((hero + 1) % 6),
                "amount": 0.0,
                "call_to": None,
                "normalized_amount_bb": 0.0,
                "pot_after": 0.08,
                "pot_before": 0.08,
                "raise_to": None,
                "street": "preflop",
            },
            {
                "action_id": "1",
                "action_type": "raise" if index % 5 == 0 else "call",
                "actor_seat": hero,
                "amount": amount * 0.02,
                "call_to": amount * 0.02,
                "normalized_amount_bb": amount,
                "pot_after": 0.08,
                "pot_before": 0.03,
                "raise_to": amount * 0.02 if index % 5 == 0 else None,
                "street": "preflop",
            },
        ],
        "streets": [{"street": "preflop"}],
        "outcome": {"winners": ["private"], "payouts": [999]},
    }


def _chunk(size: int, seed: int) -> list[dict]:
    return [_hand(index, chunk_seed=seed) for index in range(size)]


class _FixtureModel:
    branch_names = BRANCH_NAMES

    def __init__(self) -> None:
        self.branch_weights_ = np.full(len(BRANCH_NAMES), 1.0 / len(BRANCH_NAMES))

    def branch_scores(self, matrix: np.ndarray) -> np.ndarray:
        signal = np.tanh(
            np.sum(matrix[:, :12] * np.linspace(0.01, 0.12, 12), axis=1) / 25.0
        )
        center = 0.5 + 0.35 * signal
        offsets = np.linspace(-0.04, 0.04, len(BRANCH_NAMES))
        return np.clip(center[:, None] + offsets[None, :], 0.001, 0.999)


def _artifact() -> dict:
    width = len(FEATURE_NAMES)
    return {
        "artifact_version": 4,
        "architecture": "coherent_real_rank_robust_v2",
        "model": _FixtureModel(),
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_implementation_sha256": FEATURE_IMPLEMENTATION_SHA256,
        "detector_runtime_sha256": DETECTOR_RUNTIME_SHA256,
        "runtime_library_versions": dict(RUNTIME_LIBRARY_VERSIONS),
        "feature_reference": {
            "q01": np.full(width, -1_000_000.0),
            "q25": np.full(width, -1.0),
            "median": np.zeros(width),
            "q75": np.ones(width),
            "q99": np.full(width, 1_000_000.0),
        },
        "blend_mode": "rank",
        "mapper": fit_fixed_mapper(np.asarray([0.1, 0.9]), np.asarray([0, 1])),
        "batch_top_fraction": 0.10,
        "training_count": 2,
    }


class ModelV42Tests(unittest.TestCase):
    def test_fixed80_selection_is_exact_and_hand_permutation_invariant(self) -> None:
        chunks = [_chunk(80, 1), _chunk(100, 2), _chunk(83, 3)]
        before = copy.deepcopy(chunks)

        selected, diagnostics = prepare_chunks(chunks)
        reversed_selected, reversed_diagnostics = prepare_chunks(
            [list(reversed(chunk)) for chunk in chunks]
        )

        self.assertEqual(chunks, before)
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(diagnostics["algorithm"], SELECTION_ALGORITHM)
        self.assertEqual(diagnostics["hand_cap"], HAND_CAP)
        self.assertEqual(diagnostics["original_hand_counts"], [80, 100, 83])
        self.assertEqual(diagnostics["selected_hand_counts"], [80, 80, 80])
        self.assertEqual(diagnostics["dropped_hand_counts"], [0, 20, 3])
        self.assertEqual(diagnostics["dropped_hand_count"], 23)
        self.assertEqual(diagnostics, reversed_diagnostics)
        for first, second in zip(selected, reversed_selected):
            self.assertEqual(
                [hand_behavior_key(hand) for hand in first],
                [hand_behavior_key(hand) for hand in second],
            )
        self.assertTrue(
            np.array_equal(matrix_for_chunks(selected), matrix_for_chunks(reversed_selected))
        )

    def test_short_chunk_falls_back_for_the_entire_request(self) -> None:
        chunks = [_chunk(100, 4), _chunk(79, 5)]
        selected, diagnostics = prepare_chunks(chunks)

        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["fallback_reason"], "chunk_below_hand_cap")
        self.assertEqual(diagnostics["original_hand_counts"], [100, 79])
        self.assertEqual(diagnostics["selected_hand_counts"], [100, 79])
        self.assertEqual(diagnostics["dropped_hand_count"], 0)
        baseline_clean = Poker44V4Detector._clean(chunks)
        self.assertEqual(chunk_tie_key(selected[0]), chunk_tie_key(baseline_clean[0]))

    def test_selection_ignores_private_fields_and_received_action_order(self) -> None:
        chunks = [_chunk(100, 6)]
        changed = copy.deepcopy(chunks)
        for index, hand in enumerate(changed[0]):
            hand["hand_id"] = f"different-{index}"
            hand["outcome"] = {"winners": [index], "payouts": [index * 1000]}
            hand["metadata"]["rng_seed_commitment"] = f"secret-{index}"
            hand["players"][0]["player_uid"] = f"uid-{index}"
            hand["players"][0]["hole_cards"] = ["As", "Ah"]
            hand["actions"] = list(reversed(hand["actions"]))

        selected, _ = prepare_chunks(chunks)
        changed_selected, _ = prepare_chunks(changed)
        self.assertEqual(
            [hand_behavior_key(hand) for hand in selected[0]],
            [hand_behavior_key(hand) for hand in changed_selected[0]],
        )
        self.assertTrue(
            np.array_equal(
                matrix_for_chunks(selected),
                matrix_for_chunks(changed_selected),
            )
        )

    def test_behavior_duplicate_ties_cannot_change_features(self) -> None:
        prototype = _hand(7, chunk_seed=7)
        first = []
        for index in range(100):
            hand = copy.deepcopy(prototype)
            hand["hand_id"] = f"copy-{index}"
            first.append(hand)

        selected, _ = prepare_chunks([first])
        reversed_selected, _ = prepare_chunks([list(reversed(first))])
        self.assertEqual(len(selected[0]), HAND_CAP)
        self.assertEqual(
            [hand_behavior_key(hand) for hand in selected[0]],
            [hand_behavior_key(hand) for hand in reversed_selected[0]],
        )
        self.assertTrue(
            np.array_equal(matrix_for_chunks(selected), matrix_for_chunks(reversed_selected))
        )

    def test_detector_matches_v41_at_80_and_keeps_mapper_contract(self) -> None:
        artifact = _artifact()
        v41 = Poker44V4Detector(artifact)
        v42 = Poker44V42Detector(artifact)
        exactly_80 = [_chunk(80, seed) for seed in range(10, 20)]

        self.assertEqual(v41.predict_chunks(exactly_80), v42.predict_chunks(exactly_80))

        long_chunks = [_chunk(81 + index * 2, 100 + index) for index in range(10)]
        scores, diagnostics = v42.predict_chunks(long_chunks, return_diagnostics=True)
        permutation = np.asarray([7, 1, 9, 0, 5, 3, 8, 2, 6, 4], dtype=int)
        permuted = [
            list(reversed(long_chunks[index]))
            for index in permutation
        ]
        permuted_scores = v42.predict_chunks(permuted)
        restored = np.empty(len(scores), dtype=float)
        restored[permutation] = np.asarray(permuted_scores, dtype=float)

        self.assertTrue(np.array_equal(np.asarray(scores), restored))
        self.assertEqual(sum(value >= 0.5 for value in scores), 1)
        self.assertEqual(diagnostics["positive_count"], 1)
        self.assertEqual(diagnostics["hand_counts"], [80] * 10)
        self.assertTrue(diagnostics["length_balance"]["applied"])
        self.assertEqual(
            diagnostics["length_balance"]["selected_hand_counts"],
            [80] * 10,
        )

    def test_empty_request_is_preserved(self) -> None:
        detector = Poker44V42Detector(_artifact())
        self.assertEqual(detector.predict_chunks([]), [])
        scores, diagnostics = detector.predict_chunks([], return_diagnostics=True)
        self.assertEqual(scores, [])
        self.assertEqual(diagnostics["length_balance"]["fallback_reason"], "empty_request")


if __name__ == "__main__":
    unittest.main()
