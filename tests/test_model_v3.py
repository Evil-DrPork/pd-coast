from __future__ import annotations

import copy
import random

import numpy as np

from detection_model.model_v3.calibration import apply_batch_mapper
from detection_model.model_v3.features import matrix_for_chunks
from poker44.score.scoring import reward


def _hand(types):
    return {
        "metadata": {"hero_seat": 1, "bb": 0.02},
        "players": [{"seat": 1}, {"seat": 2}],
        "streets": [{"street": "preflop"}],
        "actions": [
            {
                "action_id": str(i + 1), "street": "preflop",
                "actor_seat": 1 + (i % 2), "action_type": action,
                "normalized_amount_bb": float(i), "pot_after": 0.02 * (i + 1),
            }
            for i, action in enumerate(types)
        ],
    }


def test_chunk_features_are_exactly_hand_permutation_invariant():
    chunk = [_hand(["check", "fold"]), _hand(["call", "raise"]), _hand(["bet", "fold"])]
    shuffled = copy.deepcopy(chunk)
    random.Random(44).shuffle(shuffled)
    assert np.array_equal(matrix_for_chunks([chunk]), matrix_for_chunks([shuffled]))


def test_batch_mapper_is_monotone_and_places_requested_tail_above_half():
    raw = np.linspace(0.1, 0.9, 40)
    mapped = apply_batch_mapper(raw, top_fraction=0.20)
    assert np.all(np.diff(mapped) >= 0)
    assert 7 <= int(np.sum(mapped >= 0.5)) <= 9


def test_validator_reward_gives_full_threshold_quality_at_ten_percent_fpr():
    labels = np.asarray([0] * 20 + [1] * 20)
    scores = np.asarray([0.9] + [0.1] * 19 + [0.8] * 20)
    _, detail = reward(scores, labels)
    assert detail["hard_fpr"] == 0.05
    assert detail["threshold_sanity_quality"] == 1.0

