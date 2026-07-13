from __future__ import annotations

import copy
import random

import numpy as np

from detection_model.model_v3.features import matrix_for_chunks
from detection_model.model_v3_large.augmentation import (
    LargeAugmentationConfig,
    build_training_views,
    deterministic_subbags,
)
from detection_model.model_v3_large.features import CachedChunkFeaturizer
from detection_model.model_v3_large.model import LargeChunkTabularEnsemble


def _hand(index: int) -> dict:
    return {
        "marker": index,
        "metadata": {"hero_seat": 1, "bb": 0.02},
        "players": [{"seat": 1}, {"seat": 2}],
        "streets": [{"street": "preflop"}],
        "actions": [
            {
                "action_id": "1",
                "street": "preflop",
                "actor_seat": 1 + index % 2,
                "action_type": ("raise" if index % 3 == 0 else "call"),
                "normalized_amount_bb": float(1 + index % 8),
                "pot_after": 0.02 * (2 + index % 5),
            }
        ],
    }


def _bag_markers(bags):
    return sorted(sorted(hand["marker"] for hand in bag) for bag in bags)


def test_subbags_are_permutation_invariant_balanced_and_lossless():
    chunk = [_hand(i) for i in range(91)]
    shuffled = copy.deepcopy(chunk)
    random.Random(44).shuffle(shuffled)
    first = deterministic_subbags(chunk)
    second = deterministic_subbags(shuffled)
    assert _bag_markers(first) == _bag_markers(second)
    assert sorted(hand["marker"] for bag in first for hand in bag) == list(range(91))
    assert max(map(len, first)) - min(map(len, first)) <= 1
    assert len(first) == 3


def test_large_training_views_preserve_originals_and_add_requested_scales():
    chunks = [[_hand(1000 * i + j) for j in range(35)] for i in range(12)]
    labels = np.asarray([0, 1] * 6, dtype=int)
    config = LargeAugmentationConfig(large_ratio=0.5, medium_ratio=0.25)
    views, view_labels, stats = build_training_views(chunks, labels, config, seed=44)
    assert stats == {"original": 12, "large": 6, "medium": 3, "full_total": 21}
    assert len(views) == len(view_labels) == 21
    assert [len(chunk) for chunk in views[:12]] == [35] * 12
    assert all(80 <= len(chunk) <= 100 for chunk in views[12:18])
    assert all(50 <= len(chunk) <= 75 for chunk in views[18:])


def test_cached_feature_matrix_is_exactly_the_legacy_v3_matrix():
    chunks = [[_hand(1000 * i + j) for j in range(35)] for i in range(4)]
    expected = matrix_for_chunks(chunks)
    actual = CachedChunkFeaturizer().matrix_for_chunks(chunks)
    assert np.array_equal(actual, expected)


class _FeatureProbabilityModel:
    def branch_scores(self, x):
        x = np.asarray(x, dtype=float)
        base = 1.0 / (1.0 + np.exp(-np.clip(x[:, :4], -10, 10)))
        return base


class _MustNotRunModel:
    def branch_scores(self, _x):
        raise AssertionError("micro model should be disabled")


def test_large_wrapper_branch_scores_are_exactly_hand_permutation_invariant():
    chunk = [_hand(i) for i in range(91)]
    shuffled = copy.deepcopy(chunk)
    random.Random(144).shuffle(shuffled)
    model = LargeChunkTabularEnsemble()
    model.full = _FeatureProbabilityModel()
    model.micro = _FeatureProbabilityModel()
    first = model.branch_scores(matrix_for_chunks([chunk]), [chunk])
    second = model.branch_scores(matrix_for_chunks([shuffled]), [shuffled])
    assert first.shape == (1, 12)
    assert np.array_equal(first, second)


def test_fast_ablation_skips_micro_inference():
    chunk = [_hand(i) for i in range(91)]
    model = LargeChunkTabularEnsemble()
    model.full = _FeatureProbabilityModel()
    model.micro = _MustNotRunModel()
    model.disable_micro_ = True
    branches = model.branch_scores(matrix_for_chunks([chunk]), [chunk])
    assert branches.shape == (1, 12)
    assert np.all(branches[:, 4:] == 0.0)
