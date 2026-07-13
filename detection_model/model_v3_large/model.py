"""Hierarchical tabular ensemble for 80--100 hand Poker44 chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .augmentation import (
    LargeAugmentationConfig,
    build_micro_training_views,
    build_training_views,
    deterministic_subbags,
)
from .features import CachedChunkFeaturizer
from .tabular import TabularBaseEnsemble, choose_tabular_weights


@dataclass(frozen=True)
class LargeWeightSelection:
    full_share: float
    micro_mean_share: float
    micro_median_share: float
    original_objective: float
    merged_objective: float
    combined_objective: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "full_share": self.full_share,
            "micro_mean_share": self.micro_mean_share,
            "micro_median_share": self.micro_median_share,
            "original_objective": self.original_objective,
            "merged_objective": self.merged_objective,
            "combined_objective": self.combined_objective,
        }


class LargeChunkTabularEnsemble:
    """Full-chunk learner plus a deterministic micro-bag learner.

    The object follows the existing v3 detector protocol: ``requires_chunks``
    asks the unchanged detector to pass raw chunks, and ``branch_scores`` emits
    probabilities combined by ``branch_weights_``.  The first four columns are
    full-chunk branches, the next four are the mean micro-bag branches, and the
    final four are the median micro-bag branches.
    """

    requires_chunks = True
    channel_names = tuple(
        [f"full_{name}" for name in ("logistic", "extra", "hist", "prototype")]
        + [f"micro_mean_{name}" for name in ("logistic", "extra", "hist", "prototype")]
        + [f"micro_median_{name}" for name in ("logistic", "extra", "hist", "prototype")]
    )

    def __init__(
        self,
        seed: int = 44,
        config: LargeAugmentationConfig | None = None,
    ) -> None:
        self.seed = int(seed)
        self.config = config or LargeAugmentationConfig()
        self.full = TabularBaseEnsemble(self.seed)
        self.micro = TabularBaseEnsemble(self.seed + 1009)
        self.disable_micro_ = False
        base = np.asarray([0.45, 0.25, 0.20, 0.10], dtype=float)
        self.branch_weights_ = np.concatenate([0.50 * base, 0.30 * base, 0.20 * base])
        self.augmentation_stats_: Dict[str, int] = {}

    def fit(
        self,
        chunks: Sequence[Sequence[dict]],
        labels: np.ndarray,
    ) -> "LargeChunkTabularEnsemble":
        labels = np.asarray(labels, dtype=int)
        full_chunks, full_labels, stats = build_training_views(
            chunks, labels, self.config, self.seed + 101
        )
        micro_chunks, micro_labels = build_micro_training_views(
            full_chunks,
            full_labels,
            original_count=len(chunks),
            config=self.config,
        )
        featurizer = CachedChunkFeaturizer()
        self.full.fit(featurizer.matrix_for_chunks(full_chunks), full_labels)
        self.micro.fit(featurizer.matrix_for_chunks(micro_chunks), micro_labels)
        self.augmentation_stats_ = {
            **stats,
            "micro_total": len(micro_chunks),
        }
        return self

    def branch_scores(
        self,
        full_features: np.ndarray,
        chunks: Sequence[Sequence[dict]] | None = None,
        *,
        featurizer: CachedChunkFeaturizer | None = None,
    ) -> np.ndarray:
        if chunks is None:
            raise ValueError("large-chunk tabular ensemble requires raw chunks")
        full_features = np.asarray(full_features, dtype=float)
        full_branches = self.full.branch_scores(full_features)
        if getattr(self, "disable_micro_", False):
            zeros = np.zeros_like(full_branches)
            return np.column_stack([full_branches, zeros, zeros])
        all_bags = []
        spans = []
        featurizer = featurizer or CachedChunkFeaturizer()
        for chunk in chunks:
            bags = deterministic_subbags(
                chunk,
                target_hands=self.config.subbag_target_hands,
                split_threshold=self.config.subbag_split_threshold,
            )
            start = len(all_bags)
            all_bags.extend(bags)
            spans.append((start, len(all_bags)))
        # One vectorized call is dramatically faster than launching the tree
        # ensemble separately for every validator chunk.
        all_micro_branches = self.micro.branch_scores(
            featurizer.matrix_for_chunks(all_bags)
        )
        means = []
        medians = []
        for start, stop in spans:
            micro_branches = all_micro_branches[start:stop]
            means.append(np.mean(micro_branches, axis=0))
            medians.append(np.median(micro_branches, axis=0))
        return np.column_stack(
            [full_branches, np.asarray(means, dtype=float), np.asarray(medians, dtype=float)]
        )

    def raw_score(self, full_features: np.ndarray, chunks: Sequence[Sequence[dict]]) -> np.ndarray:
        branches = self.branch_scores(full_features, chunks)
        return np.clip(branches @ self.branch_weights_, 1e-5, 1.0 - 1e-5)


def _ranking_objective(detail: Dict[str, float]) -> float:
    return 0.60 * float(detail["ap_score"]) + 0.40 * float(detail["bot_recall"])


def choose_large_weights(
    original_branches: np.ndarray,
    original_y: np.ndarray,
    scorer,
    *,
    merged_branches: np.ndarray,
    merged_y: np.ndarray,
    original_guard: float = 0.02,
) -> Tuple[np.ndarray, LargeWeightSelection]:
    """Select within-view compositions, then the full/micro view shares."""

    original_branches = np.asarray(original_branches, dtype=float)
    merged_branches = np.asarray(merged_branches, dtype=float)
    if original_branches.shape[1] != 12 or merged_branches.shape[1] != 12:
        raise ValueError("large ensemble weight selection expects 12 channels")

    group_slices = (slice(0, 4), slice(4, 8), slice(8, 12))
    group_weights = [
        choose_tabular_weights(original_branches[:, group], original_y, scorer)
        for group in group_slices
    ]
    original_group_scores = [
        original_branches[:, group] @ weights
        for group, weights in zip(group_slices, group_weights)
    ]
    merged_group_scores = [
        merged_branches[:, group] @ weights
        for group, weights in zip(group_slices, group_weights)
    ]
    best_original = max(
        _ranking_objective(scorer(original_y, score)) for score in original_group_scores
    )

    candidates = []
    grid = np.arange(0.0, 1.001, 0.10)
    for full_share in grid:
        for mean_share in grid:
            median_share = 1.0 - full_share - mean_share
            if median_share < -1e-9:
                continue
            shares = np.asarray([full_share, mean_share, max(0.0, median_share)], dtype=float)
            original_score = sum(s * values for s, values in zip(shares, original_group_scores))
            merged_score = sum(s * values for s, values in zip(shares, merged_group_scores))
            original_obj = _ranking_objective(scorer(original_y, original_score))
            if original_obj < best_original - float(original_guard):
                continue
            merged_obj = _ranking_objective(scorer(merged_y, merged_score))
            combined = 0.25 * original_obj + 0.75 * merged_obj
            weights = np.concatenate(
                [share * composition for share, composition in zip(shares, group_weights)]
            )
            selection = LargeWeightSelection(
                full_share=float(shares[0]),
                micro_mean_share=float(shares[1]),
                micro_median_share=float(shares[2]),
                original_objective=float(original_obj),
                merged_objective=float(merged_obj),
                combined_objective=float(combined),
            )
            candidates.append((combined, original_obj, -float(shares[2]), weights, selection))
    if not candidates:
        raise RuntimeError("no large-chunk weight candidate passed the original-data guard")
    _, _, _, weights, selection = max(candidates, key=lambda item: item[:3])
    return weights, selection
