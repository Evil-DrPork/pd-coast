"""Permutation-invariant multi-scale views for large Poker44 chunks.

This package deliberately lives outside ``model_v3``.  The stable v3 tabular
implementation remains unchanged while this challenger learns from synthetic
50--100 hand views and deterministic micro-bags.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LargeAugmentationConfig:
    """Ratios are relative to the number of original training chunks."""

    large_ratio: float = 1.25
    medium_ratio: float = 0.40
    sources_per_merge: int = 3
    large_min_hands: int = 80
    large_max_hands: int = 100
    medium_min_hands: int = 50
    medium_max_hands: int = 75
    subbag_target_hands: int = 34
    subbag_split_threshold: int = 45

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hand_bytes(hand: Dict[str, Any]) -> bytes:
    return json.dumps(
        hand, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")


def hand_digest(hand: Dict[str, Any], *, salt: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_hand_bytes(hand))
    return digest.hexdigest()


def deterministic_subbags(
    hands: Sequence[Dict[str, Any]],
    *,
    target_hands: int = 34,
    split_threshold: int = 45,
) -> List[List[Dict[str, Any]]]:
    """Partition a hand multiset without using received hand order.

    Every hand appears exactly once.  Content hashes define a stable ordering,
    then round-robin assignment keeps bag sizes balanced.  Chunks at or below
    ``split_threshold`` stay intact so the original 30--40 hand distribution is
    not needlessly fragmented.
    """

    clean = [hand for hand in hands if isinstance(hand, dict)]
    if not clean:
        return [[]]
    ordered = sorted(
        clean,
        key=lambda hand: (hand_digest(hand, salt="p44-v3-large-subbag"), _hand_bytes(hand)),
    )
    if len(ordered) <= max(1, int(split_threshold)):
        return [ordered]

    target = max(1, int(target_hands))
    parts = max(2, int(np.ceil(len(ordered) / target)))
    bags: List[List[Dict[str, Any]]] = [[] for _ in range(parts)]
    for index, hand in enumerate(ordered):
        bags[index % parts].append(hand)
    return [bag for bag in bags if bag]


def _balanced_labels(count: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(([0, 1] * ((count + 1) // 2))[:count], dtype=int)
    rng.shuffle(labels)
    return labels


def generate_merged_chunks(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    labels: np.ndarray,
    *,
    count: int,
    min_hands: int,
    max_hands: int,
    sources_per_merge: int,
    rng: np.random.Generator,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray]:
    """Generate same-label mixtures from sources already inside one data fold."""

    labels = np.asarray(labels, dtype=int)
    if count <= 0:
        return [], np.zeros(0, dtype=int)
    if min_hands <= 0 or min_hands > max_hands:
        raise ValueError("invalid merged hand range")
    if sources_per_merge < 2:
        raise ValueError("sources_per_merge must be at least 2")

    by_label = {
        label: np.flatnonzero(labels == label) for label in (0, 1)
    }
    for label, indices in by_label.items():
        if len(indices) < sources_per_merge:
            raise ValueError(
                f"need at least {sources_per_merge} label={label} source chunks, got {len(indices)}"
            )

    merged: List[List[Dict[str, Any]]] = []
    merged_labels: List[int] = []
    for label in _balanced_labels(count, rng):
        candidates = by_label[int(label)]
        selected = None
        pool: List[Dict[str, Any]] = []
        # Public chunks are normally 30--40 hands, so three sources are enough.
        # The retry makes malformed or unusually small chunks fail explicitly.
        for _ in range(64):
            selected = rng.choice(candidates, size=sources_per_merge, replace=False)
            pool = [hand for idx in selected for hand in chunks[int(idx)] if isinstance(hand, dict)]
            if len(pool) >= min_hands:
                break
        if len(pool) < min_hands:
            raise ValueError(
                f"could not build label={int(label)} merged chunk with at least {min_hands} hands"
            )
        target = min(len(pool), int(rng.integers(min_hands, max_hands + 1)))
        chosen = rng.choice(len(pool), size=target, replace=False)
        # Ordering is intentionally arbitrary: all downstream operations are
        # functions of the hand multiset.
        merged.append([pool[int(i)] for i in chosen])
        merged_labels.append(int(label))
    return merged, np.asarray(merged_labels, dtype=int)


def build_training_views(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    labels: np.ndarray,
    config: LargeAugmentationConfig,
    seed: int,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, Dict[str, int]]:
    """Return original + medium + evaluation-sized full-chunk training views."""

    original = [list(chunk) for chunk in chunks]
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    large, y_large = generate_merged_chunks(
        original,
        labels,
        count=int(round(len(original) * max(0.0, config.large_ratio))),
        min_hands=config.large_min_hands,
        max_hands=config.large_max_hands,
        sources_per_merge=config.sources_per_merge,
        rng=rng,
    )
    medium, y_medium = generate_merged_chunks(
        original,
        labels,
        count=int(round(len(original) * max(0.0, config.medium_ratio))),
        min_hands=config.medium_min_hands,
        max_hands=config.medium_max_hands,
        sources_per_merge=config.sources_per_merge,
        rng=rng,
    )
    full_chunks = original + large + medium
    full_labels = np.concatenate([labels, y_large, y_medium])
    stats = {
        "original": len(original),
        "large": len(large),
        "medium": len(medium),
        "full_total": len(full_chunks),
    }
    return full_chunks, full_labels, stats


def build_micro_training_views(
    full_chunks: Sequence[Sequence[Dict[str, Any]]],
    full_labels: np.ndarray,
    *,
    original_count: int,
    config: LargeAugmentationConfig,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray]:
    """Keep originals and split every augmented view into labeled micro-bags."""

    micro_chunks = [list(chunk) for chunk in full_chunks[:original_count]]
    micro_labels = [int(v) for v in np.asarray(full_labels[:original_count], dtype=int)]
    for chunk, label in zip(full_chunks[original_count:], full_labels[original_count:]):
        bags = deterministic_subbags(
            chunk,
            target_hands=config.subbag_target_hands,
            split_threshold=config.subbag_split_threshold,
        )
        micro_chunks.extend(bags)
        micro_labels.extend([int(label)] * len(bags))
    return micro_chunks, np.asarray(micro_labels, dtype=int)
