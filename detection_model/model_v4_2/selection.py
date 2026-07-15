"""Deterministic V4.2 request preprocessing."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from detection_model.model_v3.schema import clean_hand
from detection_model.model_v4.mapping import chunk_tie_key


HAND_CAP = 80
SELECTION_ALGORITHM = "v42_policy_v1"


def hand_behavior_key(hand: Mapping[str, Any]) -> str:
    """Return the stable public-payload key used by this policy."""
    return chunk_tie_key((hand,))


def _clean_chunks(
    chunks: Sequence[Sequence[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    return [
        [clean_hand(hand) for hand in (chunk or []) if isinstance(hand, dict)]
        for chunk in chunks
    ]


def _select_chunk(chunk: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keyed = [(hand_behavior_key(hand), hand) for hand in chunk]
    keyed.sort(key=lambda item: item[0])
    return [hand for _, hand in keyed[:HAND_CAP]]


def prepare_chunks(
    chunks: Sequence[Sequence[Dict[str, Any]]],
) -> tuple[List[List[Dict[str, Any]]], dict[str, Any]]:
    """Prepare a complete request and return policy diagnostics."""
    clean = _clean_chunks(chunks)
    original_counts = [len(chunk) for chunk in clean]
    applied = bool(clean) and all(count >= HAND_CAP for count in original_counts)

    if applied:
        selected = [_select_chunk(chunk) for chunk in clean]
        fallback_reason = None
    else:
        selected = clean
        fallback_reason = "empty_request" if not clean else "chunk_below_hand_cap"

    selected_counts = [len(chunk) for chunk in selected]
    dropped_counts = [
        original - retained
        for original, retained in zip(original_counts, selected_counts)
    ]
    diagnostics = {
        "algorithm": SELECTION_ALGORITHM,
        "hand_cap": HAND_CAP,
        "applied": applied,
        "fallback_reason": fallback_reason,
        "original_hand_counts": original_counts,
        "selected_hand_counts": selected_counts,
        "dropped_hand_counts": dropped_counts,
        "truncated_chunk_count": sum(count > 0 for count in dropped_counts),
        "dropped_hand_count": sum(dropped_counts),
    }
    return selected, diagnostics
