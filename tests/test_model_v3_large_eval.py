from __future__ import annotations

import numpy as np

from detection_model.model_v3.evaluate_large import build_merged_batch
from detection_model.model_v3.schema import Chunk


def _chunk(idx: int, label: int) -> Chunk:
    hands = [{"metadata": {}, "players": [], "streets": [], "actions": [], "marker": f"{idx}:{j}"} for j in range(35)]
    return Chunk(idx, hands, label, "2026-07-12")


def test_merged_batch_has_target_size_labels_and_unique_sources():
    chunks = [_chunk(i, 0) for i in range(40)] + [_chunk(100 + i, 1) for i in range(40)]
    merged, labels, stats = build_merged_batch(
        chunks, batch_size=20, bot_fraction=0.5, sources_per_chunk=3,
        min_hands=80, max_hands=100, rng=np.random.default_rng(44),
    )
    assert len(merged) == 20
    assert int(labels.sum()) == 10
    assert all(80 <= len(hands) <= 100 for hands in merged)
    assert stats["unique_sources_used"] == 60

