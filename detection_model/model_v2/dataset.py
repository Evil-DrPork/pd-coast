"""Build chunk-level feature matrices from either file format."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .features import chunk_feature_vector, feature_names_for
from .schema import Chunk, load_chunks


def build_feature_matrix(
    path: str | Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], List[int]]:
    """Return (X, y_or_None, feature_names, chunk_ids).

    ``y`` is None when the file carries no labels (evaluation set). Column order
    is the fixed :func:`feature_names_for` order so train and serve always align.
    """
    chunks: List[Chunk] = load_chunks(path)
    names = feature_names_for()
    rows: List[List[float]] = []
    labels: List[Optional[int]] = []
    ids: List[int] = []

    for ch in chunks:
        feats = chunk_feature_vector(ch.hands)
        rows.append([float(feats.get(name, 0.0)) for name in names])
        labels.append(ch.label)
        ids.append(ch.chunk_id)

    x = np.asarray(rows, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    has_labels = all(v is not None for v in labels) and len(labels) > 0
    y = np.asarray([int(v) for v in labels], dtype=np.int64) if has_labels else None
    return x, y, names, ids


def chunk_split(
    n: int, val_ratio: float = 0.2, seed: int = 44
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic chunk-level train/val index split (never split by hand)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(val_ratio * n)))
    return order[n_val:], order[:n_val]


def features_for_chunks(chunks, names: List[str]) -> np.ndarray:
    """Feature matrix for a list of raw chunks, aligned to ``names``.

    NOTE: compute chunk_feature_vector ONCE per chunk (not once per name) — the
    previous nested-comprehension form recomputed it len(names)x per chunk (~345x
    slower), which is what made every augmented/walk-forward run crawl.
    """
    rows = []
    for c in chunks:
        feats = chunk_feature_vector(c)
        rows.append([float(feats.get(n, 0.0)) for n in names])
    x = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(names)))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def multiscale_augment(chunks, y, per_chunk: int, size_range=(15, 30), seed: int = 44):
    """Single-entity size-variety augmentation (the SAFE alternative to pooling).

    For each chunk, draw ``per_chunk`` random sub-samples at varying smaller sizes
    from THAT chunk's own hands only. Because every sub-sample stays single-entity,
    the within-entity consistency signal is preserved (unlike pooling, which mixes
    entities and destroys it). Teaches size-invariance. Returns (aug_chunks, labels).
    """
    if per_chunk <= 0:
        return [], np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    lo, hi = int(size_range[0]), int(size_range[1])
    aug_chunks, aug_y = [], []
    for i, chunk in enumerate(chunks):
        n = len(chunk)
        if n < lo + 2:                      # too small to sub-sample meaningfully
            continue
        top = min(hi, n - 1)
        for _ in range(per_chunk):
            size = int(rng.integers(lo, top + 1))
            sel = rng.choice(n, size=size, replace=False)
            aug_chunks.append([chunk[j] for j in sel])
            aug_y.append(int(y[i]))
    return aug_chunks, np.asarray(aug_y, dtype=int)
