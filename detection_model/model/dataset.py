from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChunkSample:
    chunk: List[Dict[str, Any]]
    label: int  # 0 = human, 1 = bot
    chunk_id: str | None = None


def _open_json_or_gz(path: str | Path) -> Any:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_hand(obj: Any) -> bool:
    return isinstance(obj, dict) and {"metadata", "players", "streets", "actions", "outcome"}.issubset(obj.keys())


def _normalize_label_value(value: Any) -> int:
    """Return 0 for human, 1 for bot."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return 1 if value == 1 else 0
    if isinstance(value, float):
        return 1 if value >= 0.5 else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"human", "person", "real", "player", "0", "false", "true_human"}:
            return 0
        if text in {"bot", "ai", "machine", "synthetic", "generated", "agent", "1", "true", "is_bot", "false_human"}:
            return 1
    raise ValueError(f"Unsupported label value: {value!r}")


def _extract_label(item: Dict[str, Any]) -> int:
    # Explicit bot/human flags are safest.
    if "is_bot" in item:
        return 1 if bool(item["is_bot"]) else 0
    if "is_human" in item:
        return 0 if bool(item["is_human"]) else 1
    if "bot" in item:
        return 1 if bool(item["bot"]) else 0
    if "human" in item:
        return 0 if bool(item["human"]) else 1

    for key in ("label", "target", "class", "class_name", "kind", "source_type"):
        if key in item:
            return _normalize_label_value(item[key])

    # Some generated benchmark formats nest label metadata.
    for key in ("label_info", "labels", "metadata", "target_info"):
        nested = item.get(key)
        if isinstance(nested, dict):
            try:
                return _extract_label(nested)
            except Exception:
                pass

    raise ValueError(f"Could not infer label from item keys: {list(item.keys())}")


def _extract_chunk(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    if _looks_like_hand(item):
        return [item]

    for key in (
        "chunk", "chunks", "hands", "sanitized_hands", "sanitizedHands",
        "hand_payloads", "payload", "data", "batch",
    ):
        value = item.get(key)
        if isinstance(value, list):
            if not value:
                return []
            if all(isinstance(x, dict) for x in value):
                return value
            if len(value) == 1 and isinstance(value[0], list):
                return value[0]

    for wrapper_key in ("payload", "data", "batch", "body"):
        wrapper = item.get(wrapper_key)
        if isinstance(wrapper, dict):
            for key in ("chunk", "chunks", "hands", "sanitized_hands", "sanitizedHands", "hand_payloads"):
                value = wrapper.get(key)
                if isinstance(value, list):
                    if not value:
                        return []
                    if all(isinstance(x, dict) for x in value):
                        return value
                    if len(value) == 1 and isinstance(value[0], list):
                        return value[0]

    raise ValueError(f"Could not infer chunk/hands from item keys: {list(item.keys())}")


def _items_to_samples(items: List[Any]) -> List[ChunkSample]:
    samples: List[ChunkSample] = []
    skipped = 0
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            chunk = _extract_chunk(item)
            label = _extract_label(item)
            if not chunk:
                skipped += 1
                continue
            chunk_id = (
                item.get("chunk_id")
                or item.get("id")
                or item.get("sample_id")
                or item.get("uid")
                or f"sample_{idx}"
            )
            samples.append(ChunkSample(chunk=chunk, label=label, chunk_id=str(chunk_id)))
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"[dataset] skipped item {idx}: {exc}")
    if skipped:
        print(f"[dataset] skipped total items: {skipped}")
    return samples


def _split_items_by_split_field(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []
    for item in items:
        split = str(item.get("split", item.get("dataset_split", item.get("partition", "")))).strip().lower()
        if split in {"validation", "valid", "val", "dev", "eval", "test"}:
            val_items.append(item)
        elif split in {"train", "training"}:
            train_items.append(item)
    return train_items, val_items


def _extract_root_items(obj: Any) -> Tuple[Optional[List[Any]], Optional[List[Any]], Optional[List[Any]]]:
    if isinstance(obj, list):
        return None, None, obj
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported dataset root type: {type(obj)}")

    train_items = obj.get("train") or obj.get("training")
    val_items = obj.get("validation") or obj.get("valid") or obj.get("val") or obj.get("dev") or obj.get("test")
    if isinstance(train_items, list):
        return train_items, val_items if isinstance(val_items, list) else None, None

    splits = obj.get("splits")
    if isinstance(splits, dict):
        train_items = splits.get("train") or splits.get("training")
        val_items = splits.get("validation") or splits.get("valid") or splits.get("val") or splits.get("dev") or splits.get("test")
        if isinstance(train_items, list):
            return train_items, val_items if isinstance(val_items, list) else None, None

    labeled_chunks = obj.get("labeled_chunks")
    if isinstance(labeled_chunks, list):
        train_items, val_items = _split_items_by_split_field(labeled_chunks)
        if train_items or val_items:
            return train_items, val_items, None
        return None, None, labeled_chunks

    if isinstance(labeled_chunks, dict):
        train_items = labeled_chunks.get("train") or labeled_chunks.get("training")
        val_items = labeled_chunks.get("validation") or labeled_chunks.get("valid") or labeled_chunks.get("val") or labeled_chunks.get("dev") or labeled_chunks.get("test")
        if isinstance(train_items, list):
            return train_items, val_items if isinstance(val_items, list) else None, None
        for key in ("samples", "data", "chunks", "items"):
            if isinstance(labeled_chunks.get(key), list):
                return None, None, labeled_chunks[key]

    for key in ("samples", "data", "chunks", "items", "records"):
        value = obj.get(key)
        if isinstance(value, list):
            return None, None, value

    raise ValueError(f"Unsupported benchmark dict keys: {list(obj.keys())}")


def _random_split(samples: List[ChunkSample], val_ratio: float, seed: int) -> Tuple[List[ChunkSample], List[ChunkSample]]:
    """Deterministic stratified split when both classes are present."""

    if not samples:
        return [], []

    rng = random.Random(seed)
    by_label: Dict[int, List[ChunkSample]] = {}
    for sample in samples:
        by_label.setdefault(int(sample.label), []).append(sample)

    if len(by_label) < 2:
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        return shuffled[n_val:], shuffled[:n_val]

    train: List[ChunkSample] = []
    val: List[ChunkSample] = []

    for label, label_samples in sorted(by_label.items()):
        label_samples = list(label_samples)
        rng.shuffle(label_samples)
        n_val = max(1, int(round(len(label_samples) * val_ratio)))
        n_val = min(n_val, max(1, len(label_samples) - 1)) if len(label_samples) > 1 else 1
        val.extend(label_samples[:n_val])
        train.extend(label_samples[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _print_dataset_summary(name: str, samples: List[ChunkSample]) -> None:
    if not samples:
        print(f"[dataset] {name}: 0 samples")
        return
    labels = [s.label for s in samples]
    chunk_sizes = [len(s.chunk) for s in samples]
    print(
        f"[dataset] {name}: total={len(samples)}, "
        f"human={labels.count(0)}, bot={labels.count(1)}, "
        f"avg_chunk_size={np.mean(chunk_sizes):.2f}, max_chunk_size={max(chunk_sizes)}"
    )


def load_public_benchmark(path: str | Path, val_ratio: float = 0.15, seed: int = 44) -> Tuple[List[ChunkSample], List[ChunkSample]]:
    obj = _open_json_or_gz(path)
    train_items, val_items, all_items = _extract_root_items(obj)
    if train_items is not None:
        train_samples = _items_to_samples(train_items)
        val_samples = _items_to_samples(val_items or [])
        if not val_samples:
            train_samples, val_samples = _random_split(train_samples, val_ratio, seed)
    else:
        all_samples = _items_to_samples(all_items or [])
        train_samples, val_samples = _random_split(all_samples, val_ratio, seed)

    if not train_samples:
        raise ValueError("No training samples found.")
    if not val_samples:
        raise ValueError("No validation samples found.")

    _print_dataset_summary("train", train_samples)
    _print_dataset_summary("validation", val_samples)

    if len({s.label for s in train_samples}) < 2:
        print("[dataset warning] Training set has one class only. You probably loaded human-only data.")
    if len({s.label for s in val_samples}) < 2:
        print("[dataset warning] Validation set has one class only. Metrics will be unstable.")
    return train_samples, val_samples


def augment_chunk_prefixes(
    samples: List[ChunkSample],
    min_prefix_hands: int = 4,
    max_prefixes_per_chunk: Optional[int] = 32,
    include_full_chunk: bool = True,
) -> List[ChunkSample]:
    """
    Prefix augmentation. Keep hand order.

    Example:
      [h1,h2,h3,h4,h5], label=bot
      -> [h1,h2,h3,h4] label=bot
      -> [h1,h2,h3,h4,h5] label=bot

    Split train/validation BEFORE using this function.
    """
    augmented: List[ChunkSample] = []
    min_prefix_hands = max(1, int(min_prefix_hands))

    for sample in samples:
        chunk = sample.chunk
        n = len(chunk)
        if n == 0:
            continue
        if n < min_prefix_hands:
            if include_full_chunk:
                augmented.append(sample)
            continue

        prefix_lengths = list(range(min_prefix_hands, n + 1))
        if max_prefixes_per_chunk is not None and len(prefix_lengths) > max_prefixes_per_chunk:
            max_prefixes = max(1, int(max_prefixes_per_chunk))
            selected = []
            step = len(prefix_lengths) / max_prefixes
            for i in range(max_prefixes):
                selected.append(prefix_lengths[min(int(i * step), len(prefix_lengths) - 1)])
            if include_full_chunk:
                selected.append(n)
            prefix_lengths = sorted(set(selected))

        for prefix_len in prefix_lengths:
            if prefix_len == n and not include_full_chunk:
                continue
            augmented.append(ChunkSample(chunk=chunk[:prefix_len], label=sample.label, chunk_id=f"{sample.chunk_id or 'chunk'}:prefix:{prefix_len}"))
    return augmented


def augment_chunk_sliding_windows(
    samples: List[ChunkSample],
    window_hands: int = 8,
    stride: int = 2,
    max_windows_per_chunk: Optional[int] = 32,
    include_full_chunk: bool = True,
) -> List[ChunkSample]:
    """
    Sliding-window augmentation over sequential hands.

    Example:
      [h1,h2,h3,h4,h5,h6], window=4, stride=1
      -> [h1..h4], [h2..h5], [h3..h6]
    """
    augmented: List[ChunkSample] = []
    window_hands = max(1, int(window_hands))
    stride = max(1, int(stride))

    for sample in samples:
        chunk = sample.chunk
        n = len(chunk)
        if n == 0:
            continue
        if include_full_chunk:
            augmented.append(sample)
        if n < window_hands:
            continue

        starts = list(range(0, n - window_hands + 1, stride))
        if max_windows_per_chunk is not None and len(starts) > max_windows_per_chunk:
            max_windows = max(1, int(max_windows_per_chunk))
            step = len(starts) / max_windows
            starts = [starts[min(int(i * step), len(starts) - 1)] for i in range(max_windows)]
            starts = sorted(set(starts))

        for start in starts:
            augmented.append(ChunkSample(chunk=chunk[start:start + window_hands], label=sample.label, chunk_id=f"{sample.chunk_id or 'chunk'}:window:{start}:{start + window_hands}"))
    return augmented


def augment_chunk_windows(
    samples: list[ChunkSample],
    window_hands: int = 4,
    stride: int = 1,
    keep_short_chunks: bool = False,
) -> list[ChunkSample]:
    """
    Generate fixed-length consecutive hand windows from each chunk.

    Example:
        chunk = [h1, h2, h3, h4, h5, h6]
        window_hands = 4
        stride = 1

        generated:
            [h1, h2, h3, h4]
            [h2, h3, h4, h5]
            [h3, h4, h5, h6]

    The generated window keeps the original chunk label:
        bot chunk   -> all windows label = 1
        human chunk -> all windows label = 0

    Important:
        Split train/validation first, then apply this separately.
    """

    if window_hands <= 0:
        raise ValueError(f"window_hands must be > 0, got {window_hands}")

    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    augmented: list[ChunkSample] = []

    for sample_idx, sample in enumerate(samples):
        chunk = sample.chunk
        label = sample.label

        n = len(chunk)

        if n == 0:
            continue

        if n < window_hands:
            if keep_short_chunks:
                augmented.append(
                    ChunkSample(
                        chunk=chunk,
                        label=label,
                        chunk_id=f"{sample.chunk_id or sample_idx}:short",
                    )
                )
            continue

        for start in range(0, n - window_hands + 1, stride):
            end = start + window_hands

            augmented.append(
                ChunkSample(
                    chunk=chunk[start:end],
                    label=label,
                    chunk_id=f"{sample.chunk_id or sample_idx}:window:{start}:{end}",
                )
            )
        
        # last_start = n - window_hands
        # if last_start % stride != 0:
        #     augmented.append(
        #         ChunkSample(
        #             chunk=chunk[last_start:last_start + window_hands],
        #             label=label,
        #         )
        #     )

    return augmented