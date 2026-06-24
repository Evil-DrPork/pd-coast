from __future__ import annotations

import copy
import hashlib
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .action_vectorizer import CAT_DIM, HAND_META_DIM


def _sample_visible_indices(
    total: int,
    *,
    window_size: int,
    seed_parts: List[str],
    actions: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Match validator-style deterministic visible-action sampling.

    Keeps first/last action, at least one action from each street bucket when
    possible, then deterministic sampled middle actions. Chronological order is
    preserved.
    """
    window_size = max(1, int(window_size))
    if total <= 0:
        return []
    if total == 1:
        return [0]
    if total <= window_size:
        return list(range(total))

    seed = "|".join(seed_parts).encode("utf-8", errors="ignore")

    def _sort_key(index: int, extra: str = "") -> bytes:
        return hashlib.sha256(seed + f":{index}:{extra}".encode("utf-8")).digest()

    picked = {0, total - 1}
    if actions:
        street_buckets: Dict[str, List[int]] = {}
        for idx in range(1, total - 1):
            action = actions[idx] if idx < len(actions) else {}
            street = str(action.get("street", "") or "preflop").lower()
            street_buckets.setdefault(street, []).append(idx)

        for street in sorted(street_buckets.keys()):
            if len(picked) >= window_size:
                break
            ordered = sorted(street_buckets[street], key=lambda idx: _sort_key(idx, street))
            if ordered:
                picked.add(ordered[0])

    middle = [idx for idx in range(1, total - 1) if idx not in picked]
    for idx in sorted(middle, key=_sort_key):
        if len(picked) >= window_size:
            break
        picked.add(idx)

    return sorted(picked)


def calibrate_hand_visible_actions(
    hand: Dict[str, Any],
    *,
    window_size: int,
    seed_parts: List[str],
) -> Dict[str, Any]:
    if not isinstance(hand, dict):
        return hand
    actions = hand.get("actions") or []
    if not isinstance(actions, list) or not actions:
        return hand

    indices = _sample_visible_indices(
        total=len(actions),
        window_size=window_size,
        seed_parts=seed_parts,
        actions=actions,
    )
    sampled_actions = [actions[idx] for idx in indices if 0 <= idx < len(actions)]
    new_hand = copy.deepcopy(hand)
    new_hand["actions"] = sampled_actions
    return new_hand


def calibrate_chunk_visible_actions(
    chunk: List[Dict[str, Any]],
    *,
    window_size: int,
    chunk_id: str,
) -> List[Dict[str, Any]]:
    calibrated: List[Dict[str, Any]] = []
    for hand_idx, hand in enumerate(chunk):
        actions = hand.get("actions") or [] if isinstance(hand, dict) else []
        seed_parts = [str(chunk_id), f"hand_{hand_idx}", f"actions_{len(actions)}"]
        calibrated.append(
            calibrate_hand_visible_actions(
                hand,
                window_size=window_size,
                seed_parts=seed_parts,
            )
        )
    return calibrated


class HierarchicalPokerChunkDataset(Dataset):
    def __init__(
        self,
        samples,
        action_vectorizer,
        feature_vectorizer,
        max_hands: int,
        calibrate_visible_actions: bool = False,
        min_visible_action_window_size: int = 5,
        max_visible_action_window_size: int = 8,
        recompute_features_after_calibration: bool = True,
    ):
        self.samples = samples
        self.action_vectorizer = action_vectorizer
        self.feature_vectorizer = feature_vectorizer
        self.max_hands = int(max_hands)
        self.calibrate_visible_actions = bool(calibrate_visible_actions)
        self.min_visible_action_window_size = max(1, int(min_visible_action_window_size))
        self.max_visible_action_window_size = max(1, int(max_visible_action_window_size))
        if self.max_visible_action_window_size < self.min_visible_action_window_size:
            self.max_visible_action_window_size = self.min_visible_action_window_size
        self.recompute_features_after_calibration = bool(recompute_features_after_calibration)

        chunks = [sample.chunk for sample in samples]
        self.feature_matrix = feature_vectorizer.transform(chunks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        chunk = sample.chunk
        chunk_id = getattr(sample, "chunk_id", None) or f"sample_{idx}"

        if self.calibrate_visible_actions:
            window_size = random.randint(
                self.min_visible_action_window_size,
                self.max_visible_action_window_size,
            )
            chunk = calibrate_chunk_visible_actions(
                chunk,
                window_size=window_size,
                chunk_id=str(chunk_id),
            )

        encoded = self.action_vectorizer.encode_chunk(chunk, max_hands=self.max_hands)

        if self.calibrate_visible_actions and self.recompute_features_after_calibration:
            features = self.feature_vectorizer.transform([chunk])[0]
        else:
            features = self.feature_matrix[idx]

        return {
            "action_cat": encoded["cat"],
            "action_num": encoded["num"],
            "hand_meta": encoded["hand_meta"],
            "hand_end": encoded["hand_end"],
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(float(sample.label), dtype=torch.float32),
            "num_hands": torch.tensor(len(encoded["cat"]), dtype=torch.long),
        }


def hierarchical_collate_batch(
    batch: List[Dict[str, Any]],
    cat_pad_id: int = 0,
    numeric_dim: int = 22,
    meta_dim: int = HAND_META_DIM,
) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_hands = max(1, max(len(item["action_cat"]) for item in batch))
    max_actions = max(1, max(len(hand_actions) for item in batch for hand_actions in item["action_cat"]))

    action_cat = torch.full((batch_size, max_hands, max_actions, CAT_DIM), cat_pad_id, dtype=torch.long)
    action_num = torch.zeros((batch_size, max_hands, max_actions, numeric_dim), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_hands, max_actions), dtype=torch.bool)
    hand_mask = torch.zeros((batch_size, max_hands), dtype=torch.bool)
    hand_meta = torch.zeros((batch_size, max_hands, meta_dim), dtype=torch.float32)
    hand_end = torch.zeros((batch_size, max_hands), dtype=torch.long)

    features = torch.stack([item["features"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    for batch_idx, item in enumerate(batch):
        metas = item.get("hand_meta") or []
        ends = item.get("hand_end") or []
        for hand_idx, (cat_rows, num_rows) in enumerate(zip(item["action_cat"], item["action_num"])):
            if hand_idx >= max_hands:
                break
            length = min(len(cat_rows), max_actions)
            if length <= 0:
                continue

            cat_tensor = torch.tensor(cat_rows[:length], dtype=torch.long)
            action_cat[batch_idx, hand_idx, :length, :] = cat_tensor[:, :CAT_DIM]

            num_tensor = torch.tensor(num_rows[:length], dtype=torch.float32)
            if num_tensor.ndim == 1:
                num_tensor = num_tensor.unsqueeze(0)
            num_d = min(num_tensor.shape[-1], numeric_dim)
            action_num[batch_idx, hand_idx, :length, :num_d] = num_tensor[:, :num_d]

            action_mask[batch_idx, hand_idx, :length] = True
            hand_mask[batch_idx, hand_idx] = True

            if hand_idx < len(metas):
                meta_row = torch.tensor(metas[hand_idx], dtype=torch.float32)
                meta_d = min(meta_row.shape[-1], meta_dim)
                hand_meta[batch_idx, hand_idx, :meta_d] = meta_row[:meta_d]
            if hand_idx < len(ends):
                hand_end[batch_idx, hand_idx] = int(ends[hand_idx])

    return {
        "action_cat": action_cat,
        "action_num": action_num,
        "action_mask": action_mask,
        "hand_mask": hand_mask,
        "hand_meta": hand_meta,
        "hand_end": hand_end,
        "features": features,
        "labels": labels,
    }
