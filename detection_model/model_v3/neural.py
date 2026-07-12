"""Compact hierarchical multiple-instance network for unordered hand sets.

Valid order exists only among actions inside one hand. Hands are content-sorted
before tensorization and pooled with symmetric operations, so received hand
order cannot affect the prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import hand_features


ACTION_TYPES = {"check": 1, "call": 2, "bet": 3, "raise": 4, "fold": 5}
STREETS = {"preflop": 1, "flop": 2, "turn": 3, "river": 4}
HAND_META_NAMES = sorted(hand_features({"metadata": {}, "players": [], "streets": [], "actions": []})[0])


def _f(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _stable_hands(hands: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (h for h in hands if isinstance(h, dict)),
        key=lambda h: hashlib.sha256(
            json.dumps(h, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).digest(),
    )


def _bucket(value: float) -> int:
    bounds = (0.0, 0.25, 0.75, 1.25, 2.25, 4.5, 9.0, 18.0, 50.0)
    return int(np.clip(np.searchsorted(bounds, max(0.0, value), side="right"), 1, 9))


def _encode_hand(hand: Dict[str, Any], max_actions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = hand.get("metadata") or {}
    hero = int(_f(meta.get("hero_seat"), 0))
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)][:max_actions]
    cat = np.zeros((max_actions, 4), dtype=np.int64)
    cont = np.zeros((max_actions, 3), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=np.bool_)
    for i, action in enumerate(actions):
        action_type = str(action.get("action_type") or "").lower()
        street = str(action.get("street") or "").lower()
        actor = int(_f(action.get("actor_seat"), 0))
        amount = max(0.0, _f(action.get("normalized_amount_bb")))
        pot_bb = max(0.0, _f(action.get("pot_after")) / 0.02)
        cat[i] = [
            ACTION_TYPES.get(action_type, 0), STREETS.get(street, 0),
            1 if hero > 0 and actor == hero else (2 if actor > 0 else 0), _bucket(amount),
        ]
        cont[i] = [math.log1p(amount), math.log1p(pot_bb), min(20.0, amount / max(0.25, pot_bb))]
        mask[i] = True
    return cat, cont, mask


def _hand_meta(hand: Dict[str, Any]) -> np.ndarray:
    row = hand_features(hand)[0]
    values = []
    for name in HAND_META_NAMES:
        value = float(row.get(name, 0.0))
        if name.startswith("n_"):
            value = min(value, 16.0) / 16.0
        elif "amount" in name or "pot" in name:
            value = math.log1p(max(0.0, value)) / 6.0
        elif "ratio" in name:
            value = min(max(value, 0.0), 20.0) / 20.0
        values.append(value)
    return np.asarray(values, dtype=np.float32)


@dataclass
class NeuralConfig:
    d_model: int = 48
    n_heads: int = 4
    action_layers: int = 1
    dropout: float = 0.20
    max_actions: int = 12
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 6e-4
    weight_decay: float = 2e-3
    patience: int = 4
    merged_ratio: float = 0.60
    merged_sources: int = 3
    merged_min_hands: int = 80
    merged_max_hands: int = 100
    log_every: int = 1
    tag: str = "neural"


class _ChunkDataset(Dataset):
    def __init__(self, chunks, labels):
        self.chunks = list(chunks); self.labels = np.asarray(labels, np.float32)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index], float(self.labels[index])


def _collate(batch, max_actions: int):
    chunks = [_stable_hands(item[0]) for item in batch]
    max_hands = max(1, max(len(c) for c in chunks))
    b = len(chunks)
    cat = np.zeros((b, max_hands, max_actions, 4), dtype=np.int64)
    cont = np.zeros((b, max_hands, max_actions, 3), dtype=np.float32)
    action_mask = np.zeros((b, max_hands, max_actions), dtype=np.bool_)
    hand_mask = np.zeros((b, max_hands), dtype=np.bool_)
    hand_meta = np.zeros((b, max_hands, len(HAND_META_NAMES)), dtype=np.float32)
    for bi, chunk in enumerate(chunks):
        for hi, hand in enumerate(chunk):
            c, n, m = _encode_hand(hand, max_actions)
            cat[bi, hi], cont[bi, hi], action_mask[bi, hi] = c, n, m
            hand_meta[bi, hi] = _hand_meta(hand)
            hand_mask[bi, hi] = bool(m.any())
    return {
        "cat": torch.from_numpy(cat), "cont": torch.from_numpy(cont),
        "action_mask": torch.from_numpy(action_mask), "hand_mask": torch.from_numpy(hand_mask),
        "hand_meta": torch.from_numpy(hand_meta),
        "labels": torch.tensor([item[1] for item in batch], dtype=torch.float32),
    }


class HierarchicalSetNet(nn.Module):
    def __init__(self, cfg: NeuralConfig):
        super().__init__(); d = cfg.d_model
        self.cfg = cfg
        self.action_type = nn.Embedding(6, d, padding_idx=0)
        self.street = nn.Embedding(5, d, padding_idx=0)
        self.actor_role = nn.Embedding(3, d, padding_idx=0)
        self.amount_bucket = nn.Embedding(10, d, padding_idx=0)
        self.position = nn.Embedding(cfg.max_actions, d)
        self.continuous = nn.Sequential(nn.Linear(3, d), nn.LayerNorm(d), nn.GELU())
        self.input_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d * 3,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(layer, num_layers=cfg.action_layers)
        self.action_gate = nn.Linear(d, 1)
        self.meta_projection = nn.Sequential(
            nn.Linear(len(HAND_META_NAMES), d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.hand_projection = nn.Sequential(
            nn.Linear(d * 4, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.set_phi = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d))
        self.hand_gate = nn.Linear(d, 1)
        self.head = nn.Sequential(
            nn.Linear(d * 4, d * 2), nn.LayerNorm(d * 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, 1),
        )

    def forward(self, cat, cont, action_mask, hand_mask, hand_meta):
        b, h, a, _ = cat.shape; d = self.cfg.d_model
        positions = torch.arange(a, device=cat.device).view(1, 1, a)
        x = (
            self.action_type(cat[..., 0]) + self.street(cat[..., 1])
            + self.actor_role(cat[..., 2]) + self.amount_bucket(cat[..., 3])
            + self.continuous(cont) + self.position(positions)
        )
        x = self.input_norm(x).reshape(b * h, a, d)
        am = action_mask.reshape(b * h, a)
        safe_am = am.clone(); safe_am[~safe_am.any(1), 0] = True
        x = self.action_encoder(x, src_key_padding_mask=~safe_am)
        mf = safe_am.unsqueeze(-1).float()
        mean = (x * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        maximum = x.masked_fill(~safe_am.unsqueeze(-1), -1e4).max(1).values
        gate = self.action_gate(x).squeeze(-1).masked_fill(~safe_am, -1e4)
        attention = (torch.softmax(gate, 1).unsqueeze(-1) * x).sum(1)
        meta = self.meta_projection(hand_meta).reshape(b * h, d)
        hands = self.hand_projection(torch.cat([mean, maximum, attention, meta], -1)).reshape(b, h, d)
        hands = self.set_phi(hands).masked_fill(~hand_mask.unsqueeze(-1), 0.0)
        hf = hand_mask.unsqueeze(-1).float(); denom = hf.sum(1).clamp(min=1.0)
        set_mean = (hands * hf).sum(1) / denom
        set_var = ((hands - set_mean.unsqueeze(1)).square() * hf).sum(1) / denom
        set_std = torch.sqrt(set_var + 1e-6)
        set_max = hands.masked_fill(~hand_mask.unsqueeze(-1), -1e4).max(1).values
        hgate = self.hand_gate(hands).squeeze(-1).masked_fill(~hand_mask, -1e4)
        set_attention = (torch.softmax(hgate, 1).unsqueeze(-1) * hands).sum(1)
        return self.head(torch.cat([set_mean, set_std, set_max, set_attention], -1)).squeeze(-1)


def merged_augmentation(chunks, labels, cfg: NeuralConfig, seed: int):
    rng = np.random.default_rng(seed); y = np.asarray(labels, int)
    count = int(round(len(chunks) * cfg.merged_ratio))
    augmented, aug_labels = [], []
    for _ in range(count):
        label = int(rng.integers(0, 2)); candidates = np.flatnonzero(y == label)
        if len(candidates) < cfg.merged_sources:
            continue
        selected = rng.choice(candidates, cfg.merged_sources, replace=False)
        pool = [hand for idx in selected for hand in chunks[int(idx)]]
        if len(pool) < cfg.merged_min_hands:
            continue
        target = min(len(pool), int(rng.integers(cfg.merged_min_hands, cfg.merged_max_hands + 1)))
        chosen = rng.choice(len(pool), target, replace=False)
        augmented.append([pool[int(i)] for i in chosen]); aug_labels.append(label)
    return augmented, np.asarray(aug_labels, int)


class HierarchicalSetModel:
    """Pickle-friendly sklearn-style wrapper around the hierarchical network."""

    def __init__(self, config: Optional[NeuralConfig] = None, seed: int = 44, device: str = "cpu"):
        self.config = config or NeuralConfig(); self.seed = int(seed); self.device = str(device)
        self.state_dict_: Optional[Dict[str, torch.Tensor]] = None

    def fit(self, chunks, labels):
        started = time.perf_counter()
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        y = np.asarray(labels, int); rng = np.random.default_rng(self.seed)
        train_idx, val_idx = [], []
        for label in (0, 1):
            idx = np.flatnonzero(y == label); rng.shuffle(idx)
            nval = max(1, int(round(0.12 * len(idx))))
            val_idx.extend(idx[:nval]); train_idx.extend(idx[nval:])
        tr_chunks = [chunks[i] for i in train_idx]; tr_y = y[train_idx]
        original_train_count = len(tr_chunks)
        aug, aug_y = merged_augmentation(tr_chunks, tr_y, self.config, self.seed + 991)
        tr_chunks = tr_chunks + aug; tr_y = np.concatenate([tr_y, aug_y])
        va_chunks = [chunks[i] for i in val_idx]; va_y = y[val_idx]
        collate = lambda batch: _collate(batch, self.config.max_actions)
        train_loader = DataLoader(
            _ChunkDataset(tr_chunks, tr_y), batch_size=self.config.batch_size,
            shuffle=True, collate_fn=collate,
        )
        val_loader = DataLoader(
            _ChunkDataset(va_chunks, va_y), batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate,
        )
        model = HierarchicalSetNet(self.config).to(self.device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if self.config.log_every > 0:
            print(
                f"[{self.config.tag}] start | device={self.device} params={parameter_count:,} "
                f"train_original={original_train_count} train_merged={len(aug)} "
                f"train_total={len(tr_chunks)} validation={len(va_chunks)} "
                f"epochs={self.config.epochs} batch={self.config.batch_size}",
                flush=True,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        bce = nn.BCEWithLogitsLoss(); best = float("inf"); best_state = None; bad = 0
        best_epoch = 0
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            train_total = 0.0; train_bce = 0.0; train_pair = 0.0; train_count = 0
            for batch in train_loader:
                logits = model(*(batch[k].to(self.device) for k in ("cat", "cont", "action_mask", "hand_mask", "hand_meta")))
                labels_b = batch["labels"].to(self.device)
                base = bce(logits, labels_b)
                pos, neg = logits[labels_b == 1], logits[labels_b == 0]
                pair = torch.nn.functional.softplus(-(pos[:, None] - neg[None, :])).mean() if len(pos) and len(neg) else base * 0
                loss = 0.65 * base + 0.35 * pair
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                count = len(labels_b)
                train_total += float(loss.detach()) * count
                train_bce += float(base.detach()) * count
                train_pair += float(pair.detach()) * count
                train_count += count
            val_loss = self._loss(model, val_loader, bce)
            improved = val_loss < best - 1e-4
            if improved:
                best = val_loss; bad = 0; best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if self.config.log_every > 0 and (
                epoch == 1 or epoch % self.config.log_every == 0 or improved or epoch == self.config.epochs
            ):
                print(
                    f"[{self.config.tag}] epoch {epoch:02d}/{self.config.epochs} | "
                    f"loss={train_total/max(1, train_count):.4f} "
                    f"bce={train_bce/max(1, train_count):.4f} "
                    f"pair={train_pair/max(1, train_count):.4f} "
                    f"val={val_loss:.4f} {'*best' if improved else ''} "
                    f"bad={bad}/{self.config.patience} elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
            if bad >= self.config.patience:
                if self.config.log_every > 0:
                    print(
                        f"[{self.config.tag}] early-stop | epoch={epoch} "
                        f"best_epoch={best_epoch} best_val={best:.4f}",
                        flush=True,
                    )
                break
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.state_dict_ = best_state
        if self.config.log_every > 0:
            print(
                f"[{self.config.tag}] done | best_epoch={best_epoch or self.config.epochs} "
                f"best_val={best:.4f} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
        return self

    def _loss(self, model, loader, loss_fn):
        model.eval(); total = count = 0
        with torch.no_grad():
            for batch in loader:
                logits = model(*(batch[k].to(self.device) for k in ("cat", "cont", "action_mask", "hand_mask", "hand_meta")))
                loss = loss_fn(logits, batch["labels"].to(self.device))
                total += float(loss) * len(batch["labels"]); count += len(batch["labels"])
        return total / max(1, count)

    def _model(self):
        if self.state_dict_ is None:
            raise RuntimeError("hierarchical model is not fitted")
        model = HierarchicalSetNet(self.config); model.load_state_dict(self.state_dict_); model.to(self.device); model.eval()
        return model

    def predict_proba(self, chunks):
        model = self._model(); collate = lambda batch: _collate(batch, self.config.max_actions)
        loader = DataLoader(_ChunkDataset(chunks, np.zeros(len(chunks))), batch_size=self.config.batch_size, shuffle=False, collate_fn=collate)
        values = []
        with torch.no_grad():
            for batch in loader:
                logits = model(*(batch[k].to(self.device) for k in ("cat", "cont", "action_mask", "hand_mask", "hand_meta")))
                values.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        p = np.asarray(values, float)
        return np.column_stack([1 - p, p])
