"""Strict loading and canonicalization for v3.

Chunk order is preserved only so predictions can be returned in request order.
Hand order is never used as a feature. Action order inside a hand is retained.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: int
    hands: List[Dict[str, Any]]
    label: Optional[int] = None
    source_date: Optional[str] = None


def _action_key(action: Dict[str, Any], fallback: int) -> tuple[int, int]:
    try:
        return int(action.get("action_id")), fallback
    except (TypeError, ValueError):
        return fallback, fallback


def clean_hand(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Defensively normalize a miner-visible hand without inventing behavior."""
    out = dict(hand)
    raw_actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
    actions = [dict(a) for a in raw_actions if isinstance(a, dict)]
    actions = [a for _, a in sorted(enumerate(actions), key=lambda z: _action_key(z[1], z[0]))]
    out["actions"] = actions
    out["metadata"] = dict(hand.get("metadata") or {})
    out["players"] = [dict(p) for p in (hand.get("players") or []) if isinstance(p, dict)]
    out["streets"] = [dict(s) for s in (hand.get("streets") or []) if isinstance(s, dict)]
    return out


def looks_canonical(hand: Dict[str, Any]) -> bool:
    """Conservative recognition of the validator-visible payload shape."""
    meta = hand.get("metadata") or {}
    outcome = hand.get("outcome") or {}
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    return (
        abs(_finite(meta.get("bb"), -1.0) - 0.02) < 1e-9
        and str(meta.get("hand_ended_on_street") or "") == ""
        and not outcome.get("winners")
        and not outcome.get("payouts")
        and all(not p.get("hole_cards") for p in players if isinstance(p, dict))
        and all(str(a.get("action_id", "")).isdigit() for a in actions if isinstance(a, dict))
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _load_canonicalizer(repo_root: Path):
    path = repo_root / "poker44" / "validator" / "payload_view.py"
    if not path.exists():
        raise FileNotFoundError(f"validator canonicalizer not found: {path}")
    spec = importlib.util.spec_from_file_location("p44_v3_payload_view", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import canonicalizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_hand_for_miner


def canonicalize_chunks(chunks: List[Chunk], repo_root: Path, mode: str = "auto") -> List[Chunk]:
    """Apply the exact validator canonicalizer to raw hands.

    ``auto`` leaves already canonical miner payloads untouched. ``always`` is
    useful for raw platform exports; ``never`` is intended for live requests.
    """
    mode = mode.lower()
    if mode not in {"auto", "always", "never"}:
        raise ValueError("canonicalize mode must be auto, always, or never")
    fn = None if mode == "never" else _load_canonicalizer(repo_root)
    out: List[Chunk] = []
    for chunk in chunks:
        hands: List[Dict[str, Any]] = []
        for hand in chunk.hands:
            visible = hand
            if fn is not None and (mode == "always" or not looks_canonical(hand)):
                visible = fn(hand)
            if isinstance(visible, dict):
                hands.append(clean_hand(visible))
        out.append(Chunk(chunk.chunk_id, hands, chunk.label, chunk.source_date))
    return out


def load_chunks(path: str | Path) -> List[Chunk]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "labeled_chunks" in payload:
        payload = payload["labeled_chunks"]
    if not isinstance(payload, list):
        raise ValueError("top-level dataset must be a list")
    chunks: List[Chunk] = []
    for idx, item in enumerate(payload):
        if isinstance(item, dict):
            raw_hands = item.get("hands") or []
            label_raw = item.get("is_bot")
            label = None if label_raw is None else int(bool(label_raw))
            source_date = item.get("source_date")
        elif isinstance(item, list):
            raw_hands, label, source_date = item, None, None
        else:
            continue
        hands = [clean_hand(h) for h in raw_hands if isinstance(h, dict)]
        chunks.append(Chunk(idx, hands, label, source_date))
    if not chunks:
        raise ValueError("dataset contains no usable chunks")
    return chunks


def chunk_multiset_hash(chunk: Chunk) -> str:
    hand_hashes = []
    for hand in chunk.hands:
        raw = json.dumps(hand, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        hand_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
    raw = "|".join(sorted(hand_hashes))
    return hashlib.sha256(raw.encode()).hexdigest()

