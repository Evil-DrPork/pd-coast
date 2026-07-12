from __future__ import annotations

import random

import torch

from detection_model.model_v3.neural import HierarchicalSetNet, NeuralConfig, _collate


def _hand(action):
    return {
        "metadata": {"hero_seat": 1}, "players": [{"seat": 1}, {"seat": 2}],
        "streets": [{"street": "preflop"}],
        "actions": [{
            "action_id": "1", "street": "preflop", "actor_seat": 1,
            "action_type": action, "normalized_amount_bb": 1.0, "pot_after": 0.04,
        }],
    }


def test_hierarchical_network_is_hand_permutation_invariant():
    torch.manual_seed(44)
    cfg = NeuralConfig(d_model=24, n_heads=4, dropout=0.0, max_actions=4)
    model = HierarchicalSetNet(cfg).eval()
    chunk = [_hand("check"), _hand("raise"), _hand("fold")]
    shuffled = list(chunk); random.Random(9).shuffle(shuffled)
    a = _collate([(chunk, 0.0)], cfg.max_actions)
    b = _collate([(shuffled, 0.0)], cfg.max_actions)
    with torch.no_grad():
        pa = model(a["cat"], a["cont"], a["action_mask"], a["hand_mask"], a["hand_meta"])
        pb = model(b["cat"], b["cont"], b["action_mask"], b["hand_mask"], b["hand_meta"])
    assert torch.equal(pa, pb)
