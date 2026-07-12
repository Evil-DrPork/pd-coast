# model_v2 Upgrade Workflow — New Scoring Formula, Rank Blending, Sub-Bagging

**Audience:** implementing agent (Opus). This document is self-contained: current
state, exact changes, exact test commands, and acceptance criteria per phase.
Execute phases IN ORDER — each later phase assumes the earlier ones landed.

---

## 0. Context & Current State (read first, verify before coding)

### The problem
- Bittensor subnet miner scores poker "chunks" (list of hands) as bot(1)/human(0).
- **Training data**: labeled daily files, chunks of ~30–40 hands, one entity per chunk.
- **Live/eval data**: `data/chunks1.json` format — unlabeled, ~80–100 hands/chunk.
- Hands within a chunk are an **unordered bag** (proven: no button rotation, no
  stack carry-over, lag-1 autocorrelation ≈ random). One entity per chunk.
- The bot signal is **within-entity consistency** (repeated action/sizing patterns).
  Any augmentation that mixes hands across entities DESTROYS the signal
  (measured: AP ≈ 0.50 on entity-pooled chunks). Never pool across entities.

### The validator reward (authoritative: `poker44/score/scoring.py`)
```
S = threshold_sanity at 0.5:
    1.0  if no positives OR no negatives in the labeled window
    0.0  if zero true-positives at 0.5          <- HARD GATE: reward = 0
    1.0  if fpr@0.5 <= 0.10
    1 - (fpr@0.5 - 0.10)/0.90  otherwise

reward = 0                                       if S <= 0
       = clip( 0.35*AP
             + 0.30*best_recall_with_fpr<=0.05   (threshold SWEPT — rank-based)
             + 0.20*S + 0.10*S
             + 0.05*1.0 , 0, 1)
```
**65% of reward is pure ranking** (AP + swept recall). **30% is a 0.5-boundary
sanity term with a hard zero.** Strategy: maximize ranking; make the gate unfailable.

### Repo layout facts
- Python: `..\.venv\Scripts\python.exe` relative to `detection_model/` (Windows;
  bash tool available — use forward slashes).
- Run all module commands from `detection_model/`: `../.venv/Scripts/python.exe -m model_v2.<mod>`
- Data dir: `detection_model/data/`. Daily labeled files
  `benchmark_chunks_YYYY-MM-DD.json` = list of `{hands, is_bot, source_date}`.
  Unlabeled eval `chunks1.json` = list of bare hand-lists. Reference these files
  directly for structure questions.
- Download tool: `python -m tools.get_dump --start-date D --end-date D --out-dir data`
  (public API; a date with 0 chunks = not yet released; delete the empty file).
- Canonicalizer (**mandatory before training**; matches the live payload transform):
  `python -m tools.canonicalize_benchmark --input <raw.json> --output <canonical.json>`
  Success check: prints `first-action-is-seat1: ~24% -> 100.0%`.
- `model_v2/` modules: `schema.py` (parses both formats, sorts actions by
  `action_id`), `features.py` (330 order-invariant chunk features),
  `dataset.py`, `metrics.py`, `calibrate.py` (isotonic + boundary remap),
  `train.py` (LGBM-only), `sequence_model.py` (TCN + attention pool,
  sklearn-style), `train_stack.py` (OOF A/B + artifact saving),
  `inference.py` (`Poker44V2Detector` — the ONLY serving path; supports
  lgbm-only / tcn-only / blend artifacts), `evaluate.py`, `predict.py`, `drift.py`.
- Miner: `neurons/miner.py` loads `Poker44V2Detector` from `P44_MODEL_PATH`;
  already has rank-preserving `top_k` cap (`P44_TOP_K`, `P44_TOP_K_FRAC`).
- **KNOWN STALE**: `model_v2/metrics.py` implements the OLD formula
  (0.65*AP + 0.35*recall@0.5, (1-fpr)^2 cliff). Phase 1 replaces it.
- `inference.py` contains `_alias_model_v2_packages()` — REQUIRED for unpickling
  blend artifacts (pickled as `model_v2.*`, imported as `detection_model.model_v2.*`).
  Never remove it; call it before `joblib.load`.

### Prior measured results (context for expectations)
- Random-split OOF on canonical small chunks: LGBM AP ~0.94, TCN AP ~0.93,
  blend AP ~0.96. Blend > LGBM consistently (+0.01–0.04) and TCN errors are
  decorrelated (stacker weights both strongly positive).
- Prob-blend FAILS on shifted data: a collapsed LGBM (all scores < 0.5) gates the
  blend below 0.5 → sanity 0 risk. Rank-blend fixes this (Phase 2).
- `top_k_frac` sweep on large-chunk proxies: 0.1 optimal for every model;
  0.2 always worse (forces fpr@0.5 ≈ 0.2 → sanity decay). 0.1 is the default cap.
- Enemy miner ("poker44-rank-detector-b", HG2Blend): weighted-RANK trio
  (stacked-trees 0.35 / monotone-LGBM 0.30 / PCA56→MLP80 0.35), 294 tree + 611
  wide features, walk-forward validation over 46 dates, deploy-threshold remap to
  0.5, batch positive cap 16%. Their cv_reward 0.928 (under a slightly different
  interim formula — not directly comparable, but the design blueprint is proven).

### Performance / operational warnings
- TCN trains on CPU torch. Each OOF fold + final refit = one full TCN training.
  Budget: folds=3, epochs=12–16. A folds=5/epochs=16/heavy-augment run took >20 min.
- Feature extraction is pure Python — augmentation multiplies its cost. Keep
  multiscale counts modest (≤4/chunk) or cache features.
- Long runs: use `-u` (unbuffered), redirect to a log file, run in background,
  poll the log. Piped stdout stays empty until flush — don't assume a hang from
  an empty file; check the python process before killing.
- LightGBM "X does not have valid feature names" warning is cosmetic — filtered.

---

## Phase 1 — Metric alignment (new formula)

**Goal:** `model_v2/metrics.py` must reproduce `poker44/score/scoring.py` exactly.

**Replace `model_v2/metrics.py` with:**

```python
"""Validator-aligned metrics — mirrors poker44/score/scoring.py (2026-07 formula)."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

AP_WEIGHT = 0.35
BOT_RECALL_WEIGHT = 0.30
HUMAN_SAFETY_WEIGHT = 0.20
CALIBRATION_WEIGHT = 0.10
LATENCY_WEIGHT = 0.05
SANITY_FPR = 0.10


def recall_at_fpr(scores, labels, max_fpr: float = 0.05) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=int); scores = np.asarray(scores, dtype=float)
    pos = int((labels == 1).sum()); neg = int((labels == 0).sum())
    if pos <= 0 or neg <= 0 or scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    tp = np.cumsum(sl == 1); fp = np.cumsum(sl == 0)
    recall = tp / max(pos, 1); fpr = fp / max(neg, 1)
    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0
    ai = np.flatnonzero(allowed)
    best = int(ai[np.argmax(recall[allowed])])
    return float(recall[best]), float(fpr[best])


def _threshold_sanity(scores, labels, threshold: float = 0.5):
    labels = np.asarray(labels, dtype=int); scores = np.asarray(scores, dtype=float)
    pos = int((labels == 1).sum()); neg = int((labels == 0).sum())
    if scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    hp = scores >= float(threshold)
    ppr = float(hp.mean())
    tp = int((hp & (labels == 1)).sum()); fp = int((hp & (labels == 0)).sum())
    hard_recall = tp / max(pos, 1) if pos > 0 else 0.0
    hard_fpr = fp / max(neg, 1) if neg > 0 else 0.0
    if pos <= 0 or neg <= 0:
        sanity = 1.0
    elif tp <= 0:
        sanity = 0.0
    elif hard_fpr <= SANITY_FPR:
        sanity = 1.0
    else:
        sanity = max(0.0, 1.0 - (hard_fpr - SANITY_FPR) / (1.0 - SANITY_FPR))
    return hard_recall, hard_fpr, ppr, sanity


def reward_metrics(labels, scores) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int); s = np.asarray(scores, dtype=float)
    both = len(set(y.tolist())) > 1
    ap = float(average_precision_score(y, s)) if (s.size and (y == 1).any()) else 0.0
    bot_recall, best_fpr = recall_at_fpr(s, y, max_fpr=0.05)
    hard_recall, hard_fpr, ppr, sanity = _threshold_sanity(s, y, 0.5)
    if sanity <= 0.0:
        base = 0.0; reward = 0.0
    else:
        base = (AP_WEIGHT * ap + BOT_RECALL_WEIGHT * bot_recall
                + HUMAN_SAFETY_WEIGHT * sanity + CALIBRATION_WEIGHT * sanity
                + LATENCY_WEIGHT * 1.0)
        reward = float(np.clip(base, 0.0, 1.0))
    return {
        "reward": reward, "ap": ap,
        "bot_recall_at_fpr005": bot_recall, "fpr_at_best": best_fpr,
        "threshold_sanity": sanity, "recall_at_0.5": hard_recall,
        "fpr_at_0.5": hard_fpr, "positive_rate": ppr,
        "roc_auc": float(roc_auc_score(np.clip(y, 0, 1), s)) if both else 0.0,
        "base_score": float(base),
    }


def format_metrics(m: Dict[str, float]) -> str:
    return (f"reward={m['reward']:.4f} ap={m['ap']:.4f} "
            f"recall@fpr05={m['bot_recall_at_fpr005']:.4f} sanity={m['threshold_sanity']:.3f} "
            f"recall@0.5={m['recall_at_0.5']:.4f} fpr@0.5={m['fpr_at_0.5']:.4f}")
```

**Fix callers:** other modules reference old keys. Grep and update:
`grep -rn "recall_at_0.5\|fpr_at_0.5\|FPR_CLIFF\|log_loss" model_v2/*.py` —
`calibrate.py` selects candidates by `m["reward"]` and constrains on
`m["fpr_at_0.5"]` (keys kept, OK); `evaluate.py`/`train.py`/`train_stack.py` may
print `log_loss`/`roc_auc` (drop or guard missing keys).

**Acceptance test (must pass exactly):**
```bash
cd d:/Work/AI/Poker44-subnet && .venv/Scripts/python.exe -c "
import numpy as np, sys; sys.path.insert(0,'detection_model')
from model_v2.metrics import reward_metrics
from poker44.score.scoring import reward as official
rng=np.random.default_rng(0)
for t in range(4):
    y=(rng.random(200)<0.5).astype(int)
    s=np.clip(0.5+0.3*(y-0.5)+rng.normal(0,0.3,200),0,1)
    off,_=official(s,y); mine=reward_metrics(y,s)['reward']
    assert abs(off-mine)<1e-9, (t,off,mine)
s=np.clip(rng.uniform(0.1,0.45,200),0,1); y=(rng.random(200)<0.5).astype(int)
off,_=official(s,y); mine=reward_metrics(y,s)['reward']
assert off==0.0 and mine==0.0
print('PARITY OK (incl. collapse->0 gate)')"
```

---

## Phase 2 — Rank-based blending

**Goal:** blend members by batch-relative rank so a collapsed member cannot gate
the blend. 65% of reward is rank-based; this protects it.

**2a. `model_v2/inference.py`** — add module-level:
```python
def rank01(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)
```
Add `blend_mode: str = "prob"` to `Poker44V2Detector.__init__` (store as
`self.blend_mode`); in `load()` pass `blend_mode=art.get("blend_mode", "prob")`.
In `predict_chunks`, replace the blend arithmetic:
```python
if self.blend_weights is not None:
    w0, w1 = self.blend_weights
    if self.blend_mode == "rank" and self.model is not None:
        raw = (w0 * rank01(raw) + w1 * rank01(seq)) / (w0 + w1)
    else:
        raw = w0 * raw + w1 * seq
else:
    raw = seq
```

**2b. `model_v2/train_stack.py`** — add `--blend-mode {prob,rank}` (default
`rank`), a helper
`_combine(a,b,w,mode) = w*rank01(a)+(1-w)*rank01(b) if mode=="rank" else w*a+(1-w)*b`,
use it for the OOF blend report and for the saved artifact's calibrator input,
and persist `"blend_mode": args.blend_mode` in the saved blend dict.

**Acceptance:** unit — rank blend of `lgbm=[0.30,0.31,0.32]` (collapsed) and
`tcn=[0.1,0.9,0.5]` must order by TCN’s ranking, not sit at ~0.3. Artifact
round-trip: save with `blend_mode="rank"`, `load()` → `detector.blend_mode=="rank"`.

---

## Phase 3 — Sanity-gate insurance (miner)

**Goal:** hard-zero gate can never fire in production.
- Default `P44_TOP_K_FRAC=0.1` in `.env.example` (documented as the measured
  optimum; 0.2 measured worse; keep the miner code default at 0 so env controls it).
- Confirm miner top-k is rank-preserving (existing `_apply_top_k` is; forces the
  top K by rank above 0.5, min-max preserved within bands, exactly K positives).

**Acceptance:** with any score vector, `top_k_frac=0.1` yields exactly
`round(0.1*n)` scores ≥ 0.5, ranking unchanged (`np.argsort` identical), and
AP identical for any labels.

---

## Phase 4 — Sub-bagging inference (MIL mini-bags)

**Goal:** serve 85-hand chunks as averaged 35-hand single-entity sub-samples so
every model input is in-distribution. No retraining required.

**`model_v2/inference.py`** — constructor params + env overrides in `load()`:
```python
subbag_hands: int = 35, subbag_stride: int = 20, subbag_min_hands: int = 50
# in load():
import os
subbag_hands=int(os.getenv("P44_SUBBAG_HANDS", "35")),
subbag_stride=int(os.getenv("P44_SUBBAG_STRIDE", "20")),
subbag_min_hands=int(os.getenv("P44_SUBBAG_MIN_HANDS", "50")),
```
Methods:
```python
def _subsamples(self, hands):
    n = len(hands); size = self.subbag_hands
    if size <= 0 or n <= max(size, self.subbag_min_hands):
        return [hands]
    subs = [hands[s:s+size] for s in range(0, n - size + 1, max(1, self.subbag_stride))]
    return subs or [hands]

def _member_scores(self, chunks):
    all_subs, ranges, cur = [], [], 0
    for chunk in chunks:
        hands = [_sort_actions(h) for h in (chunk or []) if isinstance(h, dict)]
        subs = self._subsamples(hands)
        ranges.append((cur, cur + len(subs))); all_subs += subs; cur += len(subs)
    lg = self._raw_scores(self._feature_rows(all_subs)) if self.model is not None else None
    sq = (np.asarray(self.seq_model.predict_proba(all_subs))[:, 1]
          if self.seq_model is not None else None)
    agg = lambda v: None if v is None else np.array(
        [float(np.mean(v[a:b])) if b > a else 0.5 for a, b in ranges])
    return agg(lg), agg(sq)
```
`predict_chunks` calls `_member_scores` once, then blends the per-chunk member
means (rank or prob), then calibrates. **Averaging happens per member BEFORE the
blend** so rank01 ranks original chunks.

**Acceptance:**
1. Chunks ≤ 50 hands: scores byte-identical with `P44_SUBBAG_HANDS=35` vs `0`.
2. An 86-hand chunk with (35, 20) → exactly 3 sub-samples.
3. `chunks1.json` scored end-to-end without error; std of scores > 0.02.

---

## Phase 5 — Multi-scale single-entity augmentation (training)

**Goal:** size-invariance learned safely. NEVER cross-entity pooling.

**`model_v2/dataset.py`** — add:
```python
def multiscale_augment(chunks, y, per_chunk, size_range=(15, 30), seed=44):
    if per_chunk <= 0:
        return [], np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed); y = np.asarray(y, dtype=int)
    lo, hi = int(size_range[0]), int(size_range[1])
    out_c, out_y = [], []
    for i, chunk in enumerate(chunks):
        n = len(chunk)
        if n < lo + 2:
            continue
        top = min(hi, n - 1)
        for _ in range(per_chunk):
            size = int(rng.integers(lo, top + 1))
            sel = rng.choice(n, size=size, replace=False)
            out_c.append([chunk[j] for j in sel]); out_y.append(int(y[i]))
    return out_c, np.asarray(out_y, dtype=int)

def features_for_chunks(chunks, names):
    rows = [[float(chunk_feature_vector(c).get(n, 0.0)) for n in names] for c in chunks]
    x = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(names)))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
```
(import `chunk_feature_vector` from `.features`.)

**`model_v2/train_stack.py`** — args `--augment-multiscale N` (default 0; use 3),
`--ms-min 15 --ms-max 30`. **Fold-safe**: inside the fold loop, augment ONLY the
fold's training chunks (`seed=args.seed+k`); OOF predictions stay on original
held-out chunks. Same augmentation in the final full-data refit (`seed=args.seed`).

**Acceptance:**
1. `multiscale_augment(chunks[:10], y[:10], 3)` → 30 chunks, sizes within 15–30,
   labels match parents.
2. AP-preservation gate: LGBM-only OOF (3 folds, one daily canonical file
   e.g. `benchmark_chunks_canonical_2026-07-09.json` — create it first via the
   canonicalizer if missing) with `ms=0` vs `ms=3`: AP within ±0.02. If aug drops
   AP by more than that, reduce `per_chunk` or widen `ms-min`.

---

## Phase 6 — Size-invariant features

**Goal:** remove residual hand-count dependence from the feature view.
1. In `features.py::chunk_feature_vector`, add `hand_count_log = log1p(num_hands)`
   (and keep `num_hands`). Append to `_CHUNK_ONLY_KEYS` + `feature_names_for()`.
2. Signature features are size-biased (`top_action_pattern_share` falls with N).
   Add **fixed-reference-size** versions: compute the signature features on up to
   `K=5` random 30-hand subsamples of the chunk (seeded by chunk content hash for
   determinism) and average → `*_at30` columns. Chunks ≤ 30 hands: value = plain
   feature. Deterministic: same chunk → same features (use
   `np.random.default_rng(hash(chunk_fingerprint) % 2**32)`).
3. **Feature-count changes invalidate old artifacts** — all artifacts must be
   retrained in Phase 7; the detector already refuses mismatched feature lists.

**Acceptance:** for a 90-hand chunk, `top_action_pattern_share_at30` computed at
N=90 ≈ the same feature computed directly on its 30-hand subsets (definitionally
true); features deterministic across two calls; `feature_names_for()` length
updated consistently everywhere (train == serve).

---

## Phase 7 — Walk-forward validation harness (the ONE trusted number)

**Goal:** replicate the enemy's honest protocol: train on the past → test the
next unseen date. Random-split OOF is optimistic; this is the decision metric.

**New file `model_v2/walkforward.py`:**
```
usage: python -m model_v2.walkforward --days 3 [--with-tcn] [--epochs 12]
       [--blend 0.6] [--blend-mode rank] [--augment-multiscale 3]
```
Logic:
1. Discover daily canonical files `data/benchmark_chunks_canonical_YYYY-MM-DD.json`
   (create any missing ones from raw dailies via `tools.canonicalize_benchmark`
   first — automate: for every raw daily without a canonical twin, run the tool).
2. For each of the last `--days` dates D (with ≥ 10 prior dates available):
   - TRAIN = concat all canonical days < D (+ multiscale aug), TEST = day D.
   - Fit LGBM (+ TCN if `--with-tcn`), rank-blend, fit calibrator on a 15%
     held-out slice of TRAIN (never on TEST).
   - Report `reward_metrics` on day D for lgbm / tcn / blend.
3. Print per-day table + mean per model. Mean blend reward = THE number.

**Acceptance:** runs end-to-end for `--days 3` without TCN in < 5 min; with TCN
in < 25 min; no TEST-day leakage (assert no test date in train file list).

---

## Phase 8 — Retrain + ship

1. Refresh data: `python -m tools.get_dump --start-date <last+1> --end-date <today>`;
   delete 0-chunk files; canonicalize each new daily.
2. Build merged canonical corpus 2026-05-27 → latest (merge dailies, canonicalize).
3. Train three artifacts:
```bash
cd detection_model
../.venv/Scripts/python.exe -u -m model_v2.train_stack \
  --data data/benchmark_chunks_canonical_2026-05-27_to_<LATEST>.json \
  --folds 3 --epochs 14 --blend 0.6 --blend-mode rank \
  --augment-multiscale 3 --ms-min 15 --ms-max 30 \
  --out-prefix artifacts/p44_v3_<LATEST> > artifacts/train_v3.log 2>&1
```
4. Walk-forward gate: `python -m model_v2.walkforward --days 3 --with-tcn` —
   ship the blend only if `blend >= lgbm` on mean reward; else ship lgbm.
5. Sanity checks on the chosen artifact:
   - `predict` on `chunks1.json`: no error, score std > 0.02, and with
     `P44_TOP_K_FRAC=0.1` exactly 10 of 100 ≥ 0.5.
   - `evaluate` on the newest single canonical day (labeled): sanity = 1.0,
     fpr@0.5 ≤ 0.10.
6. Deployment env:
```
P44_MODEL_PATH=detection_model/artifacts/p44_v3_<LATEST>_blend.joblib
P44_TOP_K_FRAC=0.1
P44_SUBBAG_HANDS=35
P44_SUBBAG_STRIDE=20
P44_SUBBAG_MIN_HANDS=50
```
7. Manifest: `implementation_files` in `neurons/miner.py` must list every
   `model_v2/*.py` actually serving (incl. `sequence_model.py`, `walkforward.py`
   excluded — serving files only). Verify compliance == `transparent`.

---

## 9. Do-NOT list (all measured, do not re-litigate)
- NO cross-entity pooling augmentation (AP ≈ 0.5 on pooled bags — destroys signal).
- NO hand-order / sequence modeling across hands (bag proven; GRU shuffle-diff 0.5).
- NO `top_k_frac >= 0.2` (forces fpr@0.5 ≈ 0.2 → sanity decay).
- NO training on raw (non-canonical) data (live feed is canonicalized; 24%→100%
  seat-1 shift breaks transfer).
- NO tuning on `chunks1.json` (unlabeled; use only as pipeline/spread monitor).
- NO trusting random-split OOF for ship decisions (walk-forward only).

## 10. Stretch (after Phase 8 ships, optional)
- Third rank-vote member: monotone-constrained LGBM on sign-stable features
  (enemy's `mono`, weight ~0.3). Expected small AP gain via decorrelation.
- Wide-view PCA→MLP member (enemy's third member).
- Feature caching keyed by chunk fingerprint to cut augmented-training cost.
- Latency: if the validator activates the 5% latency term, prefer lgbm-only
  (TCN forward pass is the slow path).
