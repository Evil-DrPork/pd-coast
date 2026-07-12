"""Walk-forward validation — the ONE trusted ship metric.

    python -m model_v2.walkforward --days 3 [--with-tcn] [--epochs 12]
        [--blend 0.6] [--blend-mode rank] [--augment-multiscale 3]

For each of the last ``--days`` dates D (with >= 10 prior dates), TRAIN on all
canonical days < D and TEST on day D — the enemy's honest protocol (train past ->
test next unseen date). Random-split OOF is optimistic; this simulates live
"score the future". Calibrators are fit on a held-out slice of TRAIN, never TEST.

Note: daily chunks are ~35 hands, so sub-bagging is not exercised here (it only
triggers > 50 hands). This measures temporal generalization + model quality on the
distribution we CAN label; the large-chunk size gap is handled structurally by
sub-bagging inference + multiscale augmentation (unlabeled, cannot be scored here).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .calibrate import apply_calibrator, fit_calibrator
from .dataset import chunk_split, features_for_chunks, multiscale_augment
from .features import feature_names_for
from .inference import rank01
from .metrics import format_metrics, reward_metrics
from .schema import load_chunks

DATA = Path("data")
_RAW_DATE = re.compile(r"^benchmark_chunks_(2026-\d\d-\d\d)\.json$")
_CANON_DATE = re.compile(r"^benchmark_chunks_canonical_(2026-\d\d-\d\d)\.json$")


def _ensure_canonical(start_date: str) -> dict:
    """Canonicalize raw single-date dailies (>= start_date) lacking a canonical
    twin; return {date: canonical_path} sorted by date. start_date keeps us in the
    canonical-serving era and skips the huge pre-05-26 old-format files."""
    for raw in sorted(DATA.glob("benchmark_chunks_2026-*.json")):
        m = _RAW_DATE.match(raw.name)
        if not m or m.group(1) < start_date:
            continue
        canon = DATA / f"benchmark_chunks_canonical_{m.group(1)}.json"
        if not canon.exists():
            print(f"  canonicalizing {m.group(1)} ...")
            subprocess.run(
                [sys.executable, "-m", "tools.canonicalize_benchmark",
                 "--input", str(raw), "--output", str(canon)],
                check=True, capture_output=True,
            )
    out = {}
    for f in DATA.glob("benchmark_chunks_canonical_2026-*.json"):
        m = _CANON_DATE.match(f.name)
        if m and m.group(1) >= start_date:
            out[m.group(1)] = f
    return dict(sorted(out.items()))


def _load_xy(dates, canon, names):
    chunks, ys = [], []
    for d in dates:
        for c in load_chunks(canon[d]):
            if c.label is None:
                continue
            chunks.append(c.hands)
            ys.append(int(c.label))
    return chunks, np.asarray(ys, dtype=int), features_for_chunks(chunks, names)


def _combine(a, b, w, mode):
    return (w * rank01(a) + (1 - w) * rank01(b)) if mode == "rank" else (w * a + (1 - w) * b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--min-prior", type=int, default=10)
    ap.add_argument("--start-date", default="2026-05-27", help="Ignore dailies before this (canonical era).")
    ap.add_argument("--with-tcn", action="store_true")
    ap.add_argument("--with-mono", action="store_true", help="Add monotone-LGBM member (fast, no torch).")
    ap.add_argument("--mono-min-corr", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--blend", type=float, default=0.6)
    ap.add_argument("--blend-mode", choices=("prob", "rank"), default="rank")
    ap.add_argument("--augment-multiscale", type=int, default=3)
    ap.add_argument("--ms-min", type=int, default=15)
    ap.add_argument("--ms-max", type=int, default=30)
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    from .train import _proba, build_model, build_monotone_model, monotone_constraints
    tcn_cls = None
    if args.with_tcn:
        from .sequence_model import TCNSequenceModel as tcn_cls
    m2_name = "tcn" if args.with_tcn else ("mono" if args.with_mono else None)

    names = feature_names_for()
    canon = _ensure_canonical(args.start_date)
    all_dates = list(canon.keys())
    if len(all_dates) < args.min_prior + 1:
        raise SystemExit(f"Need >= {args.min_prior + 1} canonical daily files, have {len(all_dates)}.")
    test_dates = [d for i, d in enumerate(all_dates) if i >= args.min_prior][-args.days:]
    print(f"{len(all_dates)} canonical days {all_dates[0]}..{all_dates[-1]} | testing {test_dates}"
          f" | member2={m2_name}\n", flush=True)

    rows = {"lgbm": [], "m2": [], "blend": []}
    for D in test_dates:
        import time as _t
        t0 = _t.time()
        train_dates = [d for d in all_dates if d < D]
        assert D not in train_dates, "TEST leakage"
        tr_chunks, ytr, Xtr = _load_xy(train_dates, canon, names)
        te_chunks, yte, Xte = _load_xy([D], canon, names)

        fit_idx, cal_idx = chunk_split(len(ytr), 0.15, args.seed)
        ac, ay = multiscale_augment([tr_chunks[i] for i in fit_idx], ytr[fit_idx],
                                    args.augment_multiscale, (args.ms_min, args.ms_max), args.seed)
        Xf = np.vstack([Xtr[fit_idx], features_for_chunks(ac, names)]) if ac else Xtr[fit_idx]
        yf = np.concatenate([ytr[fit_idx], ay]) if ac else ytr[fit_idx]

        lg = build_model(args.seed, int(yf.sum()), int((yf == 0).sum())); lg.fit(Xf, yf)
        lg_cal, lg_te = _proba(lg, Xtr[cal_idx]), _proba(lg, Xte)

        m2_cal = m2_te = None
        if args.with_tcn:
            sq = tcn_cls(seed=args.seed, epochs=args.epochs, verbose=False)
            sq.fit([tr_chunks[i] for i in fit_idx] + ac, yf)
            m2_cal = sq.predict_proba([tr_chunks[i] for i in cal_idx])[:, 1]
            m2_te = sq.predict_proba(te_chunks)[:, 1]
        elif args.with_mono:
            mono = monotone_constraints(Xf, yf, args.mono_min_corr)
            mm = build_monotone_model(args.seed, int(yf.sum()), int((yf == 0).sum()), mono)
            mm.fit(Xf, yf)
            m2_cal, m2_te = _proba(mm, Xtr[cal_idx]), _proba(mm, Xte)

        def _score(name, cal_scores, te_scores):
            cal = fit_calibrator(cal_scores, ytr[cal_idx])
            m = reward_metrics(yte, apply_calibrator(cal, te_scores))
            rows[name].append(m)
            return m

        m_l = _score("lgbm", lg_cal, lg_te)
        line = f"  {D}  (train={len(ytr)}) lgbm r={m_l['reward']:.4f} ap={m_l['ap']:.4f}"
        if m2_te is not None:
            m_2 = _score("m2", m2_cal, m2_te)
            b_cal = _combine(lg_cal, m2_cal, args.blend, args.blend_mode)
            b_te = _combine(lg_te, m2_te, args.blend, args.blend_mode)
            m_b = _score("blend", b_cal, b_te)
            line += f" | {m2_name} r={m_2['reward']:.4f} | blend r={m_b['reward']:.4f} ap={m_b['ap']:.4f}"
        print(line + f" | {_t.time()-t0:.1f}s", flush=True)

    print("\n=== walk-forward mean (THE ship metric) ===", flush=True)
    for key, label in (("lgbm", "lgbm"), ("m2", m2_name or "m2"), ("blend", "blend")):
        if rows[key]:
            mr = np.mean([m["reward"] for m in rows[key]])
            ma = np.mean([m["ap"] for m in rows[key]])
            ms = np.mean([m["threshold_sanity"] for m in rows[key]])
            print(f"  {label:6s}: reward={mr:.4f}  ap={ma:.4f}  sanity={ms:.3f}  (n={len(rows[key])})", flush=True)


if __name__ == "__main__":
    main()
