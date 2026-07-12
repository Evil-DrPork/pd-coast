"""A/B the TCN sequence model against the LightGBM — does stacking help?

    python -m model_v2.train_stack --data data/<canonical-labeled>.json

Builds out-of-fold (OOF) predictions for both base learners with StratifiedKFold,
then reports the validator reward for: LightGBM alone, TCN alone, a fixed 0.6/0.4
blend, and a logistic stack. The one number that matters is whether blend/stack
beats LGBM-alone — if not, the sequence model isn't earning its complexity.

Note: the cliff-aware calibrator is fit and evaluated on the same OOF scores, so
absolute rewards are mildly optimistic — but the procedure is identical for every
variant, so the *comparison* between them is fair. AP is calibration-free.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .calibrate import apply_calibrator, fit_calibrator
from .dataset import build_feature_matrix, features_for_chunks, multiscale_augment
from .inference import rank01
from .metrics import format_metrics, reward_metrics
from .schema import load_chunks
from .sequence_model import TCNSequenceModel
from .train import _proba, build_model


def _combine(a: np.ndarray, b: np.ndarray, w: float, mode: str) -> np.ndarray:
    """Blend two member score vectors by probability or batch-relative rank."""
    if mode == "rank":
        return w * rank01(a) + (1.0 - w) * rank01(b)
    return w * a + (1.0 - w) * b


def _report(tag: str, y: np.ndarray, scores: np.ndarray) -> dict:
    cal = fit_calibrator(scores, y)
    m = reward_metrics(y, apply_calibrator(cal, scores))
    print(f"  {tag:16s}: {format_metrics(m)}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Canonical labeled JSON.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--out", default="", help="If set, refit on all data and save a deployable blend artifact.")
    ap.add_argument("--out-prefix", default="", help="If set, save all three: {prefix}_lgbm/_tcn/_blend.joblib.")
    ap.add_argument("--blend", type=float, default=0.6, help="LGBM weight in the blend (TCN gets 1-blend).")
    ap.add_argument("--blend-mode", choices=("prob", "rank"), default="rank",
                    help="Combine members by probability or batch-relative rank (rank = collapse-robust).")
    ap.add_argument("--augment-multiscale", type=int, default=0,
                    help="Single-entity size-variety sub-samples per chunk per fold (safe size-invariance).")
    ap.add_argument("--ms-min", type=int, default=15)
    ap.add_argument("--ms-max", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=2, help="TCN epoch progress cadence (0 = quiet).")
    args = ap.parse_args()

    x, y, names, _ = build_feature_matrix(args.data)
    if y is None:
        raise SystemExit("Data has no labels.")
    chunks = [c.hands for c in load_chunks(args.data)]
    n = len(y)
    from .sequence_model import _TCNChunkNet, SeqConfig
    tcn_params = sum(p.numel() for p in _TCNChunkNet(SeqConfig()).parameters())
    print(f"Loaded {n} chunks | bot={int(y.sum())} human={int((y==0).sum())} | features={x.shape[1]}", flush=True)
    print(f"Models: LGBM(400 trees x<=31 leaves) + TCN({tcn_params:,} params) | blend_mode={args.blend_mode} "
          f"| folds={args.folds} epochs={args.epochs}", flush=True)
    if args.augment_multiscale > 0:
        print(f"Multi-scale augmentation: +{args.augment_multiscale}/chunk/fold at "
              f"{args.ms_min}-{args.ms_max} hands (single-entity)", flush=True)

    def _augment(ch_tr, y_tr, seed):
        return multiscale_augment(ch_tr, y_tr, args.augment_multiscale, (args.ms_min, args.ms_max), seed)

    t_start = time.time()
    oof_lgbm = np.zeros(n)
    oof_seq = np.zeros(n)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(x, y), start=1):
        t_fold = time.time()
        print(f"\n--- fold {k}/{args.folds} (train={len(tr)} val={len(va)}) ---", flush=True)
        # Fold-safe augmentation: augment ONLY this fold's training rows; OOF
        # predictions stay on the original held-out chunks (no leakage).
        ch_tr = [chunks[i] for i in tr]
        ac, ay = _augment(ch_tr, y[tr], args.seed + k)
        x_tr = np.vstack([x[tr], features_for_chunks(ac, names)]) if ac else x[tr]
        y_tr = np.concatenate([y[tr], ay]) if ac else y[tr]
        if ac:
            print(f"  augmented: +{len(ac)} chunks -> train rows={len(y_tr)}", flush=True)

        lg = build_model(args.seed, int(y_tr.sum()), int((y_tr == 0).sum()))
        lg.fit(x_tr, y_tr)
        oof_lgbm[va] = _proba(lg, x[va])
        print(f"  lgbm fit done ({time.time()-t_fold:.1f}s) ap={_ap(y[va], oof_lgbm[va]):.3f}", flush=True)

        sq = TCNSequenceModel(seed=args.seed, epochs=args.epochs, verbose=False,
                              log_every=args.log_every, tag=f"fold {k}/{args.folds}")
        sq.fit(ch_tr + ac, y_tr)
        oof_seq[va] = sq.predict_proba([chunks[i] for i in va])[:, 1]
        print(f"  fold {k}/{args.folds} complete: lgbm_ap={_ap(y[va], oof_lgbm[va]):.3f} "
              f"tcn_ap={_ap(y[va], oof_seq[va]):.3f} | fold {time.time()-t_fold:.1f}s "
              f"| total {time.time()-t_start:.1f}s", flush=True)

    blend = _combine(oof_lgbm, oof_seq, args.blend, args.blend_mode)
    stacker = LogisticRegression(max_iter=1000)
    feats = np.column_stack([oof_lgbm, oof_seq])
    stacker.fit(feats, y)
    oof_stack = stacker.predict_proba(feats)[:, 1]

    print(f"\nOut-of-fold reward comparison (blend_mode={args.blend_mode}):")
    m_lgbm = _report("LGBM alone", y, oof_lgbm)
    _report("TCN alone", y, oof_seq)
    m_blend = _report(f"blend {args.blend}/{1-args.blend:.1f}", y, blend)
    m_stack = _report("logistic stack", y, oof_stack)

    print(f"\nstacker weights: lgbm={stacker.coef_[0][0]:+.3f} tcn={stacker.coef_[0][1]:+.3f}")
    best = max([("blend", m_blend["reward"]), ("stack", m_stack["reward"])], key=lambda t: t[1])
    delta = best[1] - m_lgbm["reward"]
    verdict = (
        f"{best[0]} beats LGBM-alone by {delta:+.4f} reward -> stacking helps"
        if delta > 1e-3 else
        f"no combo beats LGBM-alone (best {best[0]} {delta:+.4f}) -> ship the tabular model"
    )
    print(f"\nVERDICT: {verdict}")

    # Full-data refit (shared by --out and --out-prefix): same augmentation on all data.
    def _refit_full():
        t = time.time()
        ac, ay = _augment(chunks, y, args.seed)
        x_all = np.vstack([x, features_for_chunks(ac, names)]) if ac else x
        y_all = np.concatenate([y, ay]) if ac else y
        print(f"  refit train rows={len(y_all)} (+{len(ac)} aug)", flush=True)
        lg = build_model(args.seed, int(y_all.sum()), int((y_all == 0).sum()))
        lg.fit(x_all, y_all)
        sq = TCNSequenceModel(seed=args.seed, epochs=args.epochs, verbose=False,
                              log_every=args.log_every, tag="refit").fit(chunks + ac, y_all)
        print(f"  refit done ({time.time()-t:.1f}s)", flush=True)
        return lg, sq

    w = float(args.blend)

    if args.out:
        from pathlib import Path
        import joblib
        print(f"\nRefitting LGBM + TCN on all {n} chunks and saving blend (mode={args.blend_mode}, w_lgbm={w})...")
        lgbm_full, seq_full = _refit_full()
        blend_oof = _combine(oof_lgbm, oof_seq, w, args.blend_mode)
        cal = fit_calibrator(blend_oof, y)
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "blend_v1", "lgbm_model": lgbm_full, "seq_model": seq_full,
                "feature_names": names, "blend_weights": [w, 1.0 - w],
                "blend_mode": args.blend_mode, "calibrator": cal, "backend": "lgbm+tcn-blend",
                "val_metrics": reward_metrics(y, apply_calibrator(cal, blend_oof)),
            },
            out,
        )
        print(f"Saved blend artifact -> {out}")
        print(f"OOF blend (calibrated): {format_metrics(reward_metrics(y, apply_calibrator(cal, blend_oof)))}")

    if args.out_prefix:
        from pathlib import Path
        import joblib
        print(f"\nRefitting LGBM + TCN on all {n} chunks -> three artifacts (prefix={args.out_prefix})...")
        lgbm_full, seq_full = _refit_full()
        blend_oof = _combine(oof_lgbm, oof_seq, w, args.blend_mode)
        specs = {
            "lgbm": {"model": lgbm_full, "calibrator": fit_calibrator(oof_lgbm, y),
                     "backend": "lightgbm", "oof": oof_lgbm},
            "tcn": {"seq_model": seq_full, "blend_weights": [0.0, 1.0],
                    "calibrator": fit_calibrator(oof_seq, y), "backend": "tcn-only", "oof": oof_seq},
            "blend": {"lgbm_model": lgbm_full, "seq_model": seq_full, "blend_weights": [w, 1.0 - w],
                      "blend_mode": args.blend_mode, "calibrator": fit_calibrator(blend_oof, y),
                      "backend": "lgbm+tcn-blend", "oof": blend_oof},
        }
        for name, spec in specs.items():
            oof = spec.pop("oof")
            art = {"kind": "v2_" + name, "feature_names": names,
                   "val_metrics": reward_metrics(y, apply_calibrator(spec["calibrator"], oof))}
            art.update(spec)
            out = Path(f"{args.out_prefix}_{name}.joblib").expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(art, out)
            print(f"  {name:5s} -> {out}  | OOF {format_metrics(art['val_metrics'])}")


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if len(set(y.tolist())) > 1 else 0.0


if __name__ == "__main__":
    main()
