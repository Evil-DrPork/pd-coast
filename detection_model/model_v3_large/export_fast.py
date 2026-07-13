"""Export the latency-safe full-view ablation from a trained large artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    source = Path(args.model)
    artifact = joblib.load(source)
    if artifact.get("artifact_version") != 3:
        raise SystemExit("expected a v3-compatible artifact")
    model = artifact.get("model")
    weights = np.asarray(getattr(model, "branch_weights_", []), dtype=float)
    if weights.size != 12:
        raise SystemExit("expected a 12-channel large-chunk model")
    full = weights[:4]
    if float(full.sum()) <= 0:
        raise SystemExit("large artifact has no full-view weight")
    model.branch_weights_ = np.concatenate([full / full.sum(), np.zeros(8, dtype=float)])
    model.disable_micro_ = True
    artifact["parent_artifact"] = str(source)
    artifact["architecture"] = "tabular_large_augmented_fast_v1"
    artifact["micro_inference_disabled"] = True
    artifact["channel_names"] = list(getattr(model, "channel_names", []))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)
    print(f"Saved fast large-chunk artifact: {out}")
    print("Weights:", np.round(model.branch_weights_, 4).tolist())


if __name__ == "__main__":
    main()
