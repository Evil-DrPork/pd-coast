# Poker44 V4

This package contains the public serving, training, and evaluation source for
the V4 public-benchmark detector. The implementation at the published Git
commit is authoritative; experimental analysis and operator strategy are not
part of the public model card.

## Data and integrity boundary

- Training uses only public Poker44 benchmark data projected to the public
  miner-visible schema.
- Validator-private evaluation data, hidden labels, cards, outcomes, payouts,
  identifiers, and private validator state are not training inputs.
- The deployed artifact must be verified before deserialization. Joblib is a
  pickle format and untrusted artifacts must never be loaded.
- Runtime source, artifact identity, repository URL, and commit are reported by
  the miner model manifest.

## Artifact

```text
detection_model/artifacts/p44_v4_coherent.joblib
SHA-256 6a5dea8ad339e5c8f211475fc68b315dd883363bbc5c181e8d46389c346b13ee
```

Generated artifacts, datasets, caches, and evaluation reports are intentionally
excluded from Git. The exact deployed artifact is identified by its SHA-256.

## Serving

```text
P44_DETECTOR_VARIANT=v4
P44_MODEL_PATH=detection_model/artifacts/p44_v4_coherent.joblib
P44_REQUIRE_MODEL=1
POKER44_MODEL_ARTIFACT_SHA256=6a5dea8ad339e5c8f211475fc68b315dd883363bbc5c181e8d46389c346b13ee
P44_TOP_K=0
P44_TOP_K_FRAC=0
```

Use the dependency versions in `requirements.txt`. A runtime outside the
artifact-qualified environment is rejected by default. The explicit
`P44_ALLOW_RUNTIME_MISMATCH=1` override is for controlled equivalence testing
only and must not replace artifact verification.

For development entry points and their current arguments:

```powershell
.\.venv\Scripts\python.exe -m detection_model.model_v4.train --help
.\.venv\Scripts\python.exe -m detection_model.model_v4.evaluate --help
```

Rollback uses the independently maintained V3 detector and artifact.
