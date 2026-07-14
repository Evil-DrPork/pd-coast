# Poker44 V4.1 coherent rank-robust challenger

V4.1 is an additive, full-chunk detector built to preserve the stable V3 path as
a rollback. It trains only on labeled public Poker44 chunks and does not use
synthetic chunk merges, micro-bag pseudo-labels, hidden evaluation labels, hand
count, dates, identifiers, cards, outcomes, or payouts as model features.

## Architecture

- 821 order-invariant features: the 353-feature V3 view plus 468 chunk-coherence
  features covering action, actor, street, amount, pot, stack, seat, and
  within-hand behavior distributions;
- six raw-feature branches: coherent ExtraTrees, coherent random
  forest, coherent histogram gradient boosting, regularized combined-view
  ExtraTrees, robust logistic regression, and a human-tail prototype;
- three request-relative branches trained on within-date feature percentiles:
  coherent histogram gradient boosting, combined-view ExtraTrees, and robust
  logistic regression. These retain cross-chunk behavioral ordering when
  absolute stack, pot, or amount scales move outside the labeled range;
- chronological walk-forward model/mapper selection;
- deterministic behavior-hash tie breaking, so request or hand permutation
  cannot change aligned scores;
- one exact request-relative boundary mapper. The selected configuration puts
  the top 10% above 0.5 while retaining the model's complete ranking.

The selected branch weights are:

```text
coherent ExtraTrees       0.0
coherent random forest    0.0
coherent HistGradient     0.2
combined ExtraTrees       0.0
combined logistic         0.2
human-tail prototype      0.0
rank coherent HistGradient 0.2
rank combined ExtraTrees   0.2
rank combined logistic     0.2
```

## Qualified run

The current local artifact used 1,892 deduplicated real chunks from 50 nonempty
dates between 2026-05-26 and 2026-07-14. Configuration selection used
2026-07-07 through 2026-07-12. The promotion comparison retrained through
2026-07-12, then scored 2026-07-13 and 2026-07-14 without updating on either
date. The deployable artifact was subsequently fit on all available public
dates.

| Balanced window | V4 | Stable V3 0.10 | Stable V3 0.15 | V3 large |
|---:|---:|---:|---:|---:|
| 40 chunks | 0.9832 | 0.9455 | 0.9455 | 0.9365 |
| 100 chunks | 0.9805 | 0.9282 | 0.9282 | 0.9193 |

On 100-chunk windows V4.1 beat stable in 99.67% of 300 paired resamples and
beat the large model in 100%; mean paired deltas were +0.0522 and +0.0611.
Those windows overlap heavily and come from only two independent dates, so the
resampling win rates measure robustness rather than hundreds of independent
holdouts.

The labeled corpus contains 30/40-hand chunks, while the current unlabeled
serving-shaped file contains 80–100 hands per chunk. On that 100-chunk file,
37.21% of feature cells were outside the labeled q01–q99 envelope. The original
raw V4 probabilities collapsed to 0.537–0.582; V4.1's 60% feature-rank blend
restored a 0.217–0.747 raw range (std 0.145) without using unlabeled labels.
Inference took 11.72 seconds in the reference Windows environment, exactly 10
chunks crossed 0.5, and reversing the request produced a maximum aligned score
difference of 0.0.

This is a strong offline promotion gate, not a promise of live weight. The
80–100-hand regime remains unlabeled and shifted, so deploy V4.1 as a controlled
canary/A-B challenger and keep V3 ready until real competition observations
confirm the gain.

Current artifact:

```text
detection_model/artifacts/p44_v4_coherent.joblib
SHA-256 6a5dea8ad339e5c8f211475fc68b315dd883363bbc5c181e8d46389c346b13ee
```

Generated artifacts and public datasets are intentionally git-ignored. Copy or
publish the artifact separately when deploying another checkout. Joblib is a
pickle format: never load an untrusted or competitor artifact. Pin and verify
the expected SHA-256 before deserialization.

## Reproduce training

Download the public date range:

```powershell
.\.venv\Scripts\python.exe -m detection_model.tools.get_dump `
  --start-date 2026-05-26 `
  --end-date 2026-07-15 `
  --out-dir detection_model/data_v3
```

Train and write an isolated V4 artifact:

```powershell
$env:LOKY_MAX_CPU_COUNT='8'
.\.venv\Scripts\python.exe -m detection_model.model_v4.train `
  --data detection_model/data_v3/benchmark_chunks_2026-05-26_to_2026-07-15.json `
  --out detection_model/artifacts/p44_v4_coherent.joblib `
  --report detection_model/artifacts/p44_v4_coherent_train_report.json `
  --feature-cache detection_model/artifacts/p44_v4_features.npz
```

Run the paired chronological promotion comparison:

```powershell
.\.venv\Scripts\python.exe -m detection_model.model_v4.evaluate `
  --data detection_model/data_v3/benchmark_chunks_2026-05-26_to_2026-07-15.json `
  --feature-cache detection_model/artifacts/p44_v4_features.npz `
  --v4-report detection_model/artifacts/p44_v4_coherent_train_report.json `
  --stable-010 detection_model/artifacts/miner_tabular_0.1.joblib `
  --stable-015 detection_model/artifacts/miner_tabular_0.15.joblib `
  --large detection_model/artifacts/p44_v3_large.joblib `
  --out detection_model/artifacts/p44_v4_holdout_comparison.json
```

## Serve the challenger

Use one mapper only:

```text
P44_DETECTOR_VARIANT=v4
P44_MODEL_PATH=detection_model/artifacts/p44_v4_coherent.joblib
P44_REQUIRE_MODEL=1
POKER44_MODEL_ARTIFACT_SHA256=6a5dea8ad339e5c8f211475fc68b315dd883363bbc5c181e8d46389c346b13ee
P44_TOP_K=0
P44_TOP_K_FRAC=0
```

The artifact records its exact Python, NumPy, SciPy, scikit-learn and joblib
versions. A different serving environment is unsupported by scikit-learn and
is rejected by default. For a controlled equivalence test only, set
`P44_ALLOW_RUNTIME_MISMATCH=1`; detector-runtime and library-version mismatches
then emit explicit warnings while artifact SHA, architecture, feature schema,
feature implementation and branch-schema validation remain enforced. Remove
the override or retrain in the serving environment if predictions differ.

Rollback requires only changing the variant/path back to stable V3; no V3
implementation or artifact was overwritten.
