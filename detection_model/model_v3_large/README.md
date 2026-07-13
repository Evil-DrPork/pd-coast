# Poker44 v3 large-chunk tabular challenger

This is an additive challenger. It does not modify or replace
`detection_model/model_v3`, and it writes a separate artifact.

The model trains one tabular ensemble on original, 50–75-hand, and 80–100-hand
full chunks. A second ensemble learns original chunks plus deterministic
micro-bags. At inference it blends full-chunk branch scores with the mean and
median branch scores across content-hash-partitioned micro-bags. No received
hand order is used.

## Train a future-holdout artifact

```bash
python -m detection_model.model_v3_large.train \
  --data detection_model/data_v3/benchmark_chunks_2026-07-06_to_2026-07-12.json \
  --max-date 2026-07-09 \
  --batch-top-fraction 0.10 \
  --out detection_model/artifacts/p44_v3_large_through_2026-07-09.joblib \
  --report detection_model/artifacts/p44_v3_large_through_2026-07-09_train_report.json
```

## Evaluate on future 80–100-hand chunks

```bash
python -m detection_model.model_v3.evaluate_large \
  --data detection_model/data_v3/benchmark_chunks_2026-07-06_to_2026-07-12.json \
  --model detection_model/artifacts/p44_v3_large_through_2026-07-09.joblib \
  --min-date 2026-07-10 \
  --batch-size 100 \
  --sources-per-chunk 3 \
  --min-hands 80 \
  --max-hands 100 \
  --repetitions 100 \
  --out detection_model/artifacts/p44_v3_large_challenger_future_eval.json
```

## Serve only on the challenger miner

```bash
P44_MODEL_PATH=detection_model/artifacts/p44_v3_large.joblib \
P44_DETECTOR_VARIANT=large \
P44_TOP_K=0 \
P44_TOP_K_FRAC=0 \
P44_LATENCY_WARN_SECONDS=25 \
python neurons/miner.py
```

Keep the stable v3 miners pointed at their existing `p44_v3*.joblib` artifacts.

## Current upstream latency behavior

Poker44 upstream commit `367a8a264aabca4af4c2b721f077d98dbde55193`
defaults miner queries to 180 seconds and clamps configured timeouts to at least
30 seconds. The current scorer records latency for diagnostics but sets
`latency_quality = 1.0`, so latency does not change reward as long as a complete
response arrives before timeout. The cached hierarchical path measured about
18.2 seconds for one 100-chunk × 80–100-hand request in the reference test
environment. The miner logs elapsed inference time for every request and warns
when it reaches `P44_LATENCY_WARN_SECONDS` (25 seconds by default).

The provider currently defaults both `min_hands_per_chunk` and
`max_hands_per_chunk` to 100. Keep the 80–100 training range for compatibility
with validators overriding those defaults. An exploratory five-window exact-100
stress test scored 0.8378 mean reward when each synthetic chunk mixed four
same-label public sources, versus 0.8909 across twenty 80–100-hand windows made
from three sources. Because player/source identity is unavailable, treat both as
domain-shift sentries rather than estimates of single-source live performance.

If hierarchical micro-bag inference is too close to the validator timeout,
export the latency-safe full-view ablation. It retains large/medium training
augmentation but skips micro-bag inference:

```bash
python -m detection_model.model_v3_large.export_fast \
  --model detection_model/artifacts/p44_v3_large.joblib \
  --out detection_model/artifacts/p44_v3_large_fast.joblib
```
