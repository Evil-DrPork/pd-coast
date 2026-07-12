# Poker44 model v3

Fresh, validator-native baseline for unordered poker-hand chunks.

## Design

- exact validator canonicalization for raw benchmark exports;
- action order retained only inside each hand;
- hand-order-invariant distribution features;
- logistic, ExtraTrees, histogram-gradient and human-tail prototype ensemble;
- hierarchical action Transformer → hand embedding → permutation-invariant hand-set neural branch;
- OOF promotion gate: the neural challenger is packaged only when its selected weight is at least 2.5%;
- date-blocked out-of-fold blending;
- direct import of the production validator reward;
- fixed and batch-relative monotone score mapping;
- labeled reward evaluation and unlabeled drift/prediction diagnostics.

## Train

From the repository root:

```bash
python -m detection_model.tools.get_dump \
  --start-date 2026-07-06 \
  --end-date 2026-07-12 \
  --out-dir detection_model/data_v3
```

Then train:

```bash
python -m detection_model.model_v3.train \
  --data detection_model/data_v3/benchmark_chunks_2026-07-06_to_2026-07-12.json \
  --out detection_model/artifacts/p44_v3.joblib \
  --report detection_model/artifacts/p44_v3_train_report.json \
  --reward-window 100 \
  --neural-epochs 12 \
  --merged-ratio 0.60
```

For a genuinely future holdout, train only through the prior day:

```bash
python -m detection_model.model_v3.train \
  --data detection_model/data_v3/benchmark_chunks_2026-07-06_to_2026-07-12.json \
  --max-date 2026-07-11 \
  --out detection_model/artifacts/p44_v3_through_2026-07-11.joblib
```

## Evaluate labeled performance

Use a date that was not included in training for the strongest test:

```bash
python -m detection_model.model_v3.evaluate \
  --data detection_model/data_v3/benchmark_chunks_YYYY-MM-DD.json \
  --model detection_model/artifacts/p44_v3.joblib \
  --out detection_model/artifacts/p44_v3_eval_YYYY-MM-DD.json \
  --reward-window 100
```

## Inspect an unlabeled evaluation set

```bash
python -m detection_model.model_v3.evaluate \
  --data /path/to/evaluation_dataset.json \
  --model detection_model/artifacts/p44_v3.joblib \
  --out detection_model/artifacts/p44_v3_hidden_diagnostics.json
```

This reports prediction distribution and permutation invariance, but correctly
does not claim accuracy or validator reward without labels.

## Evaluate 100 batches of 100 merged 80–100-hand chunks

Train through July 9, then use July 10–12 only as future source chunks:

```bash
python -m detection_model.model_v3.evaluate_large \
  --data detection_model/data_v3/benchmark_chunks_2026-07-06_to_2026-07-12.json \
  --model detection_model/artifacts/p44_v3_through_2026-07-09.joblib \
  --min-date 2026-07-10 \
  --batch-size 100 \
  --sources-per-chunk 3 \
  --min-hands 80 \
  --max-hands 100 \
  --repetitions 100 \
  --out detection_model/artifacts/p44_v3_large_future_eval.json
```

The evaluator refuses training/evaluation date overlap and does not reuse a
source chunk inside one synthetic 100-chunk batch.
