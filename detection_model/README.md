# Poker44 Detection Model

Hierarchical encoder + XGBoost head + **reward-aware calibration**.

> New here? Read [`docs/training-strategy.md`](docs/training-strategy.md) — it
> explains how to train against the validator reward and every hyperparameter.

## Architecture (schema v2)

- Hierarchical backbone:
  - **6 action categorical channels** (street, action_type, seat, amount_bucket,
    pot_flow, first_in_street) + numeric projection
  - action Transformer + attention pooling → hand embedding
  - **per-hand meta fusion** (stack depth, actor count, per-street counts, hero
    engagement, deepest street reached)
  - **pluggable chunk encoder**: `transformer` (default, permutation-invariant
    set encoder) or `gru` (ordered); a soft hand-position channel keeps a light
    ordering hint
  - attention pooling → chunk embedding
- Final XGBoost head on `concat(chunk_embedding, engineered_features)`. The
  engineered features now include cross-hand consistency/signature signals (bot
  tells). Old v1 artifacts are rejected with a clear retrain message.
- **Embedded `ScoreCalibrator`** (`model/calibration.py`): a monotone
  isotonic → threshold-logit remap → logit-shift pipeline fitted against the
  validator reward (`model/scoring.py`) with a hard FPR ceiling below the 0.10
  cliff. It recenters the decision boundary so the miner needs no hand-rolled
  calibration. AP is preserved (calibration is rank-invariant).

## What the reward demands

The validator weights `0.65 * average_precision + 0.35 * bot_recall`, then zeroes
the reward if chunk-level FPR ≥ 0.10. So: the **model** owns ranking/AP, and the
**calibrator** banks it safely under the FPR cliff. See the strategy doc.

## Pipeline notes

- `model/train_hierarchical.py` trains the encoder, fits the XGBoost head, then
  fits and embeds the calibrator — one command, one `.pt` artifact.
- `model/scoring.py` mirrors the on-chain reward so training prints the number
  you are actually paid (not just loss).
- Inference, benchmark, simulation, tools, and the dashboard use the embedded
  XGBoost head and calibrator automatically.

## Install

```bash
cd detection_model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m model.train_hierarchical \
  --data data/public_miner_benchmark.json.gz \
  --out artifacts/p44_first_arch_xgb.pt \
  --epochs 60 \
  --batch-size 8 \
  --augment-windows \
  --augment-validation-windows \
  --window-hands 4 \
  --window-stride 1 \
  --overwrite
```

The output artifact contains:

- the neural hierarchical encoder weights
- the final XGBoost classifier head
- the embedded reward-aware `ScoreCalibrator` (fit on the validation split)

Calibration is on by default. Tune it with `--calibration-objective reward`
(default), `--calibration-target-fpr 0.04`, `--calibration-max-fpr 0.05`, or turn
it off with `--no-calibrate`. The trainer prints a before/after reward line so you
can see the FPR being pulled under the cliff.

## Predict

```bash
python -m model.simulate_result \
  --data data/chunks.json \
  --model artifacts/p44_first_arch_xgb.pt \
  --out-csv outputs/predictions.csv
```

No separate `--xgb-model` is required for new artifacts because XGBoost is embedded in the `.pt` file.

## Benchmark

```bash
python -m model.evaluate_benchmark \
  --data data/public_miner_benchmark.json.gz \
  --model artifacts/p44_first_arch_xgb.pt \
  --split all \
  --out-csv outputs/benchmark_predictions.csv \
  --out-json outputs/benchmark_metrics.json
```

## Dashboard

```bash
streamlit run dashboard/training_dashboard.py
```

The dashboard Train tab now maps to the single integrated training command and exposes the final XGBoost parameters there.

## Compliance note

Train only on public benchmark data and your own legal synthetic simulations. Do not train on validator-only evaluation payloads, live `/internal/eval/current` batches, leaked data, or payload hashes.
