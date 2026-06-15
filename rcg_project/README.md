# RCG-MoE-OPD V1

This directory contains the trainable controller, data builder, local Qwen
worker, policy engine, benchmark driver, and selected experiment artifacts.

## Selected Model

The released V13 checkpoint is:

- Architecture: `rcg_moe_opd_v1_coupled`
- Parameters: 76,190,821
- Hidden input width: 4096
- Latent width: 512
- Maximum candidate branches: 24
- Maximum waves: 12
- Maximum concurrency: 8
- Best adapter step: 200
- Best validation composite: 0.725805

The checkpoint is intentionally not committed to Git. Download it from the
`rcg-moe-opd-v1-v13` GitHub Release into `checkpoints/`.

## Training Data

The generated dataset contains:

- 10,648 trajectory states
- 173,001 action candidates
- Mean 16.25 candidates per state
- Mean 6.25 selected paths per state
- Live replay outcomes for success, timeout, authentication, server, invalid
  call, unavailable API, empty result, provenance, and blocked dependency cases
- Intermediate serial snapshots so later branches learn residual continuation

The full `rcg_moe_opd_v1.jsonl` dataset is a Release asset. The builder is
`build_rcg_moe_opd_v1_data.py`.

## Training Stages

The effective run used:

1. Coupled calibration: planner, outcome, orchestration, then joint phases.
2. Harness adapter: planner-only refinement with base trajectory oversampling.
3. Best-checkpoint selection by held-out composite score.
4. Runtime alignment: the same mean-plus-last Qwen pooling in train and infer.

The selected checkpoint comes from adapter step 200. Later steps reduced
training loss but did not improve the held-out composite score.

## Environment

GPU training:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements-gpu.txt
```

Apple Silicon runtime:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
```

## Rebuild Data

```bash
python build_rcg_moe_opd_v1_data.py \
  --source data/traject_swarm_training_failure_aware_v11.jsonl \
  --base-comparison eval_results/local_qwen_v11_best/comparison.json \
  --output data/rcg_moe_opd_v1.jsonl \
  --summary eval_results/rcg_moe_opd_v1_data_summary.json
```

## Train

```bash
python train_rcg_moe_opd_v1.py \
  --data data/rcg_moe_opd_v1.jsonl \
  --target-path /path/to/Qwen3-8B \
  --encoder-mode qwen \
  --coupled-selection \
  --latent-dim 512 \
  --max-branches 24 \
  --max-waves 12 \
  --max-concurrency 8 \
  --steps 700
```

Use `--resume` and `--phase-mode planner` for the final harness adapter stage.
The exact arguments are stored inside the released checkpoint.

## Artifacts

- `artifacts/eval/base_comparison.json`: same-model Pi baseline.
- `artifacts/eval/runtime_fixed_targeted/comparison.json`: final RCG run.
- `artifacts/eval/*validation.json`: held-out training metrics.
- `artifacts/logs`: selected coupled, adapter, and runtime logs.

Absolute local paths inside raw benchmark JSON are provenance from the original
run; use CLI arguments or environment variables on another machine.
