# Fascia Runtime Project

This directory contains the local Qwen worker, prompt/API harness policy, and
Fascia-MoE inference package used by the speculative swarm integration.

The full training data, encoder cache, training logs, and best checkpoint are
kept outside the Git repository. See `../ASSET_MANIFEST.md` for the local
paths used in the validated run.

## Runtime Modes

### Trained Fascia

`local_qwen_rcg_worker.py` loads one Qwen3-8B MLX model and one Fascia
controller checkpoint. The controller reuses `encode_texts(...)` from the
same Qwen process, so no second base model is loaded.

```bash
./pi-test.sh \
  --swarm \
  --swarm-policy trained \
  --swarm-local-model-path /path/to/Qwen3-8B \
  --swarm-local-python rcg_project/.venv/bin/python \
  --swarm-local-rcg-checkpoint /path/to/fascia_best_step_1100_0.92843.pt \
  --mode text \
  --no-session \
  -p "Your tool task"
```

### Prompt/API Harness

The prompt mode does not load Fascia weights. It asks Qwen/API to propose an
action DAG and uses deterministic harness logic for validation, routing,
completion checks, and takeover.

```bash
./pi-test.sh \
  --swarm \
  --swarm-policy prompt \
  --swarm-local-model-path /path/to/Qwen3-8B \
  --swarm-local-python rcg_project/.venv/bin/python \
  --mode text \
  --no-session \
  -p "Your tool task"
```

## Fascia-MoE v3

Runtime inference code lives in `fascia_moe/`:

- `model.py`: sparse expert controller.
- `policy_engine.py`: plan/select/route API used by the swarm harness.
- `encoding.py`: canonical task/action/evidence text and feature encoding.
- `schema.py`: recovery labels and expert taxonomy.

Validated checkpoint metadata:

- version: `fascia_moe_v3_protocol_hotpot_takeover`
- step: 1100
- parameters: 135,095,394
- hidden input width: 4096
- latent width: 512
- max branches: 24
- max waves: 12
- max concurrency: 8

## Install

Apple Silicon runtime:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
```

GPU training dependencies are listed for reproducibility, but training is not
part of the GitHub release payload:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements-gpu.txt
```

## Verify

```bash
python3 -m py_compile local_qwen_rcg_worker.py api_harness_policy.py fascia_moe/*.py
python3 test_api_harness_policy.py
```

## Legacy Files

Some older RCG training and benchmark scripts remain in this directory for
provenance, but the current validated runtime path is Fascia-MoE v3 plus the
speculative swarm harness.
