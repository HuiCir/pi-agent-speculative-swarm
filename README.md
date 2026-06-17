# Fascia Speculative Swarm

This repository contains the Pi agent speculative swarm harness integrated with
two dynamic policy modes:

- `--swarm-policy trained`: load a local Fascia-MoE controller checkpoint.
- `--swarm-policy prompt`: use a prompt/API planner plus deterministic harness
  validation, without loading Fascia weights.

The swarm runs on one local Qwen3-8B process. Main-agent generation,
subagents, prompt/KV cache, and Fascia embeddings share that process; the
controller does not launch a second base model.

## What Is Included

- Full Pi monorepo snapshot with speculative swarm orchestration.
- Fascia-aware dynamic action-DAG scheduler with direct takeover.
- Runtime support for trained Fascia and prompt-only API harness policy modes.
- Fascia-MoE inference code under `rcg_project/fascia_moe/`.
- Local MLX Qwen worker with shared cache and batched branch execution.
- Technical report, validation summary, and small benchmark evidence.

Large training data and best checkpoints are intentionally not committed. See
[ASSET_MANIFEST.md](ASSET_MANIFEST.md) for local paths and SHA-256 hashes.

## Validated Snapshot

Selected local checkpoint:

| Field | Value |
| --- | --- |
| Method name | Fascia |
| Checkpoint version | `fascia_moe_v3_protocol_hotpot_takeover` |
| Selected step | 1100 |
| Parameters | 135,095,394 |
| Best validation score | 0.928431 |
| SHA-256 | `8ebfc0b38d99dde7ded323383051f772737c023445c6001c7fe4a1952d90c392` |

Latest full validation summary:

| Suite | Raw Fascia | Trained Fascia | Main comparison |
| --- | ---: | ---: | --- |
| Offline controller, 987 rows | 0.3647 | 0.9254 | +0.5607 composite |
| Merged 4-way controller, 320 rows | 0.1894 | 0.8091 | +0.4276 vs Qwen proxy |
| Long dependency probe, 6 tasks | 100% success, 4.33 turns | 100% success, 1.00 turn | old v3 failure fixed |
| Forced multiturn probe, 4 tasks | 50% success, 5.00 turns | 100% success, 1.00 turn | +75 pp vs Qwen base |

The long-dependency failure from the previous run was not reproduced after the
harness fix. The old failure omitted two required `geocode` calls and
`route_time`; the fixed scheduler expands repeated tool instances into
argument-level branches, producing exactly two geocode calls plus weather and
route-time execution.

## Quick Start

Requirements:

- Node.js 22.19 or newer
- Qwen3-8B in Hugging Face format
- macOS Apple Silicon for the included MLX runtime
- Python with PyTorch, `mlx`, and `mlx-lm`

```bash
npm ci --ignore-scripts
npm run build

python3 -m venv rcg_project/.venv
rcg_project/.venv/bin/pip install -r rcg_project/requirements-local.txt
```

Run local Qwen + trained Fascia:

```bash
./pi-test.sh \
  --swarm \
  --swarm-policy trained \
  --swarm-local-model-path /path/to/Qwen3-8B \
  --swarm-local-python rcg_project/.venv/bin/python \
  --swarm-local-rcg-checkpoint /path/to/fascia_best_step_1100_0.92843.pt \
  --mode text \
  --no-session \
  -p "Your tool-using task"
```

Run local Qwen + prompt/API harness policy:

```bash
./pi-test.sh \
  --swarm \
  --swarm-policy prompt \
  --swarm-local-model-path /path/to/Qwen3-8B \
  --swarm-local-python rcg_project/.venv/bin/python \
  --mode text \
  --no-session \
  -p "Your tool-using task"
```

The non-swarm local Qwen baseline uses the same model and runtime:

```bash
./pi-test.sh \
  --local-qwen \
  --local-model-path /path/to/Qwen3-8B \
  --local-python rcg_project/.venv/bin/python \
  --mode text \
  --no-session \
  -p "Your tool-using task"
```

## Verification

```bash
npm --prefix packages/coding-agent test -- test/args.test.ts
npm --prefix packages/coding-agent run build
python3 -m py_compile rcg_project/local_qwen_rcg_worker.py \
  rcg_project/api_harness_policy.py rcg_project/fascia_moe/*.py
python3 rcg_project/test_api_harness_policy.py
```

## Documentation

- [FASCIA_TECHNICAL_REPORT.md](FASCIA_TECHNICAL_REPORT.md): architecture,
  training, failure analysis, and benchmark results.
- [RCG_SPECULATIVE_SWARM.md](RCG_SPECULATIVE_SWARM.md): runtime and harness
  design.
- [API_HARNESS_COMPARISON.md](API_HARNESS_COMPARISON.md): prompt/API policy
  versus trained Fascia.
- [ASSET_MANIFEST.md](ASSET_MANIFEST.md): local checkpoint, data, cache, and
  validation artifact paths.

## Layout

- `packages/coding-agent/src/core/speculative-swarm.ts`: dynamic swarm
  orchestration, repeated tool-instance expansion, evidence routing, takeover.
- `packages/coding-agent/src/core/local-qwen-runtime.ts`: shared native runtime.
- `rcg_project/local_qwen_rcg_worker.py`: MLX Qwen worker with trained/prompt
  policy modes.
- `rcg_project/fascia_moe/`: Fascia-MoE inference model and policy engine.
- `rcg_project/api_harness_policy.py`: prompt-only policy baseline.

## Scope

This is a research integration snapshot. It demonstrates that a trained
Fascia controller can reduce main-agent turns and stabilize forced multiturn
swarm behavior in local controlled tasks. It is not yet a claim of production
generalization across all external agent benchmarks.
