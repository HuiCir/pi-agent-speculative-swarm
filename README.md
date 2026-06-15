# RCG Speculative Swarm

This repository integrates a trained RCG controller with the Pi agent harness
and a single native Qwen3-8B runtime. RCG plans distinct tool paths, scores
execution coherence, transfers verified residual evidence between waves, and
can promote a complete swarm result directly to the main answer.

The base model is not duplicated. Main-agent generation, subagents, KV cache,
and RCG embeddings share one local Qwen3-8B process.

## Included

- Full Pi monorepo snapshot with the speculative swarm integration.
- RCG-MoE-OPD V1 controller and training pipeline.
- Coupled path selection, outcome scoring, recovery routing, and takeover.
- Native MLX Qwen runtime with shared prompt/KV cache.
- Final training logs, validation summaries, and benchmark traces.
- Release assets for the selected checkpoint and generated training dataset.

See [RCG_SPECULATIVE_SWARM.md](RCG_SPECULATIVE_SWARM.md) for the runtime design
and [rcg_project/README.md](rcg_project/README.md) for training details.

## Validated Snapshot

Selected checkpoint:

| Field | Value |
| --- | --- |
| Architecture | `rcg_moe_opd_v1_coupled` |
| Checkpoint version | 13 |
| Selected step | 200 |
| RCG parameters | 76,190,821 |
| Validation composite | 0.725805 |
| SHA-256 | `5b5a720b50a2fdaf5f7b1ec7457dbcf0513a40d2e58d81aae7cef719980358ae` |

Controlled seven-case local Qwen3-8B agent harness:

| System | Success | Mean main turns | Mean latency | Direct takeover |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B Pi base | 100% | 3.143 | 28.611 s | 0% |
| Qwen3-8B + RCG swarm | 100% | 1.000 | 22.978 s | 100% |

In this harness, RCG reduced mean latency by 19.7% (`1.245x`) while preserving
task success, tool coverage, dependency accuracy, failure attribution, and
hallucination-free scoring. This is a focused integration benchmark, not a
claim of broad production generalization. The long-chain case remained slower
because multi-wave execution was limited by local hardware.

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

gh release download rcg-moe-opd-v1-v13 \
  --pattern rcg_moe_opd_v1_harness_adapter_best.pt \
  --dir rcg_project/checkpoints
```

Run the native Qwen + RCG swarm:

```bash
./pi-test.sh \
  --swarm \
  --swarm-local-model-path /path/to/Qwen3-8B \
  --swarm-local-python rcg_project/.venv/bin/python \
  --swarm-local-rcg-checkpoint \
    rcg_project/checkpoints/rcg_moe_opd_v1_harness_adapter_best.pt \
  --mode text \
  --no-session \
  -p "Your tool-using task"
```

The non-swarm local baseline uses the same model and runtime:

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
npm --prefix packages/coding-agent test -- \
  test/speculative-swarm.test.ts \
  test/local-qwen-runtime.test.ts
npm run build

rcg_project/.venv/bin/python -m py_compile \
  rcg_project/*.py rcg_project/rcg/*.py
```

## Layout

- `packages/coding-agent/src/core/speculative-swarm.ts`: swarm orchestration.
- `packages/coding-agent/src/core/local-qwen-runtime.ts`: shared native runtime.
- `rcg_project/rcg/moe_opd_controller.py`: trainable RCG controller.
- `rcg_project/train_rcg_moe_opd_v1.py`: phased/coupled training.
- `rcg_project/rcg_policy_server.py`: checkpoint inference and routing.
- `rcg_project/artifacts`: selected logs and evaluation evidence.

## Attribution

The agent harness is derived from the Pi monorepo and remains under the MIT
license. RCG-specific integration and training code is included in this
snapshot under the same repository license.
