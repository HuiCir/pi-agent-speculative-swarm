# Checkpoints

Fascia checkpoints are not committed to this repository.

Validated local checkpoint:

```text
/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/checkpoints/fascia_best_step_1100_0.92843.pt
```

Expected SHA-256:

```text
8ebfc0b38d99dde7ded323383051f772737c023445c6001c7fe4a1952d90c392
```

Run trained mode by passing the checkpoint explicitly:

```bash
./pi-test.sh \
  --swarm \
  --swarm-policy trained \
  --swarm-local-rcg-checkpoint /path/to/fascia_best_step_1100_0.92843.pt \
  --swarm-local-model-path /path/to/Qwen3-8B \
  --swarm-local-python rcg_project/.venv/bin/python \
  -p "Your tool task"
```
