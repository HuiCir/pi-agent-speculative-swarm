# Fascia Asset Manifest

The GitHub repository intentionally excludes large training and model assets.
This file records the validated local artifacts.

## Model

| Artifact | Local Path | Size | SHA-256 |
| --- | --- | ---: | --- |
| Best Fascia checkpoint | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/checkpoints/fascia_best_step_1100_0.92843.pt` | 531M | `8ebfc0b38d99dde7ded323383051f772737c023445c6001c7fe4a1952d90c392` |

Checkpoint metadata:

- version: `fascia_moe_v3_protocol_hotpot_takeover`
- step: 1100
- parameters: 135,095,394
- finite weights: verified

## Data And Cache

| Artifact | Local Path | Size | SHA-256 |
| --- | --- | ---: | --- |
| Full training JSONL | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/data/fascia_events_v3_protocol_hotpot.jsonl` | 376M | `1327f6a6adc5d6427ba75acc722068eca1ece4e76977fca0693f2d12b38dfc3b` |
| Qwen encoder cache | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/cache/fascia_qwen8l_v3_protocol_hotpot.pt` | 533M | not recomputed here |

Training rows by source:

| Source | Rows |
| --- | ---: |
| trajectory | 10,648 |
| swarm_protocol_v3 | 7,694 |
| tau_bench | 7,071 |
| mmlu | 2,400 |
| tau2_banking | 1,977 |
| tau2_telecom | 1,357 |
| tau2_retail | 1,282 |
| hotpotqa | 736 |
| swe_bench | 600 |
| longbench | 500 |
| tau2_airline | 346 |
| clawbench | 130 |
| long_horizon_v2 | 43 |

HotpotQA is intentionally low-weight auxiliary data. The training config sets
`source_weights.hotpotqa = 0.03`.

## Validation

| Artifact | Local Path | Size | SHA-256 |
| --- | --- | ---: | --- |
| Validation summary | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/benchmarks/best_fascia_best_step_1100_0.92843_validation_summary.json` | small | `c2eac408688b95ae5392dcaa70f017ac76055e4ba4c98cdefb8c9bc8dbfba701` |
| Offline benchmark | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/benchmarks/best_fascia_best_step_1100_0.92843_benchmark_suite.json` | small | see local file |
| Merged four-way benchmark | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/benchmarks/best_fascia_best_step_1100_0.92843_merged_four_way.json` | small | see local file |
| Long dependency probe | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/e2e/best_fascia_best_step_1100_0.92843_long_dependency_probe.json` | small | see local file |
| Forced multiturn probe | `/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/e2e/best_fascia_best_step_1100_0.92843_forced_multiturn_probe.json` | small | see local file |

Important local-only debug artifact:

```text
/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/e2e/fix_repeated_tool_long_dependency_trained_only_v2.json
```

This file proves the repeated-tool harness fix on the prior failing
`serial_route_weather` case.
