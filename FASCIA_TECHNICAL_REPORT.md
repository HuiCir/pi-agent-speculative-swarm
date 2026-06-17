# Fascia Technical Report

## Executive Summary

Fascia is a trainable dynamic policy layer for speculative agent swarms. It
does not replace the base LLM. Instead, it controls branch admission,
dependency waves, residual evidence routing, coherence selection, failure
handling, and direct takeover.

The latest validated run uses a local Qwen3-8B base model and a Fascia-MoE
controller checkpoint at step 1100. A prior v3 long-dependency failure was
traced to the harness treating a repeated tool as one generic action. The
latest harness expands repeated single-argument tool calls into concrete
argument-level branches. After this fix, trained Fascia reaches 100% success
on both long-dependency and forced-multiturn controlled probes.

## Model Architecture

Validated controller:

| Component | Value |
| --- | ---: |
| Input hidden width | 4096 |
| Latent width | 512 |
| Shared layers | 3 |
| Experts | 8 |
| Active experts | top-2 sparse routing |
| Expert layers | 2 |
| Attention heads | 8 |
| Max branches | 24 |
| Max waves | 12 |
| Max concurrency | 8 |
| Recovery classes | 7 |
| Parameters | 135,095,394 |

Outputs:

- node-level admission, priority, stage, coherence, contribution, novelty,
  memory retention, recovery
- pair-level dependency and residual routing
- global branch count, wave sizes, wave count, completion, solvability,
  failure reporting, takeover, halt, remaining turns
- expert routing distribution

The controller consumes Qwen-derived query/action/evidence embeddings plus
runtime features. It is coupled to the Qwen3-8B hidden representation and is
not a model-agnostic API controller.

## Harness Architecture

The swarm harness provides the runtime substrate that Fascia controls:

1. Build a dynamic action DAG from task text, tools, schemas, and residual
   memory.
2. Expand repeated tool instances when a coarse action maps to multiple
   concrete query entities.
3. Run ready branches concurrently in a shared local Qwen process.
4. Record tool results in a stable ledger keyed by tool name and arguments.
5. Inject verified residual evidence into later waves.
6. Score retained and suppressed branches.
7. Allow direct takeover when retained evidence is complete and truthful.

The latest repeated-tool fix is deliberately in the harness rather than in the
checkpoint: it is an execution-grounding rule. The model may admit `geocode`;
the harness must instantiate `geocode(place=Golden Gate Bridge)` and
`geocode(place=Ferry Building)` when the query explicitly requires both.

## Training Data

Full data path:

```text
/Users/migraine/Downloads/orch/rcg/Fascia-MoE/data/fascia_events_v3_protocol_hotpot.jsonl
```

Rows by source:

| Source | Rows | Role |
| --- | ---: | --- |
| trajectory | 10,648 | primary parallel/action DAG data |
| swarm_protocol_v3 | 7,694 | residual replay, failure takeover, protocol states |
| tau_bench | 7,071 | agent domain action planning |
| mmlu | 2,400 | finite option-space branch planning |
| tau2_banking | 1,977 | domain action planning |
| tau2_telecom | 1,357 | domain action planning |
| tau2_retail | 1,282 | domain action planning |
| hotpotqa | 736 | low-weight multi-hop dependency auxiliary |
| swe_bench | 600 | issue-resolution planning |
| longbench | 500 | retrieval/verification planning |
| tau2_airline | 346 | domain action planning |
| clawbench | 130 | browser-agent plan proxy |
| long_horizon_v2 | 43 | focused long-horizon protocol cases |

HotpotQA is useful only as auxiliary long-dependency supervision. It lacks
real tool execution, API noise, residual cache transfer, and subagent runtime
behavior. It is therefore kept at low sampling weight:

```json
{"hotpotqa": 0.03}
```

## Training Configuration

```text
config: /Users/migraine/Downloads/orch/rcg/Fascia-MoE/configs/full_m5_v3_protocol_hotpot.json
steps: 1200
best step: 1100
learning rate: 1.5e-5
gradient accumulation: 2
max grad norm: 0.8
init checkpoint: runs/full_m5_v2_stable/checkpoints/fascia_best_step_800_0.89500.pt
encoder: Qwen3-8B, first 8 layers, cached embeddings
```

Best checkpoint:

```text
/Users/migraine/Downloads/orch/rcg/Fascia-MoE/runs/full_m5_v3_protocol_hotpot/checkpoints/fascia_best_step_1100_0.92843.pt
```

SHA-256:

```text
8ebfc0b38d99dde7ded323383051f772737c023445c6001c7fe4a1952d90c392
```

## Evaluation Results

### Offline Controller

987 sampled held-out rows:

| System | Composite |
| --- | ---: |
| Raw Fascia | 0.3647 |
| Trained Fascia | 0.9254 |
| Delta | +0.5607 |

### Merged Four-Way Controller Benchmark

320 rows across MMLU, LongBench, Trajectory, SWE Verified proxy, tau-bench,
GAIA text-compatible, and ClawBench:

| System | Action Composite |
| --- | ---: |
| Qwen base proxy | 0.3816 |
| Prompt/API harness proxy | 0.4230 |
| Raw Fascia | 0.1894 |
| Trained Fascia | 0.8091 |

Per-source trained Fascia action composite:

| Source | Rows | Score |
| --- | ---: | ---: |
| mmlu | 80 | 1.0000 |
| clawbench | 12 | 1.0000 |
| swe_verified | 50 | 0.9733 |
| trajectory | 80 | 0.8952 |
| longbench | 50 | 0.8400 |
| tau_bench | 24 | 0.5188 |
| gaia | 24 | 0.4365 |

GAIA caveat: the local GAIA set is text-compatible and heuristic. It is not
official exact-answer GAIA scoring.

### Long Dependency Probe

Six controlled long-chain tasks:

| System | Success | Mean Turns | Mean Tool Calls | Mean Latency | Takeover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen base | 100% | 4.33 | 6.33 | 50.68 s | 0% |
| Prompt/API harness | 100% | 1.00 | 3.83 | 64.31 s | 100% |
| Raw Fascia | 100% | 4.33 | 4.67 | 72.12 s | 0% |
| Trained Fascia | 100% | 1.00 | 3.83 | 57.91 s | 100% |

The trained controller compresses main-agent turns from 4.33 to 1.00 while
preserving task success. It is faster than prompt/API harness but still slower
than Qwen base on this easy local suite because local swarm concurrency and
branch prompt overhead dominate wall-clock time.

### Forced Multiturn Probe

Four controlled tasks requiring cross-branch reuse and takeover:

| System | Success | Mean Turns | Mean Tool Calls | Mean Latency | Takeover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen base | 25% | 5.00 | 12.00 | 162.79 s | 0% |
| Prompt/API harness | 50% | 1.00 | 6.50 | 154.93 s | 100% |
| Raw Fascia | 50% | 5.00 | 8.25 | 99.76 s | 0% |
| Trained Fascia | 100% | 1.00 | 6.50 | 73.70 s | 100% |

This is the strongest evidence for keeping v3: trained Fascia outperforms
base, prompt harness, and raw controller on success, turns, and latency.

## Failure Analysis

Prior v3 failure:

- case: `serial_route_weather`
- old result: failed
- old tool calls: only `get_weather(San Francisco)`
- old failure mode: generic `geocode` branch failed or timed out, so the
  harness reported geocode as unavailable and never executed `route_time`

Fixed result:

- case: `serial_route_weather`
- new result: success
- new tool calls:
  - `geocode(place=Golden Gate Bridge)`
  - `geocode(place=Ferry Building)`
  - `get_weather(city=San Francisco)`
  - `route_time(origin_lat=37.8199, origin_lon=-122.4783, destination_lat=37.7955, destination_lon=-122.3937)`

A first harness patch briefly over-expanded `geocode are required`. The final
patch filters non-entity phrases, and the clean rerun used exactly four tool
calls.

## Decision

The condition for falling back to v2 was not met. v3 has a clear positive
signal after the harness fix:

- old long-dependency failure fixed
- long dependency: 6/6 success
- forced multiturn: 4/4 success
- no risk flags in validation summary

The final repository should therefore publish the v3-compatible Fascia swarm
harness, not freeze the older v2 line.

## Remaining Work

- Replace heuristic GAIA controller labels with true end-to-end exact-answer
  checks where the base model can access required assets.
- Add real tau-bench/tau2 user simulator execution instead of controller proxy
  scoring.
- Improve local batching/runtime efficiency so turn compression also gives
  consistent wall-clock wins on easy tasks.
- Add more long-horizon data with repeated tool instances and argument-level
  branch labels, so the model learns less generic tool-level planning before
  harness fallback.
