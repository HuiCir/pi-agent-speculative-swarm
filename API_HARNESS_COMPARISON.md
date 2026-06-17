# Prompt/API Harness vs Trained Fascia

## Question

Can the Fascia controller be replaced by an untrained LLM API plus harness
logic?

Partly. For finite, typed tool spaces, a prompt/API planner plus deterministic
validation can often recover correct action DAGs. It is useful as a portable
baseline and fallback. The trained Fascia controller remains valuable when
policy latency, repeated planning, branch diversity, and nontrivial coherence
selection matter.

## Modes

Prompt/API harness:

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

Trained Fascia:

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

## Latest Controlled Probe Results

Long dependency probe, six tasks:

| System | Success | Mean turns | Mean latency | Direct takeover | Policy tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen base | 100% | 4.33 | 50.68 s | 0% | 0 |
| Prompt/API harness | 100% | 1.00 | 64.31 s | 100% | 4,819 |
| Raw Fascia | 100% | 4.33 | 72.12 s | 0% | 0 |
| Trained Fascia | 100% | 1.00 | 57.91 s | 100% | 0 |

Forced multiturn probe, four tasks:

| System | Success | Mean turns | Mean latency | Direct takeover | Policy tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen base | 25% | 5.00 | 162.79 s | 0% | 0 |
| Prompt/API harness | 50% | 1.00 | 154.93 s | 100% | 13,743 |
| Raw Fascia | 50% | 5.00 | 99.76 s | 0% | 0 |
| Trained Fascia | 100% | 1.00 | 73.70 s | 100% | 0 |

## Interpretation

The prompt/API harness can replace many hard-coded workflow decisions:

- candidate action planning
- serial/parallel scheduling
- dependency routing
- observable tool-call coherence
- terminal error classification
- direct takeover from verified residual evidence

Training adds:

- low-latency policy inference without generated planning tokens
- learned relevance and branch-count calibration from Qwen hidden states
- better stability on forced multiturn tasks
- learned recovery and takeover behavior under noisy tool outcomes
- a path toward deeper cache-aware decisions unavailable to normal API-only
  harnesses

## Current Recommendation

Keep both modes:

- Use `prompt` as a portable correctness baseline and emergency fallback.
- Use `trained` as the primary Fascia mode for local Qwen-controlled swarm
  experiments and latency-sensitive repeated planning.

The latest v3 run supports continuing with Fascia rather than freezing the
older v2 line: the prior long-dependency failure was fixed by harness
repeated-tool expansion, and trained Fascia reached 100% on both long
dependency and forced multiturn probes.
