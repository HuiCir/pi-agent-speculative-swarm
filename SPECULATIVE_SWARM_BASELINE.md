# Speculative Swarm Baseline

This repo contains a source-level integration of a residual speculative swarm
inside the Pi agent loop. The current baseline keeps the normal single-agent
path unchanged unless `--swarm` is enabled.

## Scope

- Main model and swarm subagents can both use `deepseek/deepseek-v4-flash`.
- The swarm runs before each main agent model request and injects a structured
  memory context as draft evidence.
- The design combines speculative fast drafts, residual memory across rounds,
  branch-level runtime state, and late background consolidation.

## Modified Files

Core agent wiring:

- `packages/agent/src/types.ts`
  - Adds `SpeculativeSwarmPrepareContext` and `SpeculativeSwarmController`.
  - Extends `AgentLoopConfig` with `speculativeSwarm`.
- `packages/agent/src/agent.ts`
  - Stores the optional swarm controller on `Agent`.
  - Passes it into the loop config.
- `packages/agent/src/agent-loop.ts`
  - Invokes `speculativeSwarm.prepareContext(...)` before a main model request.
  - Filters prior swarm memory messages to avoid recursive prompt buildup.
  - Preserves assistant tool-call messages before appending tool results.

Coding-agent integration:

- `packages/coding-agent/src/core/speculative-swarm.ts`
  - Implements the residual speculative swarm controller.
  - Handles branch briefs, subagent execution, scoring, retention/suppression,
    residual summaries, runtime deltas, FIFO early release, and trace output.
- `packages/coding-agent/src/core/sdk.ts`
  - Creates the swarm controller from session options.
- `packages/coding-agent/src/core/agent-session-services.ts`
  - Propagates swarm options into session creation.
- `packages/coding-agent/src/main.ts`
  - Resolves the swarm model and maps CLI flags into session options.
- `packages/coding-agent/src/cli/args.ts`
  - Adds `--swarm`, `--swarm-model`, `--swarm-agents`, `--swarm-rounds`,
    `--swarm-max-turns`, `--swarm-timeout-ms`, `--swarm-tool-policy`, and
    `--swarm-allow-bash`.

Provider compatibility:

- `packages/ai/src/providers/openai-completions.ts`
  - Adds compatibility handling needed by DeepSeek-style responses and replayed
    tool-call message sequences.

## Runtime Architecture

The active controller is `ResidualSpeculativeSwarmController`.

1. The agent loop calls `prepareContext` before the main model request.
2. The controller removes previous swarm memory messages from the base context.
3. On the first request, a fast `Turn0OrchestratorBrief` produces branch briefs.
4. Four default subagents run concurrently:
   - `SwarmAgent-A`
   - `SwarmAgent-B`
   - `SwarmAgent-C`
   - `SwarmAgent-D`
5. Each subagent receives a stable system prompt plus a dynamic user message
   containing its branch brief, residual memory, and branch runtime delta.
6. FIFO early release returns as soon as a satisfactory draft is available.
7. Slower drafts continue in the background and later update residual memory.
8. The main agent receives a compact swarm memory message as draft evidence.

## Current Optimization Features

- Concurrent subagent drafts rather than sequential external calls.
- FIFO early release instead of waiting for the slowest subagent.
- Background round completion for later voting and residual reuse.
- Branch-level runtime slots keyed by branch id.
- Stable subagent system prompts to improve provider-side cache reuse.
- Dynamic residual/runtime data moved into the final user message.
- Absence-claim guard to penalize drafts that claim missing evidence from
  truncated reads.
- Trace output through `PI_SWARM_TRACE_FILE`.

## Disabled Experimental Feature

Runtime replay is present but disabled by default:

```sh
PI_SWARM_ENABLE_RUNTIME_REPLAY=1
```

It should remain off for the clean baseline. Earlier smoke testing showed stale
runtime replay can mix residual entities across rounds and produce incorrect
answers. Future work should re-enable it only after adding stricter branch/task
identity checks.

## CLI Baseline

Single-agent v4flash baseline:

```sh
pi --model deepseek/deepseek-v4-flash --mode text --no-session -p "<task>"
```

Internal swarm v4flash baseline:

```sh
PI_SWARM_TRACE_FILE=swarm_results/<run>/<case>.trace.jsonl \
pi --model deepseek/deepseek-v4-flash \
  --swarm \
  --swarm-model deepseek/deepseek-v4-flash \
  --swarm-rounds 1 \
  --swarm-max-turns 3 \
  --swarm-timeout-ms 120000 \
  --swarm-tool-policy readonly \
  --swarm-allow-bash \
  --mode text \
  --no-session \
  -p "<task>"
```

## Verification State

Previously observed source-level results:

- Older hard internal swarm passed accuracy but was slower than single-agent.
- Branch-brief swarm increased branch diversity but added large latency.
- All-flash comparison removed model unfairness; swarm remained slower.
- Runtime/cache optimized swarm improved latency on 5 screened long cases while
  preserving `okRate = 1.0`.

The latest source state still needs a fresh build/check and a new unified
v4flash benchmark pass before it should be considered release-ready.

Recommended gates:

```sh
npm --prefix packages/coding-agent run build
npm run check
```

Then run paired single/swarm v4flash evaluations on:

- Prior tree/tool-graph and long composite trajectory cases.
- Deep-planning derived shopping cases.
- Deep-planning travel database exploration cases, clearly labeled as derived
  environment probes unless official query/evaluator files are added.

## Known Risks

- Swarm can be slower than single-agent when the task is short or when the main
  model can read all evidence in one pass.
- Residual transfer can cause entity drift if branch identity is weak.
- Extra draft context can distract the main agent when retained drafts are too
  broad or redundant.
- Tool-parameter accuracy is useful diagnostically, but it should not be the
  sole success metric unless parameter mismatch is known to break task success.
- Deep-planning travel data currently appears to be an environment archive, not
  a complete official task/evaluator set in this workspace.

## Next Optimization Entry Points

- Make branch identity stricter before reintroducing runtime replay.
- Add adaptive concurrency: keep FIFO fast paths while suppressing homogeneous
  duplicate branches.
- Split context selection from execution more cleanly inside the orchestrator.
- Improve residual compression so later rounds inherit useful partial work
  without leaking unrelated entities.
- Add unified benchmark scripts that report accuracy, latency, early-release
  latency, cache ratio, subagent diversity, background completion, and token use.
