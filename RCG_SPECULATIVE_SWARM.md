# Fascia Speculative Swarm Architecture

Fascia is the connective policy layer between the main agent, subagents, tool
runtime, and residual evidence cache. In the biological analogy used by the
paper draft:

- agent swarm = organs
- orchestrator = brain / nervous system
- Fascia = connective tissue that coordinates, routes, and transfers tension

## Runtime Flow

1. Serialize the user task, available tools, schemas, and retained residual
   evidence.
2. A dynamic policy builds an action DAG. In trained mode this is the
   Fascia-MoE controller; in prompt mode it is a Qwen/API JSON planner plus
   deterministic harness checks.
3. The harness expands coarse tool-level actions into concrete action
   instances when the query requests repeated calls to the same tool, for
   example two `geocode(place=...)` branches.
4. Ready branches run concurrently in one local Qwen3-8B process.
5. Tool results are recorded in the shared ledger and injected into later
   waves as explicit residual evidence.
6. The policy scores coherence, contribution, novelty, terminal failure type,
   recovery action, solvability, and takeover readiness.
7. If retained evidence is structurally complete, the swarm can directly take
   over the final answer instead of forcing the main agent to regenerate the
   same result.

## Fascia-MoE Controller

The validated v3 checkpoint uses:

- input width: 4096 Qwen hidden states
- latent width: 512
- shared Transformer layers: 3
- experts: 8
- sparse top-k experts: 2
- expert layers: 2
- max branches: 24
- max waves: 12
- max concurrency: 8
- parameters: 135,095,394

The controller heads predict:

- branch admission and priority
- execution stage and wave size
- dependency and residual routing
- coherence, contribution, novelty, and recovery class
- branch count, task completion, solvability, failure reporting, halt, and
  direct takeover

## Prompt/API Policy Mode

`--swarm-policy prompt` keeps the same harness but replaces trained policy
inference with one prompt/API planning call. It validates tool names and
schemas, then relies on deterministic harness logic for routing, failure
classification, completion checks, and direct takeover.

This mode is model-service portable and useful as a correctness baseline. It
does not require Fascia weights or Qwen hidden-state access, but it spends
extra autoregressive tokens and can be less stable on forced multiturn probes.

## Cache And Residual Semantics

Branches do not mutate one another's private KV tensors. They share:

- one model process and cache manager
- prompt-prefix reuse
- a tool-result ledger keyed by tool name and stable arguments
- explicit residual evidence messages between waves
- per-branch runtime slots for continuation

This keeps cache management safe while still letting later branches consume
verified results from earlier branches.

## Repeated Tool Instance Expansion

The latest harness fix addresses a concrete v3 failure mode. Fascia had
admitted a generic `geocode` action for a task requiring two separate places.
The first run retained only weather and reported geocode failure. The harness
now expands repeated single-string-argument tool actions into separate
argument-level briefs when the query names multiple concrete entities.

Validated fix:

- old `serial_route_weather`: failed, 1 tool call, tool coverage 0.333
- fixed `serial_route_weather`: success, 4 tool calls, coverage 1.0
- calls: `geocode(Golden Gate Bridge)`, `geocode(Ferry Building)`,
  `get_weather(San Francisco)`, `route_time(...)`

## Current Limits

- Fascia v3 is coupled to Qwen3-8B hidden width and tokenizer behavior.
- GAIA is evaluated only through text-compatible local rows with heuristic
  action labels, not official exact-answer scoring.
- Wall-clock speed depends on local MLX batching and memory pressure; main
  turns can shrink while absolute latency does not always beat a simple base
  loop on easy cases.
- Full external tau-bench/GAIA/SWE execution remains future work beyond the
  local proxy and controlled probe setup.
