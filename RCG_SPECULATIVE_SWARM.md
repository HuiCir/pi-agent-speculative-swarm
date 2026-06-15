# RCG Speculative Swarm Architecture

## Runtime

The release uses one local Qwen3-8B process for every language-model operation.
The process owns model weights, prompt cache, branch KV cache, batch prefill,
and concurrent decode. RCG is an external trainable controller, but it reuses
Qwen hidden representations instead of running a second language model.

The agent loop follows this sequence:

1. Serialize the query, available tools, parameters, and retained evidence.
2. RCG scores candidate actions and predicts branch count, wave assignment,
   dependencies, and initial concurrency.
3. Distinct admitted branches run concurrently in the shared Qwen runtime.
4. Tool results are returned to RCG for coherence, contribution, failure type,
   recovery action, and continuation scoring.
5. Verified residual evidence is routed to dependent branches in later waves.
6. RCG predicts task completion, solvability, failure reporting, and takeover.
7. A structurally complete result can become the main answer without another
   main-model reconstruction pass.

## Controller

`RcgMoeOpdV1` has a shared representation expert plus three specialists:

- Planner: relevance, residual distinction, dependency, count, and wave heads.
- Outcome: coherence, contribution, failure type, recovery, and continuation.
- Orchestration: completion, solvability, takeover, failure reporting, routing.

Version 13 enables coupled selection. Admission combines query relevance with
residual distinction so a branch must be useful and add coverage beyond paths
already selected. Coherence is not a top-1 score: all accepted, distinct,
locally valid paths can contribute to the total draft.

## Harness Behavior

The swarm integration adds:

- Dynamic action-space planning instead of fixed serial/parallel modes.
- Batched first-wave execution and dependency-aware later waves.
- Residual result transfer between branch sessions.
- Terminal tool-error classification without blind retry.
- Recovery routing to a distinct fallback tool when one exists.
- Truthful partial failure output when a task is not fully solvable.
- Direct takeover when retained evidence is complete and coherent.
- Suppression of redundant branches and duplicate tool signatures.

## Cache Semantics

Branches do not directly read another branch's mutable KV tensors. They share:

- The same model weights and cache manager.
- Stable prompt prefixes that permit prefix-cache reuse.
- Explicit verified residual evidence injected into later branch context.
- Per-branch session history for continued exploration.

This avoids unsafe cross-session KV mutation while preserving the useful
information transfer needed for speculative continuation.

## Known Limits

- The selected checkpoint is coupled to Qwen3-8B hidden width and tokenizer
  behavior. It is not a model-agnostic API controller.
- MLX concurrency is hardware-bound; long dependency chains can remain slower
  than the base loop even when main-agent turns are reduced.
- Direct takeover needs a task-specific renderer for polished natural-language
  responses. The release prioritizes verified evidence and failure truthfulness.
- The included benchmark is intentionally small and controlled. Broader
  Trajectory-Bench evaluation is still required before production claims.
