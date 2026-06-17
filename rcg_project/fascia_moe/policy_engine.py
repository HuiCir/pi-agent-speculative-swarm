from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch

from .config import FasciaConfig
from .encoding import candidate_text, evidence_text, query_text, runtime_features
from .model import FasciaMoE
from .schema import RECOVERY_NAMES


FAILURE_RECOVERY = {
    "success": "none",
    "timeout": "report_failure",
    "auth_or_quota": "report_failure",
    "server_or_network": "report_failure",
    "api_unavailable": "abandon_action",
    "blocked_dependency": "wait_dependency",
    "invalid_call": "repair_call",
    "provenance_mismatch": "replan_distinct_action",
    "empty_result": "request_more_evidence",
    "other_failure": "report_failure",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "using",
    "use",
    "return",
    "returns",
    "get",
    "check",
    "status",
    "current",
    "required",
    "requires",
    "produced",
    "produce",
    "available",
    "tool",
    "tools",
    "call",
    "exact",
    "result",
    "results",
    "independent",
    "independently",
    "complete",
    "execute",
    "report",
    "when",
    "possible",
}

TOKEN_ALIASES = {
    "summarize": "summary",
    "summarizes": "summary",
    "summarized": "summary",
    "summarizing": "summary",
    "reviews": "review",
    "reviewed": "review",
    "sentiments": "sentiment",
    "authentication": "auth",
    "authenticated": "auth",
    "unauthorized": "auth",
    "authorization": "auth",
}

INTENT_GROUPS = (
    {"review", "summary", "sentiment"},
    {"budget", "limit"},
    {"billing", "account", "auth", "401"},
    {"order", "total"},
    {"customer", "resolve"},
    {"inventory", "stock"},
    {"weather", "forecast"},
    {"route", "geocode", "coordinate"},
)


def _device(value: str | None) -> torch.device:
    requested = str(value or "cpu")
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        requested = "cpu"
    return torch.device(requested)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _clip_text(value: str, limit: int = 4000) -> str:
    value = str(value or "")
    return value if len(value) <= limit else value[-limit:]


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value or ""))
    expanded = expanded.replace("_", " ").replace("-", " ")
    items = set()
    for token in re.findall(r"[a-zA-Z0-9]{3,}", expanded.lower()):
        if token in STOPWORDS:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        token = TOKEN_ALIASES.get(token, token)
        items.add(token)
    return items


class FasciaPolicyEngine:
    """Protocol adapter between pi-agent swarm and the parameterized Fascia MoE.

    The harness still owns hard safety limits and tool execution. Fascia owns
    the dynamic policy: which actions to admit, which wave to run, which branch
    is coherent, when to halt/take over, and how residual evidence should flow.
    """

    query_tokens = 768
    candidate_tokens = 256
    evidence_tokens = 768
    encode_batch_size = 8
    path_encode_batch_size = 4

    def __init__(self, checkpoint: str | Path, encoder, hidden_size: int, device: str | None = None):
        self.checkpoint = Path(checkpoint)
        self.encoder = encoder
        self.device = _device(device)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        config = FasciaConfig.from_dict(payload["model_config"])
        if int(hidden_size) != int(config.input_dim):
            raise ValueError(
                f"encoder hidden size {hidden_size} does not match Fascia input_dim {config.input_dim}"
            )
        self.model = FasciaMoE(config).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.step = int(payload.get("step", 0))
        self.version = str(payload.get("version", "fascia_moe_v1"))
        self.state_by_key: dict[str, torch.Tensor] = {}

    @classmethod
    def from_external_encoder(cls, args, encoder, hidden_size: int) -> "FasciaPolicyEngine":
        return cls(args.rcg_checkpoint, encoder, hidden_size, getattr(args, "rcg_device", "cpu"))

    @staticmethod
    def _task_key(payload: dict[str, Any]) -> str:
        task = str(payload.get("task") or "")
        epoch = payload.get("globalState", {}).get("taskEpoch", "")
        return hashlib.sha256(f"{epoch}:{task}".encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _tool_name(action: dict[str, Any]) -> str:
        return str(action.get("toolName") or action.get("name") or action.get("title") or "")

    @staticmethod
    def _tool_description(action: dict[str, Any]) -> str:
        return str(action.get("toolDescription") or action.get("description") or action.get("objective") or "")

    @classmethod
    def _lexical_intent(cls, task: str, tools: list[dict[str, Any]]) -> list[float]:
        query_tokens = _tokens(task)
        if not query_tokens:
            return [0.0] * len(tools)
        scores = []
        for tool in tools:
            name_tokens = _tokens(cls._tool_name(tool))
            description_tokens = _tokens(cls._tool_description(tool))
            parameter_tokens = _tokens(_json(tool.get("parameters") or {}))
            weighted_hits = (
                2.5 * len(query_tokens & name_tokens)
                + 1.0 * len(query_tokens & description_tokens)
                + 0.5 * len(query_tokens & parameter_tokens)
            )
            weighted_size = (
                2.5 * max(len(name_tokens), 1)
                + 1.0 * max(len(description_tokens), 1)
                + 0.5 * max(len(parameter_tokens), 1)
            )
            coverage = weighted_hits / max(weighted_size, 1.0)
            recall = len(query_tokens & (name_tokens | description_tokens | parameter_tokens)) / max(
                len(query_tokens), 1
            )
            name_hits = len(query_tokens & name_tokens)
            name_precision = name_hits / max(len(name_tokens), 1)
            description_hits = len(query_tokens & description_tokens)
            score = 0.7 * coverage + 0.3 * recall
            semantic_tokens = name_tokens | description_tokens | parameter_tokens
            if query_tokens & name_tokens:
                score = max(score, 0.45)
            for group in INTENT_GROUPS:
                query_group_hits = query_tokens & group
                tool_group_hits = semantic_tokens & group
                if query_group_hits and tool_group_hits:
                    score = max(score, 0.42 + 0.06 * min(len(query_group_hits | tool_group_hits), 3))
            if name_tokens and name_precision < 0.75 and description_hits < 2:
                score *= max(0.25, name_precision)
            # Downstream and recovery tools are often semantically adjacent to
            # their prerequisites. Keep them out of the first plan unless the
            # user explicitly names the distinctive action they perform.
            distinctive_requirements = {
                "total": {"total"},
                "budget": {"budget", "limit"},
                "fallback": {"fallback"},
                "billing": {"billing", "account"},
                "rate": {"rate", "exchange", "currency", "fx"},
            }
            for marker, triggers in distinctive_requirements.items():
                if marker in name_tokens and not (query_tokens & triggers):
                    score *= 0.35
            scores.append(score)
        maximum = max(scores, default=0.0)
        if maximum <= 0:
            return scores
        return [score / maximum for score in scores]

    @staticmethod
    def execution_flags(action: dict[str, Any], draft: dict[str, Any]) -> dict[str, bool]:
        expected_tool = FasciaPolicyEngine._tool_name(action)
        calls = draft.get("toolCalls") or []
        expected_calls = [
            call for call in calls if str(call.get("toolName") or "") == expected_tool
        ]
        successful_expected = [
            call
            for call in expected_calls
            if not bool(call.get("isError"))
            and bool(str(call.get("resultText") or call.get("resultPreview") or "").strip())
        ]
        successful_other = [
            call
            for call in calls
            if str(call.get("toolName") or "") != expected_tool and not bool(call.get("isError"))
        ]
        recovered = bool(successful_expected)
        terminal_error = (
            str(draft.get("status")) != "ok"
            or not recovered
            or (bool(expected_calls) and bool(expected_calls[-1].get("isError")))
        )
        return {
            "success": recovered and not terminal_error,
            "terminal_error": terminal_error,
            "tool_mismatch": bool(successful_other) and not recovered,
        }

    @staticmethod
    def structural_failure_type(action: dict[str, Any], draft: dict[str, Any], flags: dict[str, bool]) -> str:
        if flags["tool_mismatch"]:
            return "provenance_mismatch"
        calls = draft.get("toolCalls") or []
        text = " ".join(
            str(
                call.get("resultText")
                or call.get("resultPreview")
                or call.get("error")
                or ""
            )
            for call in calls
        ).lower()
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if any(marker in text for marker in ("401", "403", "unauthorized", "forbidden", "quota", "api key")):
            return "auth_or_quota"
        if any(marker in text for marker in ("500", "502", "503", "504", "bad gateway", "service unavailable")):
            return "server_or_network"
        if any(marker in text for marker in ("not found", "doesn't exist", "does not exist", "suspended")):
            return "api_unavailable"
        if any(
            marker in text
            for marker in (
                "400",
                "invalid parameter",
                "invalid customer",
                "missing required",
                "validation error",
                "do not match",
                "does not match",
            )
        ):
            return "invalid_call"
        if not text.strip():
            return "empty_result"
        return "other_failure"

    @staticmethod
    def _source_type(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
        names = [str(item.get("name") or "") for item in candidates]
        if names and all(name.startswith("option_") for name in names):
            return "mmlu"
        if any(name.startswith(("retrieve_chunk_", "verify_option_")) for name in names):
            return "longbench"
        if any(name in {"inspect_constraints", "open_target_site", "submit_action"} for name in names):
            return "clawbench"
        if any("patch" in name.lower() or "test" in name.lower() for name in names):
            return "swe_bench"
        return "trajectory"

    def _row(
        self,
        payload: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        evidence_by_name: dict[str, dict[str, Any]] | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        query = str(payload.get("task") or "")
        latest = str(payload.get("latestObservation") or "")
        global_state = payload.get("globalState") or {}
        if latest and latest.strip() != query.strip():
            query = f"{query}\n\n<LATEST_OBSERVATION>\n{_clip_text(latest)}\n</LATEST_OBSERVATION>"
        if global_state:
            compact = {
                "completedActions": global_state.get("completedActions") or [],
                "toolResults": global_state.get("toolResults") or [],
                "activeBranches": global_state.get("activeBranches") or [],
            }
            if any(compact.values()):
                query = f"{query}\n\n<RUNTIME_STATE>\n{_clip_text(_json(compact))}\n</RUNTIME_STATE>"
        rows = []
        evidence_by_name = evidence_by_name or {}
        for index, item in enumerate(candidates):
            name = str(item.get("name") or self._tool_name(item) or f"action_{index + 1}")
            evidence = evidence_by_name.get(name) or {}
            runtime = evidence.get("runtime") or {
                "observed": False,
                "success": False,
                "failure_type": "",
                "latency_ratio": 0.0,
                "retry_count": 0,
            }
            rows.append(
                {
                    "name": name,
                    "description": self._tool_description(item),
                    "parameters": item.get("parameters") or {},
                    "evidence": evidence.get("evidence", ""),
                    "runtime": runtime,
                }
            )
        return {
            "schema": "fascia_event_v1",
            "task_id": self._task_key(payload),
            "group_id": self._task_key(payload),
            "source_type": source_type or self._source_type(payload, rows),
            "domain": "runtime",
            "query": query,
            "candidates": rows,
            "metadata": {"time_limit": 0},
        }

    def _encode(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        candidates = row["candidates"]
        node, global_values = runtime_features(row)
        query_hidden = self.encoder(
            [query_text(row)], self.query_tokens, 1
        )["mean"].float()
        candidate_hidden = self.encoder(
            [candidate_text(item) for item in candidates],
            self.candidate_tokens,
            self.encode_batch_size,
        )["mean"].float()
        evidence_hidden = self.encoder(
            [evidence_text(item) for item in candidates],
            self.evidence_tokens,
            self.path_encode_batch_size,
        )["mean"].float()
        return {
            "query_hidden": query_hidden.to(self.device),
            "candidate_hidden": candidate_hidden.to(self.device),
            "evidence_hidden": evidence_hidden.to(self.device),
            "node_features": node.to(device=self.device, dtype=torch.float32),
            "global_features": global_values.to(device=self.device, dtype=torch.float32),
        }

    @torch.inference_mode()
    def _forward(self, payload: dict[str, Any], row: dict[str, Any]) -> dict[str, torch.Tensor]:
        key = self._task_key(payload)
        previous = self.state_by_key.get(key)
        output = self.model(**self._encode(row), previous_state=previous)
        self.state_by_key[key] = output["recurrent_state"].detach()
        return output

    @staticmethod
    def _acyclic_dependencies(scores: torch.Tensor, selected: list[int], threshold: float) -> dict[int, list[int]]:
        depends_on: dict[int, list[int]] = {index: [] for index in selected}
        reachability: dict[int, set[int]] = {index: set() for index in selected}
        candidates = sorted(
            (
                (float(scores[target, source]), target, source)
                for target in selected
                for source in selected
                if target != source and float(scores[target, source]) >= threshold
            ),
            reverse=True,
        )
        for _, target, source in candidates:
            if source in reachability[target]:
                continue
            depends_on[target].append(source)
            downstream = {target, *reachability[target]}
            upstream = [node for node in selected if source == node or source in reachability[node]]
            for node in upstream:
                reachability[node].update(downstream)
        return depends_on

    @staticmethod
    def _textual_dependencies(tools: list[dict[str, Any]], selected: list[int]) -> dict[int, list[int]]:
        selected_set = set(selected)
        depends: dict[int, list[int]] = {index: [] for index in selected}
        for target in selected:
            description = FasciaPolicyEngine._tool_description(tools[target]).lower()
            for source in selected:
                if source == target:
                    continue
                source_name = re.escape(str(FasciaPolicyEngine._tool_name(tools[source])).lower())
                source_words = re.escape(str(FasciaPolicyEngine._tool_name(tools[source])).lower().replace("_", " "))
                source_pattern = rf"(?:{source_name}|{source_words})"
                explicit_prerequisite = bool(
                    re.search(rf"\brequires?\b.{{0,120}}\b(?:produced|returned)\s+by\s+{source_pattern}", description)
                    or re.search(rf"\b(?:after|only after)\b.{{0,120}}{source_pattern}", description)
                    or re.search(rf"\b(?:failure|error)\b.{{0,80}}\bfrom\s+{source_pattern}", description)
                )
                if explicit_prerequisite:
                    depends[target].append(source)
        for target in list(depends):
            depends[target] = [source for source in dict.fromkeys(depends[target]) if source in selected_set]
        return depends

    @staticmethod
    def _break_dependency_cycles(depends: dict[int, list[int]], selected: list[int]) -> dict[int, list[int]]:
        selected_set = set(selected)
        clean: dict[int, list[int]] = {index: [] for index in selected}
        adjacency: dict[int, list[int]] = {index: [] for index in selected}

        def reaches(start: int, goal: int) -> bool:
            stack = list(adjacency.get(start, []))
            seen = set()
            while stack:
                node = stack.pop()
                if node == goal:
                    return True
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(adjacency.get(node, []))
            return False

        for target in selected:
            for source in depends.get(target, []):
                if source not in selected_set or source == target:
                    continue
                # Dependency edge is source -> target. Adding it would create
                # a cycle if target already reaches source.
                if reaches(target, source):
                    continue
                clean[target].append(source)
                adjacency[source].append(target)
        return clean

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        tools = payload.get("tools") or []
        maximum = max(1, int(payload.get("maxBranches") or 4))
        if not tools:
            return {"actions": [], "modelStep": self.step, "latencyMs": 0.0}
        row = self._row(payload, tools)
        output = self._forward(payload, row)
        admission = output["admission"].sigmoid().detach().cpu()
        priority = output["priority"].sigmoid().detach().cpu()
        stage = output["stage_logits"].argmax(-1).detach().cpu()
        dependency = output["dependency"].sigmoid().detach().cpu()
        task_solvability = float(output["task_solvable"].sigmoid().detach().cpu())
        must_report_failure = float(output["must_report_failure"].sigmoid().detach().cpu())
        takeover_probability = float(output["takeover"].sigmoid().detach().cpu())
        halt_probability = float(output["halt"].sigmoid().detach().cpu())
        lexical = self._lexical_intent(str(payload.get("task") or ""), tools)
        lexical_tensor = torch.tensor(lexical)
        predicted_count = int(round(float(admission.sum())))
        head_count = int(output["branch_count_logits"].argmax().detach().cpu())
        explicit_intent_threshold = 0.40
        lexical_positive = [
            index
            for index, score in enumerate(lexical)
            if score >= explicit_intent_threshold
        ]
        threshold_count = int((admission >= 0.5).sum())
        if must_report_failure >= 0.62 and task_solvability <= 0.45:
            return {
                "actions": [],
                "modelStep": self.step,
                "modelVariant": self.version,
                "candidateCount": len(tools),
                "admittedCount": 0,
                "recommendedWaveSize": 0,
                "laterWaveSize": 0,
                "predictedWaveCount": 0,
                "planConfidence": 1.0 - task_solvability,
                "mustReportFailure": True,
                "taskSolvability": task_solvability,
                "takeoverProbability": takeover_probability,
                "haltProbability": halt_probability,
                "activeExperts": output["active_experts"].detach().cpu().tolist(),
                "latencyMs": (time.perf_counter() - started) * 1000,
            }
        if head_count > 0 and abs(head_count - predicted_count) <= 2:
            candidate_count = max(threshold_count, head_count)
        else:
            candidate_count = max(threshold_count, predicted_count)
        if len(lexical_positive) >= 2:
            candidate_count = max(candidate_count, max(len(lexical_positive), 1))
        desired_count = max(1, min(len(tools), candidate_count))
        rank_score = 0.45 * admission + 0.25 * priority + 0.30 * lexical_tensor
        eligible = lexical_positive if len(lexical_positive) >= 2 else list(range(len(tools)))
        selected = [
            int(index)
            for index in torch.tensor(
                eligible, dtype=torch.long
            )[rank_score[eligible].argsort(descending=True)].tolist()
            if admission[index] >= 0.35 or lexical[index] >= explicit_intent_threshold
        ][:desired_count]
        if len(selected) < desired_count:
            seen = set(selected)
            for index in rank_score.argsort(descending=True).tolist():
                index = int(index)
                if index in seen:
                    continue
                selected.append(index)
                seen.add(index)
                if len(selected) >= desired_count:
                    break
        if not selected:
            selected = [int(rank_score.argmax())]
        initial_wave_size = int(output["initial_wave_logits"].argmax().detach().cpu())
        later_wave_size = int(output["later_wave_logits"].argmax().detach().cpu())
        predicted_wave_count = int(output["wave_count_logits"].argmax().detach().cpu())
        initial_wave_size = max(1, min(maximum, len(selected), initial_wave_size))
        later_wave_size = max(1, min(maximum, len(selected), later_wave_size))
        depends_by_original = self._acyclic_dependencies(dependency, selected, threshold=0.55)
        textual_depends = self._textual_dependencies(tools, selected)
        for target, sources in textual_depends.items():
            merged = depends_by_original.setdefault(target, [])
            for source in sources:
                if source not in merged:
                    merged.append(source)
        depends_by_original = self._break_dependency_cycles(depends_by_original, selected)
        rank_by_original = {original: rank for rank, original in enumerate(selected)}
        stage_by_original = {original: max(1, int(stage[original])) for original in selected}
        for _ in range(len(selected)):
            changed = False
            for target, sources in depends_by_original.items():
                if not sources:
                    continue
                required_stage = 1 + max(stage_by_original.get(source, 1) for source in sources)
                if stage_by_original.get(target, 1) < required_stage:
                    stage_by_original[target] = required_stage
                    changed = True
            if not changed:
                break
        else:
            # The graph should already be acyclic; this bounded fallback avoids
            # a runtime stall if malformed tool text still creates a cycle.
            changed = False
        actions = []
        for rank, original in enumerate(selected):
            tool = tools[original]
            depends = [
                f"rcg-action-{rank_by_original[source] + 1}"
                for source in depends_by_original.get(original, [])
                if source in rank_by_original
            ]
            execution_wave = max(1, min(stage_by_original[original], max(predicted_wave_count, len(selected), 1)))
            actions.append(
                {
                    "id": f"rcg-action-{rank + 1}",
                    "title": str(tool.get("name") or f"Action {rank + 1}"),
                    "objective": (
                        f"Execute only the task slice matched to {tool.get('name')}: "
                        f"{tool.get('description', '')}"
                    ),
                    "whyDistinct": "Selected by Fascia-MoE admission, priority, dependency, and sparse expert routing.",
                    "suggestedChecks": [
                        "Resolve arguments from observed task evidence",
                        "Return real tool output or a grounded failure report",
                    ],
                    "risk": "Do not fabricate missing tool results; report unavailable or failed calls as local failures.",
                    "toolName": tool.get("name"),
                    "toolDescription": tool.get("description", ""),
                    "parameters": tool.get("parameters"),
                    "dependsOn": depends,
                    "admission": float(admission[original]),
                    "lexicalIntent": float(lexical[original]),
                    "priority": float(priority[original]),
                    "ready": not depends and execution_wave <= 1,
                    "executionWave": execution_wave,
                    "recommendedWaveSize": initial_wave_size,
                    "laterWaveSize": later_wave_size,
                    "predictedWaveCount": max(1, predicted_wave_count),
                    "planConfidence": float(admission[selected].mean()),
                    "stableKey": f"{tool.get('name', '')}:{str(tool.get('description') or '')[:160]}",
                }
            )
        return {
            "actions": actions,
            "modelStep": self.step,
            "modelVariant": self.version,
            "candidateCount": len(tools),
            "admittedCount": len(actions),
            "recommendedWaveSize": initial_wave_size,
            "laterWaveSize": later_wave_size,
            "predictedWaveCount": max(1, predicted_wave_count),
            "planConfidence": float(admission[selected].mean()),
            "mustReportFailure": must_report_failure >= 0.5,
            "taskSolvability": task_solvability,
            "takeoverProbability": takeover_probability,
            "haltProbability": halt_probability,
            "activeExperts": output["active_experts"].detach().cpu().tolist(),
            "latencyMs": (time.perf_counter() - started) * 1000,
        }

    def _evidence_from_drafts(
        self, actions: list[dict[str, Any]], drafts: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
        action_by_id = {str(item.get("id")): item for item in actions}
        ordered_actions = [action_by_id.get(str(draft.get("briefId")), {}) for draft in drafts]
        evidence_by_name: dict[str, dict[str, Any]] = {}
        flags = []
        failures = []
        for action, draft in zip(ordered_actions, drafts):
            name = self._tool_name(action)
            calls = draft.get("toolCalls") or []
            flag = self.execution_flags(action, draft)
            failure = "success" if flag["success"] else self.structural_failure_type(action, draft, flag)
            flags.append(flag)
            failures.append(failure)
            snippets = []
            for call in calls:
                status = "error" if call.get("isError") else "success"
                result = call.get("resultText") or call.get("resultPreview") or call.get("error") or draft.get("finalText", "")
                snippets.append(
                    f"status={status}; tool={call.get('toolName')}; args={_json(call.get('args') or {})}; result={_clip_text(result, 1600)}"
                )
            if not snippets:
                snippets.append(f"status={draft.get('status')}; result={_clip_text(draft.get('finalText', ''), 1600)}")
            evidence_by_name[name] = {
                "evidence": "\n".join(snippets),
                "runtime": {
                    "observed": True,
                    "success": flag["success"],
                    "failure_type": failure,
                    "latency_ratio": 0.0,
                    "retry_count": 0,
                },
            }
        return ordered_actions, evidence_by_name, flags, failures

    def select(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        actions = payload.get("actions") or []
        drafts = payload.get("drafts") or []
        if not drafts:
            return {"scores": [], "modelStep": self.step, "latencyMs": 0.0}
        ordered_actions, evidence_by_name, flags, failures = self._evidence_from_drafts(actions, drafts)
        row = self._row(payload, ordered_actions, evidence_by_name=evidence_by_name)
        output = self._forward(payload, row)
        coherence = output["coherence"].sigmoid().detach().cpu().tolist()
        utility = output["contribution"].sigmoid().detach().cpu().tolist()
        novelty = output["novelty"].sigmoid().detach().cpu().tolist()
        recovery = output["recovery_logits"].argmax(-1).detach().cpu().tolist()
        task_complete_probability = float(output["task_complete"].sigmoid().detach().cpu())
        task_solvability = float(output["task_solvable"].sigmoid().detach().cpu())
        must_report_probability = float(output["must_report_failure"].sigmoid().detach().cpu())
        takeover_probability = float(output["takeover"].sigmoid().detach().cpu())
        structural_failures = [flag["terminal_error"] for flag in flags]
        terminal_failure_evidence = {
            "timeout",
            "auth_or_quota",
            "server_or_network",
            "api_unavailable",
            "other_failure",
        }
        scores = []
        combined = []
        for index, draft in enumerate(drafts):
            adjusted_coherence = coherence[index]
            if structural_failures[index]:
                adjusted_coherence = min(adjusted_coherence, 0.35)
            score = 0.5 * adjusted_coherence + 0.25 * utility[index] + 0.25 * novelty[index]
            combined.append(score)
            scores.append(
                {
                    "briefId": str(draft.get("briefId")),
                    "coherence": adjusted_coherence,
                    "rawCoherence": coherence[index],
                    "utility": utility[index],
                    "novelty": novelty[index],
                    "retain": False,
                    "combined": score,
                    "mainScore": score,
                    "structuralValid": not structural_failures[index],
                    "toolMismatch": flags[index]["tool_mismatch"],
                    "terminalError": flags[index]["terminal_error"],
                    "failureSource": float(structural_failures[index]),
                    "failureType": failures[index],
                    "recoveryAction": RECOVERY_NAMES[recovery[index]],
                }
            )
        retained = [
            index
            for index, item in enumerate(scores)
            if item["structuralValid"]
            and (
                item["coherence"] >= 0.45
                or item["combined"] >= 0.60
                or item["utility"] >= 0.65
            )
        ]
        # A successful planned tool call is executable evidence, even when the
        # learned coherence head is conservative on long dependency chains.
        # Suppressing such nodes forces the main agent to replay the same chain.
        for index, flag in enumerate(flags):
            if flag["success"] and index not in retained:
                retained.append(index)
        # Terminal tool failures are not coherent successful reasoning paths, but
        # they are high-value memory: downstream turns must preserve them instead
        # of retrying blindly or fabricating a missing result.
        for index, failed in enumerate(structural_failures):
            if (
                failed
                and failures[index] in terminal_failure_evidence
                and any(
                    str(
                        call.get("resultText")
                        or call.get("resultPreview")
                        or call.get("error")
                        or ""
                    ).strip()
                    for call in (drafts[index].get("toolCalls") or [])
                )
                and index not in retained
            ):
                retained.append(index)
        if not retained:
            valid = [index for index, item in enumerate(scores) if item["structuralValid"]]
            if valid:
                retained = [max(valid, key=lambda index: combined[index])]
        retained = sorted(dict.fromkeys(retained))
        for index in retained:
            scores[index]["retain"] = True
        failure_reports = []
        for index, failed in enumerate(structural_failures):
            if not failed:
                continue
            action = ordered_actions[index]
            branch_id = str(drafts[index].get("briefId"))
            raw_error = " ".join(
                str(
                    call.get("resultText")
                    or call.get("resultPreview")
                    or call.get("error")
                    or ""
                )
                for call in (drafts[index].get("toolCalls") or [])
            ).strip()
            blocked = [
                str(candidate.get("id"))
                for candidate in actions
                if branch_id in (candidate.get("dependsOn") or [])
            ]
            failure_reports.append(
                {
                    "briefId": branch_id,
                    "toolName": self._tool_name(action),
                    "failureType": (
                        f"{failures[index]}: {_clip_text(raw_error, 180)}"
                        if raw_error
                        else failures[index]
                    ),
                    "recoveryAction": FAILURE_RECOVERY.get(failures[index], "report_failure"),
                    "status": drafts[index].get("status"),
                    "blockedBriefIds": blocked,
                    "message": (
                        f"Branch {branch_id} failed at {self._tool_name(action) or 'unknown tool'}: "
                        f"{failures[index]}"
                        + (f"; exact_error={_clip_text(raw_error, 600)}." if raw_error else ".")
                    ),
                }
            )
        failed_indices = [
            index for index, failed in enumerate(structural_failures) if failed
        ]
        retryable = bool(failed_indices) and all(
            failures[index] in {"timeout", "server_or_network", "invalid_call"}
            for index in failed_indices
        )
        nonretryable_failure = bool(failed_indices) and not retryable
        all_successful = bool(drafts) and all(not item for item in structural_failures)
        task_solvable = (
            all_successful
            or (task_solvability >= 0.5 and not nonretryable_failure)
        )
        task_complete = (
            task_complete_probability >= 0.5 and (bool(retained) or bool(failure_reports))
        ) or (all_successful and len(retained) == len(drafts))
        if failure_reports and retained:
            task_complete = task_complete or bool(retained)
        task_status = (
            "complete"
            if task_complete and (retained or not failure_reports)
            else "incomplete_retryable"
            if retryable
            else "unsolvable_current_plan"
        )
        main_candidates = [
            index for index in retained if scores[index]["mainScore"] >= 0.5
        ] or retained
        direct_takeover_ready = (
            task_complete
            and all_successful
            and len(main_candidates) == len(drafts)
            and min([scores[index]["mainScore"] for index in main_candidates], default=0.0) >= 0.45
        )
        main_decision = (
            "takeover"
            if (
                direct_takeover_ready
                or (task_complete and takeover_probability >= 0.5 and bool(main_candidates))
            )
            else "delegate"
        )
        return {
            "scores": scores,
            "taskSolvable": task_solvable,
            "taskComplete": task_complete,
            "taskStatus": task_status,
            "taskSolvability": task_solvability,
            "taskCompletion": task_complete_probability,
            "mainDecision": main_decision,
            "mainBriefIds": [str(drafts[index].get("briefId")) for index in main_candidates],
            "mainConfidence": min([scores[index]["mainScore"] for index in main_candidates], default=0.0),
            "takeoverProbability": takeover_probability,
            "mustReportFailure": (bool(failure_reports) or must_report_probability >= 0.5) and not all_successful,
            "failureReports": failure_reports,
            "modelStep": self.step,
            "modelVariant": self.version,
            "activeExperts": output["active_experts"].detach().cpu().tolist(),
            "latencyMs": (time.perf_counter() - started) * 1000,
        }

    def route(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        actions = payload.get("actions") or []
        drafts = payload.get("drafts") or []
        if not actions or not drafts:
            return {"routes": [], "modelStep": self.step, "latencyMs": 0.0}
        draft_by_id = {str(draft.get("briefId")): draft for draft in drafts}
        ordered_drafts = [
            draft_by_id.get(
                str(action.get("id")),
                {"briefId": action.get("id"), "status": "error", "finalText": "", "toolCalls": []},
            )
            for action in actions
        ]
        _, evidence_by_name, flags, failures = self._evidence_from_drafts(actions, ordered_drafts)
        row = self._row(payload, actions, evidence_by_name=evidence_by_name)
        output = self._forward(payload, row)
        dependency = output["dependency"].sigmoid().detach().cpu()
        residual = output["residual_route"].sigmoid().detach().cpu()
        coherence = output["coherence"].sigmoid().detach().cpu().tolist()
        recovery = output["recovery_logits"].argmax(-1).detach().cpu().tolist()
        terminal_failure_evidence = {
            "timeout",
            "auth_or_quota",
            "server_or_network",
            "api_unavailable",
            "other_failure",
        }
        reliable = []
        for index in range(len(actions)):
            has_text = any(
                str(
                    call.get("resultText")
                    or call.get("resultPreview")
                    or call.get("error")
                    or ""
                ).strip()
                for call in (ordered_drafts[index].get("toolCalls") or [])
            )
            reliable.append(
                (flags[index]["success"] and coherence[index] >= 0.5)
                or (
                    flags[index]["terminal_error"]
                    and failures[index] in terminal_failure_evidence
                    and has_text
                )
            )
        action_index = {str(action.get("id")): index for index, action in enumerate(actions)}
        routes = []
        for target, action in enumerate(actions):
            if flags[target]["success"]:
                continue
            explicit_sources = {
                action_index[source_id]
                for source_id in action.get("dependsOn") or []
                if source_id in action_index
            }
            learned_sources = {
                source
                for source in range(len(actions))
                if source != target
                and (float(dependency[target, source]) >= 0.55 or float(residual[target, source]) >= 0.55)
            }
            sources = [
                source for source in explicit_sources | learned_sources if reliable[source]
            ]
            sources.sort(
                key=lambda source: max(float(dependency[target, source]), float(residual[target, source])),
                reverse=True,
            )
            recovery_name = RECOVERY_NAMES[recovery[target]]
            retry_own = failures[target] == "invalid_call" and recovery_name in {"repair_call", "request_more_evidence"}
            continue_own = recovery_name in {"wait_dependency", "request_more_evidence"}
            if not sources and not retry_own and not continue_own:
                continue
            routes.append(
                {
                    "targetBriefId": str(action.get("id")),
                    "sourceBriefIds": [str(actions[source].get("id")) for source in sources],
                    "retryOwnErrors": retry_own,
                    "continueOwn": continue_own,
                    "ready": bool(sources or retry_own or continue_own),
                    "reason": (
                        "Fascia-MoE residual/dependency routing: pass only coherent peer evidence "
                        "or bounded own-error repair signals."
                    ),
                    "dependencyScores": {
                        str(actions[source].get("id")): float(dependency[target, source])
                        for source in sources
                    },
                    "residualScores": {
                        str(actions[source].get("id")): float(residual[target, source])
                        for source in sources
                    },
                    "coherence": coherence[target],
                    "recoveryAction": recovery_name,
                }
            )
        return {
            "routes": routes,
            "modelStep": self.step,
            "modelVariant": self.version,
            "activeExperts": output["active_experts"].detach().cpu().tolist(),
            "latencyMs": (time.perf_counter() - started) * 1000,
        }
