"""Prompt-only replacement for the trained RCG dynamic policy."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter


TERMINAL_FAILURES = {
    "timeout": ("timeout", "abandon_tool"),
    "timed out": ("timeout", "abandon_tool"),
    "401": ("auth_or_quota", "abandon_tool"),
    "unauthorized": ("auth_or_quota", "abandon_tool"),
    "403": ("auth_or_quota", "abandon_tool"),
    "429": ("auth_or_quota", "abandon_tool"),
    "503": ("server_or_network", "abandon_tool"),
    "service unavailable": ("server_or_network", "abandon_tool"),
    "connection": ("server_or_network", "abandon_tool"),
    "invalid": ("invalid_call", "repair_call"),
}


def clamp(value, default=0.0):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def slug(value):
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text[:48] or "action"


def compact(value, limit=12000):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text[:limit]


def extract_json(text):
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("model response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:index + 1])
    raise ValueError("model response contains incomplete JSON")


def classify_failure(draft):
    calls = draft.get("toolCalls") or []
    errors = [call for call in calls if call.get("isError")]
    if not errors:
        return "success", "none"
    text = " ".join(
        str(call.get("resultText") or call.get("resultPreview") or "")
        for call in errors
    ).lower()
    for marker, result in TERMINAL_FAILURES.items():
        if marker in text:
            return result
    return "other_failure", "report_failure"


class PromptHarnessPolicy:
    """Use the base LLM as an API planner/judge with code-level validation."""

    def __init__(self, generate_batch, log=lambda _message: None):
        self.generate_batch = generate_batch
        self.log = log
        self.stats = {
            "calls": 0,
            "latencyMs": 0,
            "promptTokens": 0,
            "outputTokens": 0,
            "parseFailures": 0,
            "byCommand": {},
        }

    def _ask(self, command, system, payload, max_tokens):
        request = {
            "context": {
                "systemPrompt": system,
                "messages": [{
                    "role": "user",
                    "content": compact(payload),
                }],
                "tools": [],
            },
            "options": {
                "reasoning": "off",
                "temperature": 0.0,
                "maxTokens": max_tokens,
                "sessionId": f"api-harness-policy-{command}",
            },
        }
        started = time.perf_counter()
        result = self.generate_batch([request])[0]
        latency_ms = round((time.perf_counter() - started) * 1000)
        self.stats["calls"] += 1
        self.stats["latencyMs"] += latency_ms
        self.stats["promptTokens"] += int(result.get("promptTokens") or 0)
        self.stats["outputTokens"] += int(result.get("outputTokens") or 0)
        by_command = self.stats["byCommand"].setdefault(
            command,
            {"calls": 0, "latencyMs": 0, "promptTokens": 0, "outputTokens": 0},
        )
        by_command["calls"] += 1
        by_command["latencyMs"] += latency_ms
        by_command["promptTokens"] += int(result.get("promptTokens") or 0)
        by_command["outputTokens"] += int(result.get("outputTokens") or 0)
        try:
            parsed = extract_json(result.get("text"))
        except Exception:
            self.stats["parseFailures"] += 1
            self.log(
                "prompt policy parse failure "
                + compact({"command": command, "text": result.get("text")}, 1000)
            )
            raise
        self.log(
            "prompt policy "
            + compact({
                "command": command,
                "latencyMs": latency_ms,
                "promptTokens": result.get("promptTokens"),
                "outputTokens": result.get("outputTokens"),
            })
        )
        if os.environ.get("RCG_PROMPT_POLICY_DEBUG") == "1":
            self.log(
                "prompt policy response "
                + compact({"command": command, "value": parsed}, 6000)
            )
        return parsed

    def plan(self, payload):
        tools = payload.get("tools") or []
        tool_by_name = {str(tool.get("name")): tool for tool in tools}
        prompt_payload = {
            "task": payload.get("task"),
            "latestObservation": payload.get("latestObservation"),
            "maxBranches": payload.get("maxBranches"),
            "tools": [{
                "name": tool.get("name"),
                "description": tool.get("description"),
                "required": (
                    (tool.get("parameters") or {}).get("required") or []
                    if isinstance(tool.get("parameters"), dict)
                    else []
                ),
            } for tool in tools],
            "residualMemory": payload.get("residualMemory"),
            "globalState": payload.get("globalState"),
        }
        system = """You are the planning API for a speculative agent swarm.
Return compact JSON only:
{"actions":[{"id":"a","tool":"tool_name","goal":"short local goal",
"deps":[],"wave":1}]}.
Select only tools required by the task. Split repeated calls into distinct
actions. Encode real serial dependencies by action id. Independent actions use
wave 1. Never invent a tool. Include a fallback only when the task explicitly
requests recovery. Do not solve the task; construct the smallest complete DAG."""
        try:
            response = self._ask("plan", system, prompt_payload, 256)
            raw_actions = response.get("actions") or []
        except Exception:
            raw_actions = []
        actions = []
        aliases = {}
        for index, raw in enumerate(raw_actions):
            if not isinstance(raw, dict):
                continue
            tool_name = str(raw.get("toolName") or raw.get("tool") or "")
            tool = tool_by_name.get(tool_name)
            if not tool:
                continue
            action_id = str(raw.get("id") or f"api-{index + 1}-{slug(tool_name)}")
            if action_id in aliases.values():
                action_id = f"{action_id}-{index + 1}"
            aliases[str(raw.get("id") or action_id)] = action_id
            objective = str(
                raw.get("objective") or raw.get("goal") or ""
            ).strip()
            if not objective:
                objective = (
                    f"Use {tool_name} only for the matching local part of the task. "
                    "Resolve exact arguments from the user request or routed evidence."
                )
            actions.append({
                "id": action_id,
                "title": tool_name,
                "objective": objective,
                "whyDistinct": str(
                    raw.get("whyDistinct")
                    or "Prompt planner assigned one distinct tool-local objective."
                ),
                "suggestedChecks": [
                    "Use exact task entities and parameters",
                    "Return the real tool result or precise failure",
                ],
                "risk": "Prompt planning is validated after execution.",
                "toolName": tool_name,
                "toolDescription": tool.get("description"),
                "parameters": tool.get("parameters"),
                "dependsOn": list(
                    raw.get("dependsOn") or raw.get("deps") or []
                ),
                "stableKey": f"{tool_name}:{objective}",
                "priority": clamp(raw.get("priority"), 0.7),
                "admission": clamp(raw.get("priority"), 0.7),
                "ready": bool(raw.get("ready", True)),
                "executionWave": max(
                    1, int(raw.get("executionWave") or raw.get("wave") or 1)
                ),
                "recommendedWaveSize": max(
                    1, int(raw.get("recommendedWaveSize") or len(raw_actions) or 1)
                ),
                "laterWaveSize": max(1, int(raw.get("laterWaveSize") or 1)),
                "predictedWaveCount": max(
                    1, int(raw.get("predictedWaveCount") or 1)
                ),
                "planConfidence": clamp(raw.get("planConfidence"), 0.7),
            })
        known_ids = {action["id"] for action in actions}
        for action in actions:
            action["dependsOn"] = [
                aliases.get(str(value), str(value))
                for value in action["dependsOn"]
                if aliases.get(str(value), str(value)) in known_ids
            ]
            if action["dependsOn"]:
                action["ready"] = False
                action["executionWave"] = max(action["executionWave"], 2)
        if actions:
            wave_one = sum(not action["dependsOn"] for action in actions)
            for action in actions:
                action["recommendedWaveSize"] = max(1, wave_one)
                action["predictedWaveCount"] = max(
                    item["executionWave"] for item in actions
                )
            return {
                "actions": actions[
                    : int(payload.get("maxBranches") or len(actions))
                ]
            }
        return {"actions": self._fallback_plan(payload)}

    def _fallback_plan(self, payload):
        task = str(payload.get("task") or "").lower()
        scored = []
        for tool in payload.get("tools") or []:
            name = str(tool.get("name") or "")
            terms = set(re.findall(r"[a-z0-9]+", name.lower().replace("_", " ")))
            description = str(tool.get("description") or "").lower()
            score = sum(term in task for term in terms if len(term) > 2)
            score += sum(
                token in task
                for token in re.findall(r"[a-z0-9]+", description)
                if len(token) > 5
            ) * 0.05
            if score > 0:
                scored.append((score, tool))
        scored.sort(key=lambda item: -item[0])
        actions = []
        for index, (_score, tool) in enumerate(
            scored[: int(payload.get("maxBranches") or 4)]
        ):
            name = str(tool.get("name"))
            actions.append({
                "id": f"api-fallback-{index + 1}-{slug(name)}",
                "title": name,
                "objective": f"Execute the task-local portion requiring {name}.",
                "whyDistinct": "Lexical fallback after malformed planner JSON.",
                "toolName": name,
                "toolDescription": tool.get("description"),
                "parameters": tool.get("parameters"),
                "dependsOn": [],
                "stableKey": f"{name}:fallback",
                "priority": 0.5,
                "admission": 0.5,
                "ready": True,
                "executionWave": 1,
                "recommendedWaveSize": max(1, len(scored)),
                "laterWaveSize": 1,
                "predictedWaveCount": 1,
                "planConfidence": 0.25,
            })
        return actions

    def select(self, payload):
        drafts = payload.get("drafts") or []
        # Tool execution gives stronger local labels than another generative
        # judgment call: a non-empty successful result is coherent, while a
        # non-empty terminal error must be retained as failure evidence.
        response = {}
        model_scores = {
            str(item.get("briefId")): item
            for item in response.get("scores") or []
            if isinstance(item, dict)
        }
        scores = []
        failure_reports = []
        successful = set()
        accounted = set()
        signatures = Counter()
        for draft in drafts:
            brief_id = str(draft.get("briefId") or "")
            calls = draft.get("toolCalls") or []
            signature = tuple(sorted(
                (str(call.get("toolName")), compact(call.get("args") or {}, 500))
                for call in calls
            ))
            signatures[signature] += 1
            failure_type, recovery = classify_failure(draft)
            has_evidence = any(
                str(call.get("resultText") or call.get("resultPreview") or "").strip()
                for call in calls
            )
            has_success = any(not call.get("isError") and has_evidence for call in calls)
            has_error = any(call.get("isError") and has_evidence for call in calls)
            model = model_scores.get(brief_id, {})
            retain = bool(has_success or has_error or model.get("retain"))
            coherence = clamp(
                model.get("coherence"),
                1.0 if has_success and not has_error else 0.15 if has_error else 0.0,
            )
            utility = clamp(model.get("utility"), 0.9 if has_evidence else 0.0)
            novelty = clamp(
                model.get("novelty"),
                1.0 if signatures[signature] == 1 else 0.25,
            )
            scores.append({
                "briefId": brief_id,
                "coherence": coherence,
                "utility": utility,
                "novelty": novelty,
                "retain": retain,
                "failureSource": 1.0 if has_error else 0.0,
                "failureType": failure_type,
            })
            if has_success:
                successful.add(brief_id)
                accounted.add(brief_id)
            if has_error:
                accounted.add(brief_id)
                call = next(call for call in calls if call.get("isError"))
                failure_reports.append({
                    "briefId": brief_id,
                    "toolName": call.get("toolName"),
                    "failureType": failure_type,
                    "recoveryAction": recovery,
                    "status": "terminal_error",
                    "message": str(
                        call.get("resultText") or call.get("resultPreview") or ""
                    ),
                })
        action_ids = {
            str(action.get("id"))
            for action in payload.get("actions") or []
            if action.get("id")
        }
        structurally_accounted = bool(action_ids) and action_ids <= accounted
        task_complete = structurally_accounted
        main_ids = [
            str(value) for value in response.get("mainBriefIds") or []
            if str(value) in accounted
        ]
        if task_complete:
            main_ids = [score["briefId"] for score in scores if score["retain"]]
        return {
            "scores": scores,
            "nextActions": response.get("nextActions"),
            "taskSolvable": not failure_reports,
            "taskComplete": task_complete,
            "taskSolvability": clamp(
                1.0 if not failure_reports else 0.2,
            ),
            "taskStatus": (
                "complete"
                if task_complete
                else str(response.get("taskStatus") or "incomplete_retryable")
            ),
            "mainDecision": (
                "takeover"
                if task_complete
                else "delegate"
            ),
            "mainBriefIds": main_ids,
            "mainConfidence": clamp(
                response.get("mainConfidence"),
                1.0 if task_complete else 0.4,
            ),
            "mustReportFailure": bool(
                response.get("mustReportFailure", failure_reports)
            ),
            "failureReports": failure_reports or response.get("failureReports") or [],
        }

    def route(self, payload):
        actions = payload.get("actions") or []
        drafts = payload.get("drafts") or []
        action_by_id = {
            str(action.get("id")): action
            for action in actions
            if action.get("id")
        }
        successful_sources = {
            str(draft.get("briefId"))
            for draft in drafts
            if any(
                not call.get("isError")
                and str(
                    call.get("resultText") or call.get("resultPreview") or ""
                ).strip()
                for call in draft.get("toolCalls") or []
            )
        }
        routes = []
        for target_id, action in action_by_id.items():
            dependencies = [
                str(value) for value in action.get("dependsOn") or []
                if str(value) in successful_sources
            ]
            sources = list(dict.fromkeys(dependencies))
            routes.append({
                "targetBriefId": target_id,
                "sourceBriefIds": sources,
                "retryOwnErrors": False,
                "continueOwn": False,
                "ready": bool(sources or not action.get("dependsOn")),
                "reason": "Route verified evidence for declared dependencies.",
            })
        return {"routes": routes}
