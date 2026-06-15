"""Canonical text serialization shared by OPD training and runtime."""

from __future__ import annotations

import json


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def bounded_text(value, limit: int = 12000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    head = int(limit * 0.75)
    tail = limit - head
    return (
        text[:head]
        + "\n<TRUNCATED_MIDDLE_PRESERVE_TAIL>\n"
        + text[-tail:]
    )


def action_text(action: dict) -> str:
    return (
        "<TOOL_ACTION>\n"
        f"TOOL: {action.get('tool_name', action.get('name', ''))}\n"
        f"DESCRIPTION: {action.get('description', '')}\n"
        f"PARAMETERS: {json_text(action.get('parameters'))}\n"
        "</TOOL_ACTION>"
    )


def query_text(query: str, runtime_state: dict | None = None) -> str:
    text = query
    if runtime_state:
        text += f"\n<RUNTIME_STATE>{json_text(runtime_state)}</RUNTIME_STATE>"
    return f"<TASK_QUERY>\n{text}\n</TASK_QUERY>"


def action_node_text(query: str, action: dict) -> str:
    return (
        "<ACTION_NODE>\n"
        f"GLOBAL_QUERY: {query}\n"
        f"TOOL: {action.get('tool_name', action.get('toolName', ''))}\n"
        f"LOCAL_OBJECTIVE: {action.get('description', action.get('objective', ''))}\n"
        f"PARAMETERS: {json_text(action.get('parameters'))}\n"
        f"DEPENDS_ON: {json_text(action.get('teacher', {}).get('depends_on_tools', action.get('dependsOn', [])))}\n"
        "</ACTION_NODE>"
    )


def execution_text(query: str, action: dict) -> str:
    execution = action.get("execution") or {}
    calls = execution.get("calls") or []
    results = [
        bounded_text(call.get("result", call.get("resultText", "")))
        for call in calls
    ]
    # The query and local objective are encoded separately. Repeating them here
    # pushes the actual tool evidence past the truncation boundary on long tasks.
    return (
        "<ACTION_EXECUTION>\n"
        f"EXPECTED_TOOL: {action.get('tool_name', action.get('toolName', ''))}\n"
        f"ACTUAL_TOOL: {json_text([call.get('tool_name', call.get('toolName', action.get('tool_name'))) for call in calls])}\n"
        f"STATUS: {json_text([call.get('status') for call in calls])}\n"
        f"ACTUAL_PARAMETERS: {json_text([call.get('arguments', call.get('args', {})) for call in calls])}\n"
        f"RESULT: {json_text(results)}\n"
        f"OBSERVED: {bool(execution.get('observed'))}\n"
        "</ACTION_EXECUTION>"
    )
