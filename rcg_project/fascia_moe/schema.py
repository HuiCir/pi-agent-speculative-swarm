from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA = "fascia_event_v1"
EXPERT_NAMES = (
    "planning",
    "scheduling",
    "diversity",
    "evidence",
    "recovery",
    "memory",
    "halting",
    "integration",
)
RECOVERY_NAMES = (
    "none",
    "repair_call",
    "wait_dependency",
    "abandon_action",
    "replan_distinct_action",
    "report_failure",
    "request_more_evidence",
)


@dataclass(frozen=True)
class ValidationSummary:
    task_id: str
    candidate_count: int
    observed_count: int
    selected_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_event(row: dict[str, Any]) -> ValidationSummary:
    _require(row.get("schema") == SCHEMA, "unexpected schema")
    task_id = str(row.get("task_id") or "")
    _require(bool(task_id), "task_id is required")
    _require(bool(str(row.get("query") or "")), f"{task_id}: empty query")
    candidates = row.get("candidates")
    _require(isinstance(candidates, list), f"{task_id}: candidates must be a list")
    _require(1 <= len(candidates) <= 64, f"{task_id}: invalid candidate count")
    names = [str(item.get("name") or "") for item in candidates]
    _require(all(names), f"{task_id}: candidate name is required")
    _require(len(names) == len(set(names)), f"{task_id}: duplicate candidate names")

    labels = row.get("labels") or {}
    node = labels.get("node") or {}
    for key in (
        "admission",
        "priority",
        "stage",
        "coherence",
        "contribution",
        "novelty",
        "memory_keep",
        "recovery",
    ):
        values = node.get(key)
        _require(
            isinstance(values, list) and len(values) == len(candidates),
            f"{task_id}: node label {key} has wrong length",
        )
    pair = labels.get("pair") or {}
    for key in ("dependency", "residual_route"):
        matrix = pair.get(key)
        _require(
            isinstance(matrix, list) and len(matrix) == len(candidates),
            f"{task_id}: pair label {key} has wrong row count",
        )
        _require(
            all(isinstance(line, list) and len(line) == len(candidates) for line in matrix),
            f"{task_id}: pair label {key} is not square",
        )
    global_labels = labels.get("global") or {}
    for key in (
        "branch_count",
        "initial_wave_size",
        "later_wave_size",
        "wave_count",
        "task_complete",
        "task_solvable",
        "must_report_failure",
        "takeover",
        "halt",
        "remaining_turns",
    ):
        _require(key in global_labels, f"{task_id}: missing global label {key}")
    expert_target = labels.get("expert_target")
    _require(
        isinstance(expert_target, list) and len(expert_target) == len(EXPERT_NAMES),
        f"{task_id}: expert_target has wrong length",
    )

    observed = sum(bool(item.get("evidence")) for item in candidates)
    selected = sum(float(value) >= 0.5 for value in node["admission"])
    return ValidationSummary(task_id, len(candidates), observed, selected)


def empty_square(size: int, value: float = 0.0) -> list[list[float]]:
    return [[value for _ in range(size)] for _ in range(size)]

