"""Build failure-aware RCG training rows from live TRAJECT-Bench replay data."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


FAILURE_TYPES = (
    "success",
    "timeout",
    "auth_or_quota",
    "server_or_network",
    "api_unavailable",
    "invalid_call",
    "empty_result",
    "provenance_mismatch",
    "blocked_dependency",
    "other_failure",
)
FAILURE_TYPE_TO_ID = {name: index for index, name in enumerate(FAILURE_TYPES)}
STATUS_TO_FAILURE_TYPE = {
    "live_valid": "success",
    "timeout": "timeout",
    "auth_or_quota": "auth_or_quota",
    "rate_limited": "auth_or_quota",
    "server_error": "server_or_network",
    "network_error": "server_or_network",
    "invalid_response": "server_or_network",
    "api_missing": "api_unavailable",
    "api_not_found": "api_unavailable",
    "service_suspended": "api_unavailable",
    "unresolved_metadata": "api_unavailable",
    "unsafe_skipped": "api_unavailable",
    "call_invalid": "invalid_call",
    "live_empty": "empty_result",
    "provenance_mismatch": "provenance_mismatch",
    "api_error": "other_failure",
    "not_replayed": "other_failure",
}
RETRYABLE_FAILURE_TYPES = {"invalid_call"}
CONTINUABLE_FAILURE_TYPES = {
    "invalid_call",
    "blocked_dependency",
}


def recovery_action(kind: str) -> str:
    return {
        "timeout": "abandon_tool_for_this_task_and_report_timeout",
        "server_or_network": "abandon_tool_for_this_task_and_report_service_failure",
        "auth_or_quota": "abandon_tool_for_this_task_and_report_auth_failure",
        "api_unavailable": "abandon_tool_and_replan_with_a_distinct_available_tool_or_report",
        "invalid_call": "repair_parameters_before_one_bounded_retry",
        "empty_result": "abandon_tool_for_this_task_and_report_empty_result",
        "provenance_mismatch": "discard_result_and_replan_without_reusing_the_tool_result",
        "blocked_dependency": "wait_for_or_repair_upstream",
        "other_failure": "stop_and_report_failure",
        "success": "none",
    }[kind]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def failure_type(status: str) -> str:
    return STATUS_TO_FAILURE_TYPE.get(status, "other_failure")


def direct_status(branch: dict[str, Any]) -> str:
    replay = branch.get("live_replay") or {}
    return str(replay.get("status") or "not_replayed")


def dependency_sources(branch: dict[str, Any]) -> list[str]:
    return [
        str(edge.get("source_branch_id"))
        for edge in branch.get("depends_on", []) or []
        if edge.get("source_branch_id")
    ]


def annotate_task(row: dict[str, Any]) -> dict[str, Any]:
    branches = row.get("branches", [])
    by_id = {str(branch["branch_id"]): branch for branch in branches}
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {branch_id: 0 for branch_id in by_id}
    for branch_id, branch in by_id.items():
        for source_id in dependency_sources(branch):
            if source_id not in by_id:
                continue
            children[source_id].append(branch_id)
            indegree[branch_id] += 1

    queue = deque(sorted(branch_id for branch_id, degree in indegree.items() if degree == 0))
    ordered = []
    while queue:
        branch_id = queue.popleft()
        ordered.append(branch_id)
        for child in children[branch_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    ordered.extend(branch_id for branch_id in by_id if branch_id not in ordered)

    effective_success: dict[str, bool] = {}
    root_failures = []
    blocked = []
    for branch_id in ordered:
        branch = by_id[branch_id]
        status = direct_status(branch)
        direct_success = status == "live_valid"
        failed_sources = [
            source_id
            for source_id in dependency_sources(branch)
            if source_id in effective_success and not effective_success[source_id]
        ]
        is_blocked = direct_success and bool(failed_sources)
        effective = direct_success and not is_blocked
        effective_success[branch_id] = effective

        kind = "blocked_dependency" if is_blocked else failure_type(status)
        branch["execution_coherence_target"] = float(direct_success)
        branch["effective_coherence_target"] = float(effective)
        branch["failure_type"] = kind
        branch["failure_type_id"] = FAILURE_TYPE_TO_ID[kind]
        branch["failure_source_target"] = float(not direct_success)
        branch["blocked_by"] = failed_sources
        branch["execution_outcome_source"] = "trajectory_live_api_replay_20260613"
        branch["live_replay"] = {
            **(branch.get("live_replay") or {}),
            "coherence_label": int(direct_success),
            "trainable": True,
        }

        if not direct_success:
            root_failures.append(
                {
                    "branch_id": branch_id,
                    "tool_name": branch.get("tool_name", ""),
                    "status": status,
                    "failure_type": kind,
                    "recovery_action": recovery_action(kind),
                    "local_objective": branch.get("tool_description", ""),
                    "blocked_descendants": [],
                }
            )
        elif is_blocked:
            blocked.append(
                {
                    "branch_id": branch_id,
                    "tool_name": branch.get("tool_name", ""),
                    "failure_type": kind,
                    "recovery_action": recovery_action(kind),
                    "blocked_by": failed_sources,
                }
            )

    failure_by_id = {item["branch_id"]: item for item in root_failures}
    for item in blocked:
        frontier = list(item["blocked_by"])
        seen = set()
        while frontier:
            source_id = frontier.pop()
            if source_id in seen:
                continue
            seen.add(source_id)
            if source_id in failure_by_id:
                failure_by_id[source_id]["blocked_descendants"].append(item["branch_id"])
            elif source_id in by_id:
                frontier.extend(dependency_sources(by_id[source_id]))

    task_solvable = all(effective_success.values()) if branches else False
    retryable = bool(root_failures) and all(
        failure["failure_type"] in RETRYABLE_FAILURE_TYPES
        for failure in root_failures
    )
    task_status = (
        "complete"
        if task_solvable
        else "incomplete_retryable"
        if retryable
        else "unsolvable_current_plan"
    )
    terminal_branches = [
        branch_id for branch_id in ordered
        if not children.get(branch_id)
    ]
    main_candidates = []
    if task_solvable:
        # A single DAG rule covers sequential, parallel, and mixed tasks:
        # every verified leaf contributes to the final response. A pure chain
        # has one leaf, while independent or mixed work has several.
        main_candidates = [
            branch_id for branch_id in terminal_branches
            if effective_success.get(branch_id)
        ]
    for branch_id, branch in by_id.items():
        branch["admission_target"] = 1.0
        branch["main_candidate_target"] = float(
            branch_id in main_candidates
        )
        branch["execution_cost_target"] = 1.0
        branch["initial_ready_target"] = float(
            not dependency_sources(branch)
        )
        branch["continue_after_success_target"] = 0.0
        branch["continuation_target"] = float(
            branch["failure_type"] in CONTINUABLE_FAILURE_TYPES
        )

    root_count = sum(
        not dependency_sources(branch) for branch in branches
    )
    row["branch_count_target"] = len(branches)
    row["recommended_wave_size"] = max(
        1, min(4, root_count if root_count else len(branches))
    )
    row["main_assembly_mode"] = (
        "direct_branch"
        if len(main_candidates) == 1
        else "coherent_set_takeover"
        if main_candidates
        else "coherent_set"
        if task_solvable
        else "failure_report"
    )
    row["takeover_target"] = float(
        task_solvable and bool(main_candidates)
    )
    row["task_solvability_target"] = float(task_solvable)
    row["task_outcome"] = {
        "status": task_status,
        "solvable": task_solvable,
        "retryable": retryable,
        "root_failures": root_failures,
        "blocked_branches": blocked,
        "successful_branches": [
            branch_id for branch_id, success in effective_success.items() if success
        ],
        "required_response_behavior": (
            "synthesize_verified_results"
            if task_solvable
            else "retry_or_report_incomplete_task_without_fabrication"
            if retryable
            else "report_unsolvable_current_plan_and_failed_steps_without_fabrication"
        ),
    }
    row["failure_report_target"] = render_failure_report(row)
    return row


def render_failure_report(row: dict[str, Any]) -> str:
    outcome = row["task_outcome"]
    if outcome["solvable"]:
        return (
            "TASK_STATUS: SOLVABLE\n"
            "Use only verified branch outputs when synthesizing the answer."
        )
    lines = [
        (
            "TASK_STATUS: INCOMPLETE_RETRYABLE"
            if outcome["retryable"]
            else "TASK_STATUS: UNSOLVABLE_CURRENT_PLAN"
        ),
        "The requested task could not be completed with verified evidence.",
        "FAILED_STEPS:",
    ]
    for failure in outcome["root_failures"]:
        lines.append(
            f"- {failure['branch_id']} | {failure['tool_name']} | "
            f"{failure['failure_type']} ({failure['status']}) | "
            f"recovery={failure['recovery_action']}"
        )
    if outcome["blocked_branches"]:
        lines.append("BLOCKED_STEPS:")
        for blocked in outcome["blocked_branches"]:
            lines.append(
                f"- {blocked['branch_id']} | {blocked['tool_name']} | "
                f"blocked_by={','.join(blocked['blocked_by'])} | "
                f"recovery={blocked['recovery_action']}"
            )
    successful = outcome["successful_branches"]
    lines.extend(
        [
            f"VERIFIED_PARTIAL_BRANCHES: {','.join(successful) if successful else 'none'}",
            "Do not invent missing tool outputs or present a complete answer before recovery succeeds.",
            "State the failed component and distinguish verified partial results from missing results.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    branch_statuses = Counter()
    failure_types = Counter()
    task_statuses = Counter()
    blocked_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = annotate_task(json.loads(line))
            task_statuses[row["task_outcome"]["status"]] += 1
            for branch in row["branches"]:
                branch_statuses[direct_status(branch)] += 1
                failure_types[branch["failure_type"]] += 1
                blocked_count += int(branch["failure_type"] == "blocked_dependency")
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, args.output)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "task_status_counts": dict(task_statuses),
        "branch_status_counts": dict(branch_statuses),
        "failure_type_counts": dict(failure_types),
        "blocked_dependency_branches": blocked_count,
        "failure_types": list(FAILURE_TYPES),
        "policy": {
            "coherence_one": "live_valid and all upstream dependencies completed",
            "coherence_zero": "every direct execution failure and every dependency-blocked branch",
            "task_solvable": "all required branches have effective_coherence_target=1",
            "direct_takeover": "a verified action DAG with one successful leaf bypasses redundant main-agent reconstruction",
            "coherent_set": "all successful leaves of the action DAG are retained and deterministically assembled, covering parallel, sequential, and mixed tasks without a mode switch",
            "turn_compression": "a successful local tool branch terminates after one model stage; only unresolved dependencies or repairable failures receive another stage",
            "incomplete_response": "tool or service failures receive low coherence and are abandoned for the current task; only a relevant invalid call gets one bounded parameter repair; never fabricate missing outputs",
        },
    }
    atomic_write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
