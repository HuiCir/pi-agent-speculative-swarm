#!/usr/bin/env python3
"""Build trajectory-level OPD data for the RCG MoE controller.

The V11 data describes only gold branches. OPD V1 instead presents the
controller with the same decision surface it sees in the harness: a complete
candidate tool set, a teacher-selected subset, an execution DAG, observed
outcomes, and a truthful final response.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from run_v10_real_swarm_benchmark import TASKS


BENCHMARK_TOOLS = {
    "resolve_customer": {
        "description": (
            "Resolve a customer name and return customer_id. Produces "
            "customer_id required by fetch_orders."
        ),
        "parameters": {"customer": "string"},
    },
    "fetch_orders": {
        "description": (
            "Fetch orders for a customer. Requires customer_id produced by "
            "resolve_customer and returns order_ids."
        ),
        "parameters": {"customer_id": "string"},
    },
    "fetch_order_total": {
        "description": (
            "Calculate an order total. Requires order_ids produced by "
            "fetch_orders and returns order_total required by check_budget."
        ),
        "parameters": {"order_ids": "string"},
    },
    "check_budget": {
        "description": (
            "Check a budget. Requires order_total produced by "
            "fetch_order_total and a limit."
        ),
        "parameters": {"order_total": "number", "limit": "number"},
    },
    "get_weather": {
        "description": (
            "Get current weather independently for a city and return "
            "weather_condition."
        ),
        "parameters": {"city": "string"},
    },
    "search_flights": {
        "description": (
            "Search flights independently using origin, destination, and "
            "date. Returns flight_id."
        ),
        "parameters": {
            "origin": "string",
            "destination": "string",
            "date": "string",
        },
    },
    "get_fx_rate": {
        "description": (
            "Get an independent currency exchange rate for base and quote. "
            "Returns fx_rate."
        ),
        "parameters": {"base": "string", "quote": "string"},
    },
    "geocode": {
        "description": (
            "Geocode a place and return latitude and longitude required by "
            "route_time. May be called for multiple places."
        ),
        "parameters": {"place": "string"},
    },
    "route_time": {
        "description": (
            "Calculate route time. Requires coordinates produced by geocode."
        ),
        "parameters": {
            "origin_lat": "number",
            "origin_lon": "number",
            "destination_lat": "number",
            "destination_lon": "number",
        },
    },
    "review_summary": {
        "description": (
            "Summarize product reviews independently and return "
            "review_sentiment."
        ),
        "parameters": {"product": "string"},
    },
    "inventory_status": {
        "description": (
            "Check inventory for a SKU. Provider failures must be reported."
        ),
        "parameters": {"sku": "string"},
    },
    "inventory_fallback": {
        "description": (
            "Fallback inventory provider. Requires failure evidence from "
            "inventory_status before use."
        ),
        "parameters": {"sku": "string"},
    },
    "billing_status": {
        "description": (
            "Check billing status. Authentication failures must be reported "
            "rather than fabricated."
        ),
        "parameters": {"account_id": "string"},
    },
}

BENCHMARK_DEPENDENCIES = {
    "fetch_orders": ["resolve_customer"],
    "fetch_order_total": ["fetch_orders"],
    "check_budget": ["fetch_order_total"],
    "route_time": ["geocode"],
    "inventory_fallback": ["inventory_status"],
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parameter_schema(branch: dict) -> dict:
    properties = {}
    required = []
    for item in branch.get("required_parameters") or []:
        name = str(item.get("name") or "")
        if not name:
            continue
        properties[name] = {"type": type_name(item.get("value"))}
        required.append(name)
    for item in branch.get("optional_parameters") or []:
        name = str(item.get("name") or "")
        if name and name not in properties:
            properties[name] = {"type": type_name(item.get("value"))}
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def type_name(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def runtime_action(tool_name: str, description: str, parameters: dict) -> dict:
    return {
        "tool_name": tool_name,
        "description": description,
        "parameters": parameters,
    }


def source_action(branch: dict) -> dict:
    return runtime_action(
        str(branch["tool_name"]),
        str(branch.get("tool_description") or ""),
        parameter_schema(branch),
    )


def tool_key(action: dict) -> str:
    return str(action["tool_name"])


def deduplicate_actions(actions: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for action in actions:
        key = tool_key(action)
        if key in seen:
            continue
        seen.add(key)
        output.append(action)
    return output


def branch_dependencies(row: dict, selected_names: set[str]) -> dict[str, list[str]]:
    branch_to_tool = {
        str(branch["branch_id"]): str(branch["tool_name"])
        for branch in row.get("branches") or []
    }
    dependencies = defaultdict(set)
    for branch in row.get("branches") or []:
        target = str(branch["tool_name"])
        if target not in selected_names:
            continue
        for edge in branch.get("depends_on") or []:
            source = branch_to_tool.get(str(edge.get("source_branch_id") or ""))
            if source and source in selected_names and source != target:
                dependencies[target].add(source)
    return {name: sorted(values) for name, values in dependencies.items()}


def topological_stages(
    selected_names: list[str],
    dependencies: dict[str, list[str]],
) -> dict[str, int]:
    stages = {}
    remaining = set(selected_names)
    for _ in range(len(selected_names) + 1):
        progressed = False
        for name in list(remaining):
            parents = [
                source
                for source in dependencies.get(name, [])
                if source in selected_names
            ]
            if all(source in stages for source in parents):
                stages[name] = (
                    1 + max((stages[source] for source in parents), default=0)
                )
                remaining.remove(name)
                progressed = True
        if not progressed:
            break
    for name in remaining:
        stages[name] = 1
    return stages


def wave_targets(stages: dict[str, int], maximum: int) -> tuple[int, int, int]:
    counts = Counter(stages.values())
    initial = min(maximum, counts.get(1, 1))
    later = min(
        maximum,
        max((count for stage, count in counts.items() if stage > 1), default=1),
    )
    return initial, later, max(stages.values(), default=1)


def source_execution(branches: list[dict]) -> dict:
    records = []
    for branch in branches:
        replay = branch.get("live_replay") or {}
        records.append(
            {
                "status": str(branch.get("failure_type") or "other_failure"),
                "success": bool(
                    float(branch.get("effective_coherence_target", 0.0))
                ),
                "arguments": {
                    str(item.get("name")): item.get("value")
                    for item in (
                        (branch.get("required_parameters") or [])
                        + (branch.get("optional_parameters") or [])
                    )
                    if item.get("name")
                },
                "result": str(
                    replay.get("live_output_preview")
                    or branch.get("executed_output")
                    or ""
                )[:3000],
                "failure_type": str(
                    branch.get("failure_type") or "other_failure"
                ),
            }
        )
    return {
        "calls": records,
        "observed": True,
        "coherent": bool(
            records and all(record["success"] for record in records)
        ),
    }


def make_candidate(
    action: dict,
    selected: bool,
    stage: int,
    dependencies: list[str],
    execution: dict | None,
) -> dict:
    execution_coherent = bool(
        selected
        and execution is not None
        and execution.get("observed")
        and execution.get("coherent")
    )
    return {
        **copy.deepcopy(action),
        "teacher": {
            "selected": selected,
            "selection_probability": 0.98 if selected else 0.02,
            "stage": stage if selected else 0,
            "depends_on_tools": dependencies if selected else [],
            "initial_ready": bool(selected and stage == 1),
            # Tool/API failures are not coherent executions. They can still
            # contribute attributed evidence to a truthful failure report.
            "coherence": float(execution_coherent),
            "contribution": float(selected),
            "continuation": float(
                selected
                and execution is not None
                and any(
                    call.get("failure_type") in {
                        "invalid_call",
                        "blocked_dependency",
                    }
                    for call in execution.get("calls", [])
                )
            ),
        },
        "execution": execution
        or {
            "calls": [],
            "observed": False,
            "coherent": False,
            "status": "not_executed_irrelevant",
        },
    }


def source_rows(
    rows: list[dict],
    distractors: int,
    max_concurrency: int,
    seed: int,
) -> list[dict]:
    domain_pool = defaultdict(list)
    global_pool = []
    for row in rows:
        for branch in row.get("branches") or []:
            action = source_action(branch)
            domain_pool[str(row.get("domain") or "")].append(action)
            global_pool.append(action)
    domain_pool = {
        key: deduplicate_actions(value) for key, value in domain_pool.items()
    }
    global_pool = deduplicate_actions(global_pool)

    output = []
    for row in rows:
        selected_branches = defaultdict(list)
        selected_actions = []
        for branch in row.get("branches") or []:
            selected_branches[str(branch["tool_name"])].append(branch)
            selected_actions.append(source_action(branch))
        selected_actions = deduplicate_actions(selected_actions)
        selected_names = [tool_key(action) for action in selected_actions]
        selected_set = set(selected_names)
        dependencies = branch_dependencies(row, selected_set)
        stages = topological_stages(selected_names, dependencies)
        initial_wave, later_wave, wave_count = wave_targets(
            stages, max_concurrency
        )
        rng = random.Random(f"{seed}:{row['task_id']}")
        negative_pool = [
            action
            for action in (
                domain_pool.get(str(row.get("domain") or ""), [])
                + global_pool
            )
            if tool_key(action) not in selected_set
        ]
        negative_pool = deduplicate_actions(negative_pool)
        rng.shuffle(negative_pool)
        candidates = selected_actions + negative_pool[:distractors]
        rng.shuffle(candidates)
        candidate_rows = []
        for action in candidates:
            name = tool_key(action)
            selected = name in selected_set
            candidate_rows.append(
                make_candidate(
                    action,
                    selected,
                    stages.get(name, 0),
                    dependencies.get(name, []),
                    source_execution(selected_branches[name])
                    if selected
                    else None,
                )
            )
        task_solvable = bool(float(row.get("task_solvability_target", 0.0)))
        output.append(
            {
                "schema": "rcg_moe_opd_v1",
                "task_id": str(row["task_id"]),
                "source": str(row.get("source") or ""),
                "domain": str(row.get("domain") or ""),
                "query": str(row["query"]),
                "final_answer": str(row.get("final_answer") or ""),
                "teacher_source": "trajectory_gold_live_replay",
                "teacher_weight": 0.65,
                "candidates": candidate_rows,
                "teacher": {
                    "selected_tools": selected_names,
                    "branch_count": len(selected_names),
                    "initial_wave_size": initial_wave,
                    "later_wave_size": later_wave,
                    "wave_count": wave_count,
                    "task_complete": True,
                    "task_solvable": task_solvable,
                    "takeover": bool(row.get("final_answer")),
                    "must_report_failure": not task_solvable,
                },
            }
        )
    return output


def benchmark_actions() -> list[dict]:
    return [
        runtime_action(
            name,
            spec["description"],
            {
                "type": "object",
                "properties": {
                    key: {"type": value}
                    for key, value in spec["parameters"].items()
                },
                "required": list(spec["parameters"]),
            },
        )
        for name, spec in BENCHMARK_TOOLS.items()
    ]


def benchmark_rows(path: Path, max_concurrency: int) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_by_id = {task["id"]: task for task in TASKS}
    output = []
    for result in payload.get("rows") or []:
        if result.get("system") != "qwen_pi_base" or not result.get(
            "task_success"
        ):
            continue
        task = task_by_id.get(str(result.get("task_id")))
        if task is None:
            continue
        traces = result.get("tool_trace") or []
        selected_names = list(dict.fromkeys(
            str(call.get("tool")) for call in traces if call.get("tool")
        ))
        selected_set = set(selected_names)
        dependencies = {
            target: [
                source
                for source in BENCHMARK_DEPENDENCIES.get(target, [])
                if source in selected_set
            ]
            for target in selected_names
        }
        stages = topological_stages(selected_names, dependencies)
        initial_wave, later_wave, wave_count = wave_targets(
            stages, max_concurrency
        )
        calls_by_tool = defaultdict(list)
        for call in traces:
            name = str(call.get("tool") or "")
            calls_by_tool[name].append(
                {
                    "status": (
                        "success"
                        if call.get("status") == "ok"
                        else failure_type(str(call.get("result") or ""))
                    ),
                    "success": call.get("status") == "ok",
                    "arguments": call.get("args") or {},
                    "result": str(call.get("result") or "")[:3000],
                    "failure_type": (
                        "success"
                        if call.get("status") == "ok"
                        else failure_type(str(call.get("result") or ""))
                    ),
                }
            )
        candidates = []
        for action in benchmark_actions():
            name = tool_key(action)
            selected = name in selected_set
            execution = (
                {
                    "calls": calls_by_tool[name],
                    "observed": True,
                    "coherent": bool(
                        calls_by_tool[name]
                        and all(call["success"] for call in calls_by_tool[name])
                    ),
                }
                if selected
                else None
            )
            candidates.append(
                make_candidate(
                    action,
                    selected,
                    stages.get(name, 0),
                    dependencies.get(name, []),
                    execution,
                )
            )
        unrecovered = {
            name
            for name, calls in calls_by_tool.items()
            if any(not call["success"] for call in calls)
            and not any(
                name in dependencies.get(candidate, [])
                and any(
                    call["success"] for call in calls_by_tool.get(candidate, [])
                )
                for candidate in selected_names
            )
        }
        output.append(
            {
                "schema": "rcg_moe_opd_v1",
                "task_id": f"base::{task['id']}",
                "source": str(path),
                "domain": "local_agent_harness",
                "query": task["prompt"],
                "final_answer": str(result.get("final_text") or ""),
                "teacher_source": "successful_qwen_base_trajectory",
                "teacher_weight": 1.0,
                "candidates": candidates,
                "teacher": {
                    "selected_tools": selected_names,
                    "branch_count": len(selected_names),
                    "initial_wave_size": initial_wave,
                    "later_wave_size": later_wave,
                    "wave_count": wave_count,
                    "task_complete": True,
                    "task_solvable": not bool(unrecovered),
                    "takeover": True,
                    "must_report_failure": bool(unrecovered),
                },
            }
        )
    return output


def failure_type(text: str) -> str:
    lowered = text.lower()
    if "timeout" in lowered:
        return "timeout"
    if any(token in lowered for token in ("401", "403", "unauthorized")):
        return "auth_or_quota"
    if any(token in lowered for token in ("500", "502", "503", "504")):
        return "server_or_network"
    if any(
        token in lowered
        for token in (
            "400",
            "invalid",
            "do not match",
            "does not match",
            "missing required",
        )
    ):
        return "invalid_call"
    if "not found" in lowered or "404" in lowered:
        return "api_unavailable"
    return "other_failure"


def validate(rows: list[dict]) -> dict:
    errors = []
    teacher_sources = Counter()
    for row in rows:
        teacher_sources[row["teacher_source"]] += 1
        candidates = row.get("candidates") or []
        names = [candidate["tool_name"] for candidate in candidates]
        selected = [
            candidate["tool_name"]
            for candidate in candidates
            if candidate["teacher"]["selected"]
        ]
        if len(names) != len(set(names)):
            errors.append(f"{row['task_id']}: duplicate candidate tool")
        if set(selected) != set(row["teacher"]["selected_tools"]):
            errors.append(f"{row['task_id']}: selected tool mismatch")
        if len(selected) != int(row["teacher"]["branch_count"]):
            errors.append(f"{row['task_id']}: branch count mismatch")
        if not selected:
            errors.append(f"{row['task_id']}: no teacher-selected tools")
        for candidate in candidates:
            for source in candidate["teacher"]["depends_on_tools"]:
                if source not in selected:
                    errors.append(
                        f"{row['task_id']}: dependency outside teacher set"
                    )
    if errors:
        raise ValueError("\n".join(errors[:30]))
    return {
        "rows": len(rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "teacher_sources": dict(teacher_sources),
        "mean_candidates": (
            sum(len(row["candidates"]) for row in rows) / max(len(rows), 1)
        ),
        "mean_selected": (
            sum(row["teacher"]["branch_count"] for row in rows)
            / max(len(rows), 1)
        ),
        "task_solvable": dict(
            Counter(bool(row["teacher"]["task_solvable"]) for row in rows)
        ),
    }


def expand_snapshots(
    rows: list[dict],
    max_snapshots_per_row: int,
) -> list[dict]:
    """Add intermediate execution states from each teacher trajectory."""
    output = list(rows)
    for row in rows:
        final_stage = int(row["teacher"]["wave_count"])
        if final_stage <= 1:
            continue
        candidates = list(range(1, final_stage))
        if len(candidates) > max_snapshots_per_row:
            positions = {
                round(
                    index * (len(candidates) - 1)
                    / max(max_snapshots_per_row - 1, 1)
                )
                for index in range(max_snapshots_per_row)
            }
            candidates = [candidates[index] for index in sorted(positions)]
        for stage in candidates:
            snapshot = copy.deepcopy(row)
            snapshot["task_id"] = f"{row['task_id']}::snapshot_{stage}"
            snapshot["teacher_source"] = (
                f"{row['teacher_source']}_intermediate"
            )
            snapshot["teacher_weight"] = min(
                float(row.get("teacher_weight", 1.0)), 0.8
            )
            observed_failure = False
            for candidate in snapshot["candidates"]:
                teacher = candidate["teacher"]
                observed = bool(
                    teacher["selected"] and int(teacher["stage"]) <= stage
                )
                if not observed:
                    candidate["execution"] = {
                        "calls": [],
                        "observed": False,
                        "coherent": False,
                        "status": "not_yet_executed",
                    }
                    teacher["coherence"] = 0.0
                    teacher["contribution"] = 0.0
                    teacher["continuation"] = 0.0
                else:
                    observed_failure = observed_failure or any(
                        not bool(call.get("success"))
                        for call in candidate["execution"].get("calls", [])
                    )
            snapshot["teacher"].update(
                {
                    "snapshot_stage": stage,
                    "task_complete": False,
                    "task_solvability_known": False,
                    "takeover": False,
                    "must_report_failure": observed_failure,
                }
            )
            output.append(snapshot)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/traject_swarm_training_failure_aware_v11.jsonl"),
    )
    parser.add_argument(
        "--base-comparison",
        type=Path,
        default=Path("eval_results/local_qwen_v11_best/comparison.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rcg_moe_opd_v1.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("eval_results/rcg_moe_opd_v1_data_summary.json"),
    )
    parser.add_argument("--distractors", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-snapshots-per-row", type=int, default=3
    )
    args = parser.parse_args()

    rows = source_rows(
        read_jsonl(args.source),
        args.distractors,
        args.max_concurrency,
        args.seed,
    )
    rows.extend(benchmark_rows(args.base_comparison, args.max_concurrency))
    rows = expand_snapshots(rows, args.max_snapshots_per_row)
    random.Random(args.seed).shuffle(rows)
    summary = validate(rows)
    write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
