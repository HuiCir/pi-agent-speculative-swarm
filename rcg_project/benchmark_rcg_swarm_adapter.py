"""Replay TRAJECT-Bench candidate actions through the live RCG swarm sidecar."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from collections import defaultdict

from dynamic_swarm_train import (
    balanced_split,
    graph_levels,
    read_rows,
    select_nodes,
    strict_coherence,
)


WORD = re.compile(r"[a-z0-9_\u0080-\uffff]+", re.IGNORECASE)


def post(url, path, payload):
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def branch_tool(branch, task_id, gold):
    name = f"{task_id}::{branch['branch_id']}::{branch['tool_name']}"
    description = (
        f"{branch['tool_description']} "
        f"Required parameters: {json.dumps(branch['required_parameters'], ensure_ascii=False)}. "
        f"Produces: {json.dumps(branch.get('produces'), ensure_ascii=False)}."
    )
    return {
        "name": name,
        "description": description,
        "parameters": {
            item.get("name"): type(item.get("value")).__name__
            for item in branch.get("required_parameters", [])
        },
        "_branch": branch,
        "_gold": gold,
    }


def lexical_rank(query, candidates, count):
    query_words = set(WORD.findall(query.lower()))
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -len(query_words & set(WORD.findall(candidate["description"].lower()))),
            candidate["name"],
        ),
    )
    return ranked[:count]


def dependency_depth(branches):
    by_id = {branch["branch_id"]: index for index, branch in enumerate(branches)}
    dependency = [[False] * len(branches) for _ in branches]
    for target, branch in enumerate(branches):
        for edge in branch.get("depends_on", []):
            source = by_id.get(edge.get("source_branch_id"))
            if source is not None and source != target:
                dependency[target][source] = True
    levels = [0] * len(branches)
    for _ in branches:
        changed = False
        for target in range(len(branches)):
            sources = [
                source
                for source, present in enumerate(dependency[target])
                if present
            ]
            level = 0 if not sources else max(levels[source] + 1 for source in sources)
            if level > levels[target]:
                levels[target] = level
                changed = True
        if not changed:
            break
    return max(levels, default=0) + 1


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--data", default="data/traject_swarm_training.jsonl")
    parser.add_argument("--cases", type=int, default=120)
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument(
        "--output",
        default="eval_results/rcg_swarm_v8_live_adapter.json",
    )
    args = parser.parse_args()

    rows = read_rows(args.data)
    _, evaluation = balanced_split(rows, 600, 120, 43)
    evaluation = evaluation[: args.cases]
    pool = []
    for row in evaluation:
        for branch in select_nodes(row, 13):
            pool.append(branch_tool(branch, row["task_id"], False))

    records = []
    randomizer = random.Random(8128)
    for case_index, row in enumerate(evaluation, start=1):
        branches = select_nodes(row, 13)
        gold = [branch_tool(branch, row["task_id"], True) for branch in branches]
        gold_names = {candidate["name"] for candidate in gold}
        distractor_pool = [
            candidate for candidate in pool if candidate["name"] not in gold_names
        ]
        distractors = randomizer.sample(
            distractor_pool, min(args.distractors, len(distractor_pool))
        )
        candidates = gold + distractors
        randomizer.shuffle(candidates)
        maximum = len(gold)

        plan_started = time.perf_counter()
        plan = post(
            args.url,
            "/plan",
            {
                "task": row["query"],
                "requestIndex": 0,
                "maxBranches": maximum,
                "residualMemory": "",
                "tools": [
                    {
                        "name": candidate["name"],
                        "description": candidate["description"],
                        "parameters": candidate["parameters"],
                    }
                    for candidate in candidates
                ],
            },
        )
        plan_wall_ms = (time.perf_counter() - plan_started) * 1000
        selected_names = {
            action.get("toolName") for action in plan.get("actions", [])
        }
        baseline_selected = {
            candidate["name"]
            for candidate in lexical_rank(row["query"], candidates, maximum)
        }

        branch_by_name = {candidate["name"]: candidate for candidate in candidates}
        actions = plan.get("actions", [])
        drafts = []
        expected_coherent = {
            candidate["name"]
            for candidate in gold
            if strict_coherence(candidate["_branch"])
        }
        for action in actions:
            candidate = branch_by_name.get(action.get("toolName"))
            branch = candidate["_branch"] if candidate else {}
            success = strict_coherence(branch) == 1.0
            drafts.append(
                {
                    "briefId": action["id"],
                    "status": "ok",
                    "finalText": (
                        f"Tool result for {action.get('toolName')}: "
                        f"{branch.get('executed_output', '')}"
                    ),
                    "toolCalls": [
                        {
                            "toolName": action.get("toolName", action.get("title")),
                            "args": branch.get("required_parameters", []),
                            "isError": not success,
                            "resultPreview": str(
                                branch.get("executed_output", "")
                            )[:2000],
                        }
                    ],
                }
            )
        selection_started = time.perf_counter()
        selection = post(
            args.url,
            "/select",
            {
                "task": row["query"],
                "actions": actions,
                "drafts": drafts,
            },
        )
        selection_wall_ms = (time.perf_counter() - selection_started) * 1000
        action_name = {
            action["id"]: action.get("toolName") for action in actions
        }
        retained = {
            action_name.get(score["briefId"])
            for score in selection.get("scores", [])
            if score.get("retain")
        }
        baseline_retained = {
            candidate["name"]
            for candidate in gold
            if candidate["name"] in baseline_selected
            and candidate["_branch"].get("execution_status") == "success"
        }

        rcg_plan_tp = len(selected_names & gold_names)
        baseline_plan_tp = len(baseline_selected & gold_names)
        rcg_tp = len(retained & expected_coherent)
        rcg_fp = len(retained - expected_coherent)
        baseline_tp = len(baseline_retained & expected_coherent)
        baseline_fp = len(baseline_retained - expected_coherent)
        rcg_sufficient = expected_coherent.issubset(retained)
        baseline_sufficient = expected_coherent.issubset(baseline_retained)
        gold_tools = {
            candidate["_branch"].get("tool_name") for candidate in gold
        }
        expected_tools = {
            candidate["_branch"].get("tool_name")
            for candidate in gold
            if candidate["name"] in expected_coherent
        }

        def tools_for(names):
            return {
                branch_by_name[name]["_branch"].get("tool_name")
                for name in names
                if name in branch_by_name
            }

        rcg_plan_tools = tools_for(selected_names)
        baseline_plan_tools = tools_for(baseline_selected)
        rcg_retained_tools = tools_for(retained)
        baseline_retained_tools = tools_for(baseline_retained)
        depth = dependency_depth(branches)
        records.append(
            {
                "task_id": row["task_id"],
                "source_type": row.get("trajectory_type"),
                "nodes": len(branches),
                "depth": depth,
                "expected_coherent": len(expected_coherent),
                "rcg_plan_recall": safe_div(rcg_plan_tp, len(gold_names)),
                "rcg_plan_exact": float(gold_names.issubset(selected_names)),
                "baseline_plan_recall": safe_div(
                    baseline_plan_tp, len(gold_names)
                ),
                "baseline_plan_exact": float(
                    gold_names.issubset(baseline_selected)
                ),
                "rcg_selection_precision": safe_div(
                    rcg_tp, rcg_tp + rcg_fp
                ),
                "rcg_selection_recall": safe_div(
                    rcg_tp, len(expected_coherent)
                ),
                "baseline_selection_precision": safe_div(
                    baseline_tp, baseline_tp + baseline_fp
                ),
                "baseline_selection_recall": safe_div(
                    baseline_tp, len(expected_coherent)
                ),
                "rcg_planned_paths": len(selected_names),
                "baseline_planned_paths": len(baseline_selected),
                "rcg_retained_paths": len(retained),
                "baseline_retained_paths": len(baseline_retained),
                "rcg_plan_unique_tool_ratio": safe_div(
                    len(rcg_plan_tools), len(selected_names)
                ),
                "baseline_plan_unique_tool_ratio": safe_div(
                    len(baseline_plan_tools), len(baseline_selected)
                ),
                "rcg_plan_tool_coverage": safe_div(
                    len(rcg_plan_tools & gold_tools), len(gold_tools)
                ),
                "baseline_plan_tool_coverage": safe_div(
                    len(baseline_plan_tools & gold_tools), len(gold_tools)
                ),
                "rcg_selection_tool_coverage": safe_div(
                    len(rcg_retained_tools & expected_tools),
                    len(expected_tools),
                ),
                "baseline_selection_tool_coverage": safe_div(
                    len(baseline_retained_tools & expected_tools),
                    len(expected_tools),
                ),
                "rcg_one_main_turn": float(rcg_sufficient),
                "baseline_one_main_turn": float(baseline_sufficient),
                "traditional_main_turns": depth + 1,
                "rcg_main_turns": 1 if rcg_sufficient else 2,
                "baseline_main_turns": 1 if baseline_sufficient else 2,
                "plan_wall_ms": plan_wall_ms,
                "selection_wall_ms": selection_wall_ms,
                "policy_reported_ms": float(plan.get("latencyMs", 0))
                + float(selection.get("latencyMs", 0)),
            }
        )
        if case_index % 10 == 0:
            print(f"evaluated {case_index}/{len(evaluation)}", flush=True)

    metric_names = [
        "rcg_plan_recall",
        "rcg_plan_exact",
        "baseline_plan_recall",
        "baseline_plan_exact",
        "rcg_selection_precision",
        "rcg_selection_recall",
        "baseline_selection_precision",
        "baseline_selection_recall",
        "rcg_planned_paths",
        "baseline_planned_paths",
        "rcg_retained_paths",
        "baseline_retained_paths",
        "rcg_plan_unique_tool_ratio",
        "baseline_plan_unique_tool_ratio",
        "rcg_plan_tool_coverage",
        "baseline_plan_tool_coverage",
        "rcg_selection_tool_coverage",
        "baseline_selection_tool_coverage",
        "rcg_one_main_turn",
        "baseline_one_main_turn",
        "traditional_main_turns",
        "rcg_main_turns",
        "baseline_main_turns",
        "plan_wall_ms",
        "selection_wall_ms",
        "policy_reported_ms",
    ]

    def aggregate(items):
        return {
            "cases": len(items),
            **{
                metric: mean([record[metric] for record in items])
                for metric in metric_names
            },
        }

    result = {
        "config": vars(args),
        "overall": aggregate(records),
        "by_source_type": {
            source_type: aggregate(
                [
                    record
                    for record in records
                    if record["source_type"] == source_type
                ]
            )
            for source_type in sorted(
                {record["source_type"] for record in records}
            )
        },
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
