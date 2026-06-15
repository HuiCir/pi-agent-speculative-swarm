#!/usr/bin/env python3
"""Run real DeepSeek Pi-agent loops against controlled agent tools."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NATIVE_REPO = ROOT.parent
RCG_REPO = ROOT.parent
RESULT_DIR = ROOT / "eval_results" / "v10_real_swarm"
MODEL = "deepseek/deepseek-v4-flash"
TOOLS = {
    "resolve_customer",
    "fetch_orders",
    "fetch_order_total",
    "check_budget",
    "get_weather",
    "search_flights",
    "get_fx_rate",
    "geocode",
    "route_time",
    "review_summary",
    "inventory_status",
    "inventory_fallback",
    "billing_status",
}

TASKS = [
    {
        "id": "parallel_travel",
        "prompt": (
            "Complete all three independent actions with the available tools: "
            "get Tokyo weather, search a flight from SFO to Tokyo on 2026-07-02, "
            "and get the USD/JPY exchange rate. Execute independent calls in "
            "parallel when possible. Report every exact result and do not omit "
            "a requested action."
        ),
        "gold_tools": ["get_weather", "search_flights", "get_fx_rate"],
        "markers": ["sunny", "DS204", "157.25"],
    },
    {
        "id": "serial_orders_reviews",
        "prompt": (
            "First resolve customer Alice to a customer_id, then fetch her "
            "orders using exactly that returned customer_id. Independently "
            "summarize reviews for laptop-pro. Complete every action, preserve "
            "the serial dependency, ignore diagnostic noise, and report the "
            "exact useful results."
        ),
        "gold_tools": [
            "resolve_customer",
            "fetch_orders",
            "review_summary",
        ],
        "markers": ["cust-42", "ord-7", "positive"],
        "dependency": "orders",
    },
    {
        "id": "serial_route_weather",
        "prompt": (
            "Geocode both Golden Gate Bridge and Ferry Building. After both "
            "results are available, call route_time using the exact returned "
            "coordinates. Independently get San Francisco weather. Multiple "
            "calls to geocode are required. Report the exact route and weather "
            "results without inventing coordinates."
        ),
        "gold_tools": ["geocode", "route_time", "get_weather"],
        "markers": ["24 minutes", "sunny"],
        "dependency": "route",
        "minimum_calls": {"geocode": 2},
    },
    {
        "id": "partial_503",
        "prompt": (
            "Check inventory for SKU-404 and independently get London weather. "
            "Use both requested tools. If inventory_status fails, do not invent "
            "availability: state that the overall request is only partially "
            "resolved or currently unsolvable, and name the failed tool and "
            "its exact error. Preserve and report the successful weather result."
        ),
        "gold_tools": ["inventory_status", "get_weather"],
        "markers": ["sunny"],
        "failure_tool": "inventory_status",
        "failure_terms": ["503", "service unavailable", "inventory_status"],
        "forbidden": ["INVENTORY_AVAILABLE", "IN_STOCK"],
    },
    {
        "id": "partial_401",
        "prompt": (
            "Resolve customer Alice, then check billing_status for account "
            "acct-9. Use both tools. If billing authentication fails, do not "
            "fabricate billing state: explicitly identify billing_status, the "
            "401/Unauthorized credential failure, and that this part cannot be "
            "completed. Preserve the successful customer result."
        ),
        "gold_tools": ["resolve_customer", "billing_status"],
        "markers": ["cust-42"],
        "failure_tool": "billing_status",
        "failure_terms": ["401", "unauthorized", "billing_status"],
        "forbidden": ["BILLING_OK", "ACCOUNT_CURRENT"],
    },
    {
        "id": "long_chain_budget",
        "prompt": (
            "Execute this full dependency chain: resolve Alice to customer_id; "
            "use that exact id to fetch orders; use the exact returned order_ids "
            "to call fetch_order_total; then check_budget with that exact total "
            "and limit 200. Independently get Berlin weather. Do not skip any "
            "stage, do not hardcode upstream values before observing them, and "
            "report the final budget and weather results."
        ),
        "gold_tools": [
            "resolve_customer",
            "fetch_orders",
            "fetch_order_total",
            "check_budget",
            "get_weather",
        ],
        "markers": ["within_limit", "125", "sunny"],
        "dependency": "long_chain",
    },
    {
        "id": "recovery_fallback",
        "prompt": (
            "Check inventory for SKU-404 with inventory_status. Only after "
            "observing its 503 failure, recover by calling inventory_fallback "
            "for the same SKU. Report the primary failure and the exact fallback "
            "inventory result. Do not call the fallback before failure evidence."
        ),
        "gold_tools": ["inventory_status", "inventory_fallback"],
        "markers": ["503", "12", "fallback"],
        "dependency": "recovery",
        "required_error_tool": "inventory_status",
    },
]


def repo_for(system: str) -> Path:
    return NATIVE_REPO if system in {"base", "pi_native"} else RCG_REPO


def extension_for(repo: Path) -> Path:
    return (
        repo
        / "packages/coding-agent/test/agent-swarm-benchmark-extension.ts"
    )


def command_for(system: str, prompt: str, rcg_url: str) -> list[str]:
    repo = repo_for(system)
    command = [
        "node",
        "packages/coding-agent/dist/cli.js",
        "--offline",
        "--model",
        MODEL,
        "--thinking",
        "off",
        "--mode",
        "json",
        "--no-session",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-builtin-tools",
        "--extension",
        str(extension_for(repo)),
    ]
    if system != "base":
        command.extend(
            [
                "--swarm",
                "--swarm-model",
                MODEL,
                "--swarm-agents",
                "A,B,C,D",
                "--swarm-rounds",
                "1",
                "--swarm-max-turns",
                "3",
                "--swarm-timeout-ms",
                "45000",
                "--swarm-tool-policy",
                "all",
            ]
        )
    if system in {"raw", "v10"}:
        command.extend(
            [
                "--swarm-rcg-url",
                rcg_url,
                "--swarm-rcg-timeout-ms",
                "60000",
            ]
        )
    command.extend(["-p", prompt])
    return command


def parse_stdout(stdout: str):
    events = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    agent_end = next(
        (event for event in reversed(events) if event.get("type") == "agent_end"),
        {},
    )
    messages = agent_end.get("messages") or []
    assistants = [
        message for message in messages if message.get("role") == "assistant"
    ]
    final_text = ""
    for message in reversed(assistants):
        texts = [
            item.get("text", "")
            for item in message.get("content") or []
            if item.get("type") == "text"
        ]
        if texts:
            final_text = "\n".join(texts)
            break
    usage = {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": 0.0,
    }
    for message in assistants:
        item = message.get("usage") or {}
        for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
            usage[key] += int(item.get(key) or 0)
        usage["cost"] += float((item.get("cost") or {}).get("total") or 0.0)
    return events, assistants, final_text, usage


def load_jsonl(path: Path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def swarm_metrics(rows):
    rounds = [row.get("round", {}) for row in rows if row.get("type") == "swarm_round"]
    retained = sum(len(row.get("retainedDrafts") or []) for row in rounds)
    suppressed = sum(len(row.get("suppressedDrafts") or []) for row in rounds)
    transfers = sum(len(row.get("dependencyTransfers") or []) for row in rounds)
    draft_tools = []
    swarm_tokens = 0
    for row in rounds:
        for draft in row.get("drafts") or []:
            signature = tuple(
                call.get("toolName") for call in draft.get("toolCalls") or []
            )
            draft_tools.append(signature)
            swarm_tokens += int((draft.get("usage") or {}).get("totalTokens") or 0)
        swarm_tokens += int(
            (row.get("preflight", {}).get("usage") or {}).get("totalTokens")
            or 0
        )
    return {
        "rounds": len(rounds),
        "retained": retained,
        "suppressed": suppressed,
        "dependency_transfers": transfers,
        "unique_draft_tool_signatures": len(set(draft_tools)),
        "swarm_tokens": swarm_tokens,
    }


def score(task, tool_rows, final_text):
    lower = final_text.lower()
    normalized = re.sub(r"[^a-z0-9]+", "", lower)
    gold = set(task["gold_tools"])
    called = {row.get("tool") for row in tool_rows}
    successful = {
        row.get("tool")
        for row in tool_rows
        if row.get("status") == "ok"
    }
    expected_failures = {
        name
        for name in (
            task.get("failure_tool"),
            task.get("required_error_tool"),
        )
        if name
    }
    expected_success = gold - expected_failures
    coverage = len(gold & called) / max(len(gold), 1)
    planning_precision = len(gold & called) / max(len(called), 1)
    success_coverage = len(expected_success & successful) / max(
        len(expected_success), 1
    )
    marker_recall = sum(
        re.sub(r"[^a-z0-9]+", "", marker.lower()) in normalized
        for marker in task.get("markers", [])
    ) / max(len(task.get("markers", [])), 1)
    minimum_calls_ok = all(
        sum(row.get("tool") == name for row in tool_rows) >= count
        for name, count in task.get("minimum_calls", {}).items()
    )
    dependency_ok = True
    if task.get("dependency") == "orders":
        dependency_ok = any(
            row.get("tool") == "fetch_orders"
            and row.get("status") == "ok"
            and (row.get("args") or {}).get("customer_id") == "cust-42"
            for row in tool_rows
        )
    elif task.get("dependency") == "route":
        dependency_ok = any(
            row.get("tool") == "route_time"
            and row.get("status") == "ok"
            for row in tool_rows
        )
    elif task.get("dependency") == "long_chain":
        dependency_ok = (
            any(
                row.get("tool") == "fetch_orders"
                and row.get("status") == "ok"
                and (row.get("args") or {}).get("customer_id") == "cust-42"
                for row in tool_rows
            )
            and any(
                row.get("tool") == "fetch_order_total"
                and row.get("status") == "ok"
                and (row.get("args") or {}).get("order_ids")
                == "ord-7,ord-8"
                for row in tool_rows
            )
            and any(
                row.get("tool") == "check_budget"
                and row.get("status") == "ok"
                and (row.get("args") or {}).get("order_total") == 125
                and (row.get("args") or {}).get("limit") == 200
                for row in tool_rows
            )
        )
    elif task.get("dependency") == "recovery":
        primary = [
            row
            for row in tool_rows
            if row.get("tool") == "inventory_status"
            and row.get("status") == "error"
        ]
        fallback = [
            row
            for row in tool_rows
            if row.get("tool") == "inventory_fallback"
            and row.get("status") == "ok"
        ]
        dependency_ok = bool(primary and fallback) and min(
            row["timestamp"] for row in fallback
        ) >= min(row["timestamp"] for row in primary)
    failure_ok = True
    if task.get("failure_tool"):
        failure_ok = (
            any(
                row.get("tool") == task["failure_tool"]
                and row.get("status") == "error"
                for row in tool_rows
            )
            and sum(term in lower for term in task["failure_terms"]) >= 2
        )
    if task.get("required_error_tool"):
        failure_ok = failure_ok and any(
            row.get("tool") == task["required_error_tool"]
            and row.get("status") == "error"
            for row in tool_rows
        )
    hallucination_free = not any(
        term.lower() in lower for term in task.get("forbidden", [])
    )
    task_success = all(
        [
            coverage == 1.0,
            success_coverage == 1.0,
            marker_recall == 1.0,
            minimum_calls_ok,
            dependency_ok,
            failure_ok,
            hallucination_free,
            planning_precision == 1.0,
        ]
    )
    return {
        "task_success": task_success,
        "tool_coverage": coverage,
        "planning_precision": planning_precision,
        "successful_tool_coverage": success_coverage,
        "marker_recall": marker_recall,
        "minimum_calls_ok": minimum_calls_ok,
        "dependency_ok": dependency_ok,
        "failure_attribution_ok": failure_ok,
        "hallucination_free": hallucination_free,
        "unique_tools_called": len(called),
        "total_tool_calls": len(tool_rows),
        "irrelevant_tools": sorted((called - gold) & TOOLS),
    }


def run_case(system, task, args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{system}_{task['id']}"
    tool_trace = RESULT_DIR / f"{stem}.tools.jsonl"
    swarm_trace = RESULT_DIR / f"{stem}.swarm.jsonl"
    stdout_path = RESULT_DIR / f"{stem}.events.jsonl"
    stderr_path = RESULT_DIR / f"{stem}.stderr.log"
    for path in (tool_trace, swarm_trace, stdout_path, stderr_path):
        path.unlink(missing_ok=True)
    env = dict(os.environ)
    env["RCG_AGENT_BENCH_TRACE"] = str(tool_trace)
    env["PI_SWARM_TRACE_FILE"] = str(swarm_trace)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command_for(system, task["prompt"], args.rcg_url),
            cwd=repo_for(system),
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        process = None
        timed_out = True
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    else:
        stdout = process.stdout
        stderr = process.stderr
    latency = time.monotonic() - started
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    events, assistants, final_text, usage = parse_stdout(stdout)
    tool_rows = load_jsonl(tool_trace)
    swarm_rows = load_jsonl(swarm_trace)
    result = {
        "system": system,
        "task_id": task["id"],
        "returncode": None if process is None else process.returncode,
        "timed_out": timed_out,
        "latency_s": round(latency, 3),
        "main_assistant_turns": len(assistants),
        "main_usage": usage,
        "swarm": swarm_metrics(swarm_rows),
        "tool_trace": tool_rows,
        "final_text": final_text,
        "stderr_tail": stderr[-2000:],
    }
    result.update(score(task, tool_rows, final_text))
    return result


def summarize(rows):
    systems = {}
    for system in sorted({row["system"] for row in rows}):
        group = [row for row in rows if row["system"] == system]
        count = max(len(group), 1)
        systems[system] = {
            "cases": len(group),
            "task_success_rate": sum(row["task_success"] for row in group)
            / count,
            "tool_coverage": sum(row["tool_coverage"] for row in group)
            / count,
            "planning_precision": sum(
                row["planning_precision"] for row in group
            )
            / count,
            "successful_tool_coverage": sum(
                row["successful_tool_coverage"] for row in group
            )
            / count,
            "dependency_accuracy": sum(row["dependency_ok"] for row in group)
            / count,
            "failure_attribution_accuracy": sum(
                row["failure_attribution_ok"] for row in group
            )
            / count,
            "hallucination_free_rate": sum(
                row["hallucination_free"] for row in group
            )
            / count,
            "mean_latency_s": sum(row["latency_s"] for row in group) / count,
            "mean_main_turns": sum(
                row["main_assistant_turns"] for row in group
            )
            / count,
            "mean_tool_calls": sum(
                row["total_tool_calls"] for row in group
            )
            / count,
            "main_tokens": sum(
                row["main_usage"]["totalTokens"] for row in group
            ),
            "swarm_tokens": sum(
                row["swarm"]["swarm_tokens"] for row in group
            ),
            "dependency_transfers": sum(
                row["swarm"]["dependency_transfers"] for row in group
            ),
        }
    return systems


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems", default="base,pi_native,v10"
    )
    parser.add_argument(
        "--rcg-url", default="http://127.0.0.1:8765"
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--task-ids", default="")
    parser.add_argument(
        "--output",
        default=str(RESULT_DIR / "comparison.json"),
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--rescore-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    rows = []
    if args.append and output.exists():
        rows = json.loads(output.read_text(encoding="utf-8")).get("rows", [])
    if args.rescore_only:
        if not output.exists():
            raise FileNotFoundError(output)
        rows = json.loads(output.read_text(encoding="utf-8")).get("rows", [])
        tasks = {task["id"]: task for task in TASKS}
        for row in rows:
            row.update(
                score(
                    tasks[row["task_id"]],
                    row.get("tool_trace", []),
                    row.get("final_text", ""),
                )
            )
        atomic_json(
            output,
            {"model": MODEL, "systems": summarize(rows), "rows": rows},
        )
        print(json.dumps(summarize(rows), indent=2), flush=True)
        return
    systems = [
        item.strip()
        for item in args.systems.split(",")
        if item.strip()
    ]
    valid = {"base", "pi_native", "raw", "v10"}
    unknown = set(systems) - valid
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}")
    selected_task_ids = {
        item.strip()
        for item in args.task_ids.split(",")
        if item.strip()
    }
    selected_tasks = [
        task
        for task in TASKS
        if not selected_task_ids or task["id"] in selected_task_ids
    ]
    missing_tasks = selected_task_ids - {
        task["id"] for task in selected_tasks
    }
    if missing_tasks:
        raise ValueError(f"unknown task ids: {sorted(missing_tasks)}")
    for system in systems:
        rows = [
            row
            for row in rows
            if not (
                row["system"] == system
                and row["task_id"] in {
                    task["id"] for task in selected_tasks
                }
            )
        ]
        for task in selected_tasks:
            print(f"[{system}] {task['id']}", flush=True)
            row = run_case(system, task, args)
            rows.append(row)
            atomic_json(
                output,
                {
                    "model": MODEL,
                    "systems": summarize(rows),
                    "rows": rows,
                },
            )
            print(
                json.dumps(
                    {
                        "success": row["task_success"],
                        "coverage": row["tool_coverage"],
                        "latency_s": row["latency_s"],
                        "turns": row["main_assistant_turns"],
                        "tool_calls": row["total_tool_calls"],
                    }
                ),
                flush=True,
            )
    atomic_json(
        output,
        {"model": MODEL, "systems": summarize(rows), "rows": rows},
    )
    print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
