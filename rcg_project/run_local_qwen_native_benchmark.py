#!/usr/bin/env python3
"""Compare native Qwen3-8B Pi base with the fully native Qwen-RCG swarm."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from run_v10_real_swarm_benchmark import (
    TASKS,
    atomic_json,
    load_jsonl,
    parse_stdout,
    score,
    summarize,
    swarm_metrics,
)


ROOT = Path(__file__).resolve().parent
SWARM_REPO = ROOT.parent
DEFAULT_RESULT_DIR = ROOT / "eval_results" / "local_qwen_native"
DEFAULT_MODEL = Path(
    os.environ.get("PI_LOCAL_QWEN_MODEL", ROOT.parent / "models/qwen3-8b")
)
DEFAULT_PYTHON = Path(
    os.environ.get("PI_LOCAL_QWEN_PYTHON", ROOT / ".venv/bin/python")
)
DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints" / "fascia_best.pt"
)


def summarize_local(rows: list[dict]) -> dict:
    systems = summarize(rows)
    for system, metrics in systems.items():
        group = [row for row in rows if row["system"] == system]
        count = max(len(group), 1)
        metrics["mean_main_model_generations"] = sum(
            row.get(
                "main_model_generations",
                row["main_assistant_turns"],
            )
            for row in group
        ) / count
        metrics["direct_takeover_rate"] = sum(
            bool(row.get("direct_takeover")) for row in group
        ) / count
        metrics["main_generation_tokens"] = sum(
            row.get(
                "main_generation_usage",
                row["main_usage"],
            )["totalTokens"]
            for row in group
        )
        policy_rows = [
            row.get("api_policy") or {}
            for row in group
        ]
        metrics["api_policy_calls"] = sum(
            int(item.get("calls") or 0) for item in policy_rows
        )
        metrics["api_policy_latency_s"] = sum(
            float(item.get("latencyMs") or 0) for item in policy_rows
        ) / 1000
        metrics["api_policy_tokens"] = sum(
            int(item.get("promptTokens") or 0)
            + int(item.get("outputTokens") or 0)
            for item in policy_rows
        )
    return systems


def parse_api_policy_metrics(stderr: str) -> dict:
    total = {
        "calls": 0,
        "latencyMs": 0,
        "promptTokens": 0,
        "outputTokens": 0,
        "byCommand": {},
    }
    pattern = re.compile(r"prompt policy (\{.*\})$")
    for line in stderr.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            row = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        command = str(row.get("command") or "unknown")
        total["calls"] += 1
        total["latencyMs"] += int(row.get("latencyMs") or 0)
        total["promptTokens"] += int(row.get("promptTokens") or 0)
        total["outputTokens"] += int(row.get("outputTokens") or 0)
        command_stats = total["byCommand"].setdefault(
            command,
            {
                "calls": 0,
                "latencyMs": 0,
                "promptTokens": 0,
                "outputTokens": 0,
            },
        )
        for key in ("calls", "latencyMs", "promptTokens", "outputTokens"):
            command_stats[key] += (
                1 if key == "calls" else int(row.get(key) or 0)
            )
    return total


def command_for(
    system: str,
    task: dict,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        "node",
        "packages/coding-agent/dist/cli.js",
        "--offline",
        "--local-qwen",
        "--local-model-path",
        str(args.model_path),
        "--local-python",
        str(args.python),
        "--local-cache-gb",
        str(args.cache_gb),
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
        str(
            SWARM_REPO
            / "packages/coding-agent/test/agent-swarm-benchmark-extension.ts"
        ),
    ]
    if system in {"qwen_rcg_swarm", "qwen_api_harness_swarm"}:
        command.extend(
            [
                "--swarm",
                "--swarm-policy",
                "prompt" if system == "qwen_api_harness_swarm" else "trained",
                "--swarm-agents",
                args.agents,
                "--swarm-rounds",
                str(args.rounds),
                "--swarm-max-turns",
                str(args.max_turns),
                "--swarm-timeout-ms",
                str(args.swarm_timeout_ms),
                "--swarm-tool-policy",
                "all",
            ]
        )
        if system == "qwen_rcg_swarm":
            command.extend(
                [
                    "--swarm-local-rcg-checkpoint",
                    str(args.checkpoint),
                ]
            )
    command.extend(["-p", task["prompt"]])
    return command


def run_case(
    system: str,
    task: dict,
    args: argparse.Namespace,
    result_dir: Path,
) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{system}_{task['id']}"
    tool_trace = result_dir / f"{stem}.tools.jsonl"
    swarm_trace = result_dir / f"{stem}.swarm.jsonl"
    stdout_path = result_dir / f"{stem}.events.jsonl"
    stderr_path = result_dir / f"{stem}.stderr.log"
    for path in (tool_trace, swarm_trace, stdout_path, stderr_path):
        path.unlink(missing_ok=True)

    env = dict(os.environ)
    env["RCG_AGENT_BENCH_TRACE"] = str(tool_trace)
    env["PI_SWARM_TRACE_FILE"] = str(swarm_trace)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command_for(system, task, args),
            cwd=SWARM_REPO,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        timed_out = False
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as error:
        process = None
        timed_out = True
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    latency = time.monotonic() - started

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    _, assistants, final_text, usage = parse_stdout(stdout)
    tool_rows = load_jsonl(tool_trace)
    swarm_rows = load_jsonl(swarm_trace)
    swarm_rounds = [
        row.get("round", row)
        for row in swarm_rows
        if row.get("type") == "swarm_round"
    ]
    direct_takeover = bool(
        swarm_rounds
        and swarm_rounds[-1].get("mainDecision") == "takeover"
    )
    result = {
        "system": system,
        "task_id": task["id"],
        "returncode": None if process is None else process.returncode,
        "timed_out": timed_out,
        "latency_s": round(latency, 3),
        "main_assistant_turns": len(assistants),
        "main_model_generations": (
            0 if direct_takeover else len(assistants)
        ),
        "direct_takeover": direct_takeover,
        "main_usage": usage,
        "main_generation_usage": (
            {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": 0.0,
            }
            if direct_takeover
            else usage
        ),
        "swarm": swarm_metrics(swarm_rows),
        "tool_trace": tool_rows,
        "final_text": final_text,
        "stderr_tail": stderr[-4000:],
        "api_policy": parse_api_policy_metrics(stderr),
    }
    result.update(score(task, tool_rows, final_text))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        default="qwen_pi_base,qwen_api_harness_swarm,qwen_rcg_swarm",
    )
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--cache-gb", type=float, default=8)
    parser.add_argument("--agents", default="A,B,C,D")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--swarm-timeout-ms", type=int, default=45000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT_DIR / "comparison.json",
    )
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    args.model_path = args.model_path.resolve()
    args.python = args.python.absolute()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()

    for path in (args.model_path, args.python, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    systems = [
        item.strip()
        for item in args.systems.split(",")
        if item.strip()
    ]
    unknown = set(systems) - {
        "qwen_pi_base",
        "qwen_api_harness_swarm",
        "qwen_rcg_swarm",
    }
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}")
    selected_ids = {
        item.strip()
        for item in args.task_ids.split(",")
        if item.strip()
    }
    selected_tasks = [
        task
        for task in TASKS
        if not selected_ids or task["id"] in selected_ids
    ]
    missing = selected_ids - {
        task["id"]
        for task in selected_tasks
    }
    if missing:
        raise ValueError(f"unknown task ids: {sorted(missing)}")

    rows: list[dict] = []
    if args.append and args.output.exists():
        rows = json.loads(
            args.output.read_text(encoding="utf-8")
        ).get("rows", [])
    result_dir = args.output.parent
    selected_task_ids = {task["id"] for task in selected_tasks}
    for system in systems:
        rows = [
            row
            for row in rows
            if not (
                row["system"] == system
                and row["task_id"] in selected_task_ids
            )
        ]
        for task in selected_tasks:
            print(f"[{system}] {task['id']}", flush=True)
            row = run_case(system, task, args, result_dir)
            rows.append(row)
            atomic_json(
                args.output,
                {
                    "model": str(args.model_path),
                    "checkpoint": str(args.checkpoint),
                    "systems": summarize_local(rows),
                    "rows": rows,
                },
            )
            print(
                json.dumps(
                    {
                        "success": row["task_success"],
                        "coverage": row["tool_coverage"],
                        "precision": row["planning_precision"],
                        "latency_s": row["latency_s"],
                        "turns": row["main_assistant_turns"],
                        "tool_calls": row["total_tool_calls"],
                    }
                ),
                flush=True,
            )
    print(json.dumps(summarize_local(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
