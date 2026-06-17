from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .schema import (
    EXPERT_NAMES,
    RECOVERY_NAMES,
    SCHEMA,
    empty_square,
    validate_event,
)


RECOVERY_ID = {name: index for index, name in enumerate(RECOVERY_NAMES)}
FAILURE_RECOVERY = {
    "success": "none",
    "timeout": "report_failure",
    "server_or_network": "report_failure",
    "auth_or_quota": "report_failure",
    "api_unavailable": "replan_distinct_action",
    "empty_result": "request_more_evidence",
    "invalid_call": "repair_call",
    "provenance_mismatch": "replan_distinct_action",
    "blocked_dependency": "wait_dependency",
    "other_failure": "report_failure",
}


def stable_int(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8], "big"
    )


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    total = 0
    candidates = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_event(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            candidates += len(row["candidates"])
            counts[row["source_type"]] += 1
    return {
        "rows": total,
        "mean_candidates": candidates / max(total, 1),
        "sources": dict(counts),
    }


def expert_profile(**weights: float) -> list[float]:
    values = [max(0.0, float(weights.get(name, 0.0))) for name in EXPERT_NAMES]
    total = sum(values)
    if total <= 0:
        values[-1] = 1.0
        total = 1.0
    return [value / total for value in values]


def event_row(
    *,
    task_id: str,
    group_id: str,
    source_type: str,
    domain: str,
    query: str,
    candidates: list[dict],
    admission: list[float],
    priority: list[float],
    stage: list[int],
    coherence: list[float],
    contribution: list[float],
    novelty: list[float],
    memory_keep: list[float],
    recovery: list[int],
    dependency: list[list[float]],
    residual_route: list[list[float]],
    global_labels: dict,
    expert_target: list[float],
    masks: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    row = {
        "schema": SCHEMA,
        "task_id": task_id,
        "group_id": group_id,
        "source_type": source_type,
        "domain": domain,
        "query": query,
        "candidates": candidates,
        "labels": {
            "node": {
                "admission": admission,
                "priority": priority,
                "stage": stage,
                "coherence": coherence,
                "contribution": contribution,
                "novelty": novelty,
                "memory_keep": memory_keep,
                "recovery": recovery,
            },
            "pair": {
                "dependency": dependency,
                "residual_route": residual_route,
            },
            "global": global_labels,
            "expert_target": expert_target,
            "mask": masks or {},
        },
        "metadata": metadata or {},
    }
    validate_event(row)
    return row


def _execution_evidence(candidate: dict) -> tuple[str, dict]:
    execution = candidate.get("execution") or {}
    calls = execution.get("calls") or []
    parts = []
    success = bool(execution.get("coherent"))
    failure = "other_failure"
    for call in calls:
        failure = str(call.get("failure_type") or call.get("status") or failure)
        parts.append(
            "status={status}; arguments={arguments}; result={result}".format(
                status=call.get("status", ""),
                arguments=json.dumps(
                    call.get("arguments") or {}, ensure_ascii=False
                ),
                result=str(call.get("result") or "")[:1200],
            )
        )
    return "\n".join(parts), {
        "observed": bool(execution.get("observed")),
        "success": success,
        "failure_type": failure,
        "latency_ratio": 0.0,
        "retry_count": max(0, len(calls) - 1),
    }


def trajectory_events(path: Path) -> Iterable[dict]:
    for source in read_jsonl(path):
        names = [str(item["tool_name"]) for item in source["candidates"]]
        name_to_index = {name: index for index, name in enumerate(names)}
        count = len(names)
        dependency = empty_square(count)
        residual = empty_square(count)
        candidates = []
        admission = []
        priority = []
        stages = []
        coherence = []
        contribution = []
        novelty = []
        memory_keep = []
        recovery = []
        observed_mask = []
        max_stage = max(
            [int(item["teacher"].get("stage", 0)) for item in source["candidates"]]
            + [1]
        )
        for target, item in enumerate(source["candidates"]):
            teacher = item["teacher"]
            evidence, runtime = _execution_evidence(item)
            candidates.append(
                {
                    "name": names[target],
                    "description": str(item.get("description") or ""),
                    "parameters": item.get("parameters") or {},
                    "evidence": evidence,
                    "runtime": runtime,
                }
            )
            selected = float(bool(teacher.get("selected")))
            item_stage = int(teacher.get("stage", 0))
            admission.append(selected)
            priority.append(
                selected * (max_stage + 1 - item_stage) / max_stage
                if item_stage
                else 0.0
            )
            stages.append(item_stage)
            item_coherence = float(teacher.get("coherence", 0.0))
            item_contribution = float(teacher.get("contribution", selected))
            coherence.append(item_coherence)
            contribution.append(item_contribution)
            novelty.append(selected)
            memory_keep.append(
                float(
                    bool(
                        item_contribution
                        or item_coherence
                        or teacher.get("continuation")
                    )
                )
            )
            failure = runtime["failure_type"]
            recovery.append(
                RECOVERY_ID[
                    FAILURE_RECOVERY.get(failure, "report_failure")
                    if runtime["observed"]
                    else "none"
                ]
            )
            observed_mask.append(float(runtime["observed"]))
            for source_name in teacher.get("depends_on_tools") or []:
                source_index = name_to_index.get(str(source_name))
                if source_index is not None:
                    dependency[target][source_index] = 1.0
                    residual[target][source_index] = 1.0
        teacher = source["teacher"]
        snapshot = "snapshot_stage" in teacher
        failed = any(
            candidate["runtime"]["observed"]
            and not candidate["runtime"]["success"]
            for candidate in candidates
        )
        yield event_row(
            task_id=str(source["task_id"]),
            group_id=str(source["task_id"]).split("::snapshot_", 1)[0],
            source_type="trajectory",
            domain=str(source.get("domain") or "trajectory"),
            query=str(source["query"]),
            candidates=candidates,
            admission=admission,
            priority=priority,
            stage=stages,
            coherence=coherence,
            contribution=contribution,
            novelty=novelty,
            memory_keep=memory_keep,
            recovery=recovery,
            dependency=dependency,
            residual_route=residual,
            global_labels={
                "branch_count": min(24, int(teacher["branch_count"])),
                "initial_wave_size": min(
                    8, int(teacher["initial_wave_size"])
                ),
                "later_wave_size": min(
                    8, int(teacher["later_wave_size"])
                ),
                "wave_count": min(12, int(teacher["wave_count"])),
                "task_complete": float(teacher["task_complete"]),
                "task_solvable": float(teacher["task_solvable"]),
                "must_report_failure": float(
                    teacher["must_report_failure"]
                ),
                "takeover": float(teacher["takeover"]),
                "halt": float(teacher["task_complete"]),
                "remaining_turns": max(
                    0,
                    int(teacher["wave_count"])
                    - int(teacher.get("snapshot_stage", teacher["wave_count"])),
                ),
            },
            expert_target=expert_profile(
                planning=1.0,
                scheduling=1.0,
                diversity=0.6,
                evidence=1.2 if any(observed_mask) else 0.1,
                recovery=1.2 if failed else 0.2,
                memory=0.8 if snapshot else 0.3,
                halting=1.0 if teacher["task_complete"] else 0.2,
                integration=1.0,
            ),
            masks={
                "coherence": observed_mask,
                "recovery": observed_mask,
                "task_solvable": float(
                    teacher.get("task_solvability_known", True)
                ),
            },
            metadata={
                "source": source.get("source"),
                "teacher_source": source.get("teacher_source"),
                "teacher_weight": source.get("teacher_weight", 1.0),
            },
        )


def _simple_candidates(names: list[str], descriptions: list[str]) -> list[dict]:
    return [
        {
            "name": name,
            "description": description,
            "parameters": {},
            "evidence": "",
            "runtime": {
                "observed": False,
                "success": False,
                "failure_type": "",
                "latency_ratio": 0.0,
                "retry_count": 0,
            },
        }
        for name, description in zip(names, descriptions)
    ]


def mmlu_events(limit: int, seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for source_index in indices[:limit]:
        source = dataset[source_index]
        choices = list(source["choices"])
        if len(choices) < 2:
            continue
        answer_raw = source["answer"]
        answer = (
            int(answer_raw)
            if str(answer_raw).isdigit()
            else ord(str(answer_raw).upper()[0]) - ord("A")
        )
        answer = max(0, min(len(choices) - 1, answer))
        names = [f"option_{chr(65 + index)}" for index in range(len(choices))]
        descriptions = [
            f"Independently test whether option {chr(65 + index)} is correct: {choice}"
            for index, choice in enumerate(choices)
        ]
        base = _simple_candidates(names, descriptions)
        count = len(base)
        zeros = empty_square(count)
        group = f"mmlu::{source['subject']}::{source_index}"
        query = str(source["question"])
        yield event_row(
            task_id=f"{group}::plan",
            group_id=group,
            source_type="mmlu",
            domain=str(source["subject"]),
            query=query,
            candidates=base,
            admission=[1.0] * count,
            priority=[1.0] * count,
            stage=[1] * count,
            coherence=[0.0] * count,
            contribution=[
                float(index == answer) for index in range(count)
            ],
            novelty=[1.0] * count,
            memory_keep=[0.0] * count,
            recovery=[0] * count,
            dependency=zeros,
            residual_route=zeros,
            global_labels={
                "branch_count": count,
                "initial_wave_size": min(8, count),
                "later_wave_size": 1,
                "wave_count": 1,
                "task_complete": 0.0,
                "task_solvable": 1.0,
                "must_report_failure": 0.0,
                "takeover": 0.0,
                "halt": 0.0,
                "remaining_turns": 1.0,
            },
            expert_target=expert_profile(
                planning=1.0, diversity=1.0, scheduling=0.4, integration=0.5
            ),
            masks={"coherence": [0.0] * count, "recovery": [0.0] * count},
            metadata={"answer": answer, "split": "auxiliary_train"},
        )
        resolved = json.loads(json.dumps(base))
        for index, candidate in enumerate(resolved):
            accepted = index == answer
            candidate["evidence"] = (
                "Independent verifier accepted this option."
                if accepted
                else "Independent verifier rejected this option."
            )
            candidate["runtime"].update(
                {
                    "observed": True,
                    "success": accepted,
                    "failure_type": "success" if accepted else "other_failure",
                }
            )
        yield event_row(
            task_id=f"{group}::resolved",
            group_id=group,
            source_type="mmlu",
            domain=str(source["subject"]),
            query=query,
            candidates=resolved,
            admission=[1.0] * count,
            priority=[float(index == answer) for index in range(count)],
            stage=[1] * count,
            coherence=[
                float(index == answer) for index in range(count)
            ],
            contribution=[
                float(index == answer) for index in range(count)
            ],
            novelty=[1.0] * count,
            memory_keep=[
                float(index == answer) for index in range(count)
            ],
            recovery=[
                RECOVERY_ID["none"]
                if index == answer
                else RECOVERY_ID["abandon_action"]
                for index in range(count)
            ],
            dependency=zeros,
            residual_route=zeros,
            global_labels={
                "branch_count": count,
                "initial_wave_size": min(8, count),
                "later_wave_size": 1,
                "wave_count": 1,
                "task_complete": 1.0,
                "task_solvable": 1.0,
                "must_report_failure": 0.0,
                "takeover": 1.0,
                "halt": 1.0,
                "remaining_turns": 0.0,
            },
            expert_target=expert_profile(
                evidence=1.2,
                halting=1.0,
                integration=1.0,
                recovery=0.4,
            ),
            metadata={"answer": answer, "split": "auxiliary_train"},
        )


def _patch_files(patch: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, re.M)
        )
    )


def swe_events(limit: int, seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        "princeton-nlp/SWE-bench_Lite", split="test"
    )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for source_index in indices[:limit]:
        source = dataset[source_index]
        files = _patch_files(str(source["patch"]))[:4]
        names = ["inspect_issue", "search_relevant_symbols", "inspect_tests"]
        descriptions = [
            "Extract the failure contract and constraints from the issue.",
            "Search the repository for symbols and call sites named in the issue.",
            "Inspect failing and regression tests before editing.",
        ]
        for file_index, path in enumerate(files):
            names.append(f"patch_{file_index}")
            descriptions.append(f"Prepare a minimal patch for {path}.")
        names.extend(
            ["run_targeted_tests", "run_regression_tests", "summarize_patch"]
        )
        descriptions.extend(
            [
                "Run FAIL_TO_PASS and directly affected tests.",
                "Run PASS_TO_PASS regression tests.",
                "Summarize changed behavior, evidence, and unresolved risk.",
            ]
        )
        selected_count = len(names)
        names.extend(["edit_unrelated_file", "skip_tests_and_finish"])
        descriptions.extend(
            [
                "Edit an unrelated file without issue evidence.",
                "Declare success without executing tests.",
            ]
        )
        candidates = _simple_candidates(names, descriptions)
        count = len(candidates)
        dependency = empty_square(count)
        residual = empty_square(count)
        index = {name: offset for offset, name in enumerate(names)}

        def edge(target: str, source_name: str) -> None:
            dependency[index[target]][index[source_name]] = 1.0
            residual[index[target]][index[source_name]] = 1.0

        edge("search_relevant_symbols", "inspect_issue")
        edge("inspect_tests", "inspect_issue")
        patch_names = [name for name in names if name.startswith("patch_")]
        for patch_name in patch_names:
            edge(patch_name, "search_relevant_symbols")
            edge(patch_name, "inspect_tests")
        for patch_name in patch_names or ["inspect_tests"]:
            edge("run_targeted_tests", patch_name)
        edge("run_regression_tests", "run_targeted_tests")
        edge("summarize_patch", "run_regression_tests")
        stage_map = {
            "inspect_issue": 1,
            "search_relevant_symbols": 2,
            "inspect_tests": 2,
            **{name: 3 for name in patch_names},
            "run_targeted_tests": 4,
            "run_regression_tests": 5,
            "summarize_patch": 6,
        }
        admission = [
            float(offset < selected_count) for offset in range(count)
        ]
        stages = [stage_map.get(name, 0) for name in names]
        group = f"swe::{source['instance_id']}"
        query = str(source["problem_statement"])
        yield event_row(
            task_id=f"{group}::plan",
            group_id=group,
            source_type="swe_bench",
            domain=str(source["repo"]),
            query=query,
            candidates=candidates,
            admission=admission,
            priority=[
                (7 - stage) / 6 if stage else 0.0 for stage in stages
            ],
            stage=stages,
            coherence=[0.0] * count,
            contribution=admission,
            novelty=admission,
            memory_keep=[0.0] * count,
            recovery=[0] * count,
            dependency=dependency,
            residual_route=residual,
            global_labels={
                "branch_count": selected_count,
                "initial_wave_size": 1,
                "later_wave_size": max(1, min(4, len(patch_names))),
                "wave_count": 6,
                "task_complete": 0.0,
                "task_solvable": 1.0,
                "must_report_failure": 0.0,
                "takeover": 0.0,
                "halt": 0.0,
                "remaining_turns": 6.0,
            },
            expert_target=expert_profile(
                planning=1.0,
                scheduling=1.3,
                diversity=0.5,
                memory=1.0,
                integration=0.7,
            ),
            masks={"coherence": [0.0] * count, "recovery": [0.0] * count},
            metadata={
                "base_commit": source["base_commit"],
                "patch_files": files,
                "fail_to_pass": source["FAIL_TO_PASS"],
                "pass_to_pass": source["PASS_TO_PASS"],
            },
        )
        resolved = json.loads(json.dumps(candidates))
        observed_mask = []
        for offset, candidate in enumerate(resolved):
            selected = offset < selected_count
            observed_mask.append(float(selected))
            if selected:
                candidate["evidence"] = (
                    "Gold patch trajectory completed this action and preserved "
                    "the listed FAIL_TO_PASS/PASS_TO_PASS contract."
                )
                candidate["runtime"].update(
                    {
                        "observed": True,
                        "success": True,
                        "failure_type": "success",
                    }
                )
        yield event_row(
            task_id=f"{group}::resolved",
            group_id=group,
            source_type="swe_bench",
            domain=str(source["repo"]),
            query=query,
            candidates=resolved,
            admission=admission,
            priority=admission,
            stage=stages,
            coherence=admission,
            contribution=admission,
            novelty=admission,
            memory_keep=[
                float(
                    offset < selected_count
                    and (
                        names[offset].startswith("patch_")
                        or names[offset]
                        in {
                            "inspect_issue",
                            "inspect_tests",
                            "run_targeted_tests",
                            "run_regression_tests",
                            "summarize_patch",
                        }
                    )
                )
                for offset in range(count)
            ],
            recovery=[0] * count,
            dependency=dependency,
            residual_route=residual,
            global_labels={
                "branch_count": selected_count,
                "initial_wave_size": 1,
                "later_wave_size": max(1, min(4, len(patch_names))),
                "wave_count": 6,
                "task_complete": 1.0,
                "task_solvable": 1.0,
                "must_report_failure": 0.0,
                "takeover": 1.0,
                "halt": 1.0,
                "remaining_turns": 0.0,
            },
            expert_target=expert_profile(
                evidence=1.0,
                memory=0.8,
                halting=1.0,
                integration=1.2,
            ),
            masks={
                "coherence": observed_mask,
                "recovery": observed_mask,
            },
            metadata={
                "base_commit": source["base_commit"],
                "patch_files": files,
                "gold_completion": True,
            },
        )


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]{3,}", value.lower()))


def _context_chunks(context: str, count: int = 6, width: int = 1800) -> list[str]:
    if len(context) <= width:
        return [context]
    positions = [
        round(index * max(0, len(context) - width) / max(count - 1, 1))
        for index in range(count)
    ]
    return [context[position : position + width] for position in positions]


def longbench_events(limit: int, seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for source_index in indices[:limit]:
        source = dataset[source_index]
        choices = [source[f"choice_{letter}"] for letter in "ABCD"]
        answer = "ABCD".index(str(source["answer"]).strip().upper())
        chunks = _context_chunks(str(source["context"]))
        target_terms = _tokens(str(source["question"]) + " " + choices[answer])
        scores = [
            len(target_terms & _tokens(chunk)) / math.sqrt(max(len(_tokens(chunk)), 1))
            for chunk in chunks
        ]
        relevant = set(
            sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)[:2]
        )
        names = [f"retrieve_chunk_{index}" for index in range(len(chunks))]
        descriptions = [
            f"Retrieve and retain context region {index}: {chunk[:500]}"
            for index, chunk in enumerate(chunks)
        ]
        names += [f"verify_option_{letter}" for letter in "ABCD"]
        descriptions += [
            f"Test option {letter} against retained evidence: {choice}"
            for letter, choice in zip("ABCD", choices)
        ]
        names.append("aggregate_supported_answer")
        descriptions.append(
            "Aggregate only supported evidence and choose the final option."
        )
        candidates = _simple_candidates(names, descriptions)
        count = len(candidates)
        dependency = empty_square(count)
        residual = empty_square(count)
        verify_start = len(chunks)
        aggregate = count - 1
        for option_index in range(4):
            for chunk_index in relevant:
                dependency[verify_start + option_index][chunk_index] = 1.0
                residual[verify_start + option_index][chunk_index] = 1.0
            dependency[aggregate][verify_start + option_index] = 1.0
            residual[aggregate][verify_start + option_index] = 1.0
        admission = [
            float(index in relevant)
            for index in range(len(chunks))
        ] + [1.0] * 5
        stages = [1] * len(chunks) + [2] * 4 + [3]
        group = f"longbench::{source['_id']}"
        query = str(source["question"])
        yield event_row(
            task_id=f"{group}::plan",
            group_id=group,
            source_type="longbench",
            domain=str(source["domain"]),
            query=query,
            candidates=candidates,
            admission=admission,
            priority=[
                scores[index] if index < len(chunks) else 0.8
                for index in range(count)
            ],
            stage=stages,
            coherence=[0.0] * count,
            contribution=[
                float(index in relevant) for index in range(len(chunks))
            ]
            + [float(index == answer) for index in range(4)]
            + [1.0],
            novelty=[1.0] * count,
            memory_keep=[
                float(index in relevant) for index in range(len(chunks))
            ]
            + [0.0] * 5,
            recovery=[0] * count,
            dependency=dependency,
            residual_route=residual,
            global_labels={
                "branch_count": int(sum(admission)),
                "initial_wave_size": min(8, max(1, len(relevant))),
                "later_wave_size": 4,
                "wave_count": 3,
                "task_complete": 0.0,
                "task_solvable": 1.0,
                "must_report_failure": 0.0,
                "takeover": 0.0,
                "halt": 0.0,
                "remaining_turns": 3.0,
            },
            expert_target=expert_profile(
                planning=0.7,
                scheduling=0.8,
                diversity=0.8,
                evidence=0.8,
                memory=1.5,
                integration=0.8,
            ),
            masks={"coherence": [0.0] * count, "recovery": [0.0] * count},
            metadata={
                "answer": answer,
                "difficulty": source["difficulty"],
                "length": source["length"],
                "sub_domain": source["sub_domain"],
            },
        )
        resolved = json.loads(json.dumps(candidates))
        coherence = []
        observed_mask = []
        recovery = []
        memory_keep = []
        for index, candidate in enumerate(resolved):
            if index < len(chunks):
                observed = index in relevant
                accepted = observed
                if observed:
                    candidate["evidence"] = chunks[index]
            elif index < verify_start + 4:
                observed = True
                option_index = index - verify_start
                accepted = option_index == answer
                candidate["evidence"] = (
                    "Retained evidence supports this option."
                    if accepted
                    else "Retained evidence contradicts or fails to support this option."
                )
            else:
                observed = True
                accepted = True
                candidate["evidence"] = (
                    f"Aggregator selected option {'ABCD'[answer]} from supported evidence."
                )
            candidate["runtime"].update(
                {
                    "observed": observed,
                    "success": accepted,
                    "failure_type": (
                        "success" if accepted else "other_failure"
                    ),
                }
            )
            observed_mask.append(float(observed))
            coherence.append(float(accepted))
            recovery.append(
                RECOVERY_ID["none"]
                if accepted or not observed
                else RECOVERY_ID["abandon_action"]
            )
            memory_keep.append(
                float(
                    (index < len(chunks) and index in relevant)
                    or index == verify_start + answer
                    or index == aggregate
                )
            )
        yield event_row(
            task_id=f"{group}::resolved",
            group_id=group,
            source_type="longbench",
            domain=str(source["domain"]),
            query=query,
            candidates=resolved,
            admission=admission,
            priority=[
                float(index in relevant)
                if index < len(chunks)
                else float(index == verify_start + answer or index == aggregate)
                for index in range(count)
            ],
            stage=stages,
            coherence=coherence,
            contribution=memory_keep,
            novelty=[1.0] * count,
            memory_keep=memory_keep,
            recovery=recovery,
            dependency=dependency,
            residual_route=residual,
            global_labels={
                "branch_count": int(sum(admission)),
                "initial_wave_size": min(8, max(1, len(relevant))),
                "later_wave_size": 4,
                "wave_count": 3,
                "task_complete": 1.0,
                "task_solvable": 1.0,
                "must_report_failure": 0.0,
                "takeover": 1.0,
                "halt": 1.0,
                "remaining_turns": 0.0,
            },
            expert_target=expert_profile(
                evidence=1.3,
                memory=1.0,
                halting=1.0,
                integration=1.2,
                recovery=0.4,
            ),
            masks={
                "coherence": observed_mask,
                "recovery": observed_mask,
            },
            metadata={
                "answer": answer,
                "difficulty": source["difficulty"],
                "length": source["length"],
                "sub_domain": source["sub_domain"],
            },
        )


def clawbench_events(limit: int, seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/ClawBench", split="test")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for source_index in indices[:limit]:
        source = dataset[source_index]
        names = [
            "inspect_constraints",
            "open_target_site",
            "load_required_private_context",
            "authenticate_if_needed",
            "navigate_task_flow",
            "validate_before_commit",
            "submit_action",
            "verify_server_receipt",
            "report_grounded_result",
            "bypass_confirmation",
            "fabricate_success",
        ]
        descriptions = [
            "Parse the instruction, time limit, allowed sites, and success contract.",
            f"Open only the target sites: {source['sites']}.",
            f"Load task-scoped context: {source['extra_info']} {source['shared_info']}.",
            "Authenticate only when the target flow requires it.",
            f"Navigate the {source['platform']} flow required by the instruction.",
            "Validate fields, side effects, and user constraints before committing.",
            f"Submit the action matching evaluator contract: {source['eval_schema']}.",
            "Verify the server response and observable postcondition.",
            "Report success or the exact blocked step without fabrication.",
            "Bypass confirmation and commit an unchecked side effect.",
            "Claim success without a receipt or observable postcondition.",
        ]
        candidates = _simple_candidates(names, descriptions)
        count = len(candidates)
        selected_count = count - 2
        dependency = empty_square(count)
        residual = empty_square(count)
        index = {name: offset for offset, name in enumerate(names)}

        def edge(target: str, source_name: str) -> None:
            dependency[index[target]][index[source_name]] = 1.0
            residual[index[target]][index[source_name]] = 1.0

        edge("open_target_site", "inspect_constraints")
        edge("load_required_private_context", "inspect_constraints")
        edge("authenticate_if_needed", "open_target_site")
        edge("navigate_task_flow", "open_target_site")
        edge("navigate_task_flow", "load_required_private_context")
        edge("validate_before_commit", "navigate_task_flow")
        edge("submit_action", "validate_before_commit")
        edge("verify_server_receipt", "submit_action")
        edge("report_grounded_result", "verify_server_receipt")
        stages = [1, 2, 2, 3, 3, 4, 5, 6, 7, 0, 0]
        admission = [1.0] * selected_count + [0.0, 0.0]
        group = f"clawbench::{source['task_id']}"
        yield event_row(
            task_id=f"{group}::plan",
            group_id=group,
            source_type="clawbench",
            domain=f"{source['metaclass']}/{source['class']}",
            query=str(source["instruction"]),
            candidates=candidates,
            admission=admission,
            priority=[
                (8 - stage) / 7 if stage else 0.0 for stage in stages
            ],
            stage=stages,
            coherence=[0.0] * count,
            contribution=admission,
            novelty=admission,
            memory_keep=[
                float(name in {"inspect_constraints", "load_required_private_context"})
                for name in names
            ],
            recovery=[0] * count,
            dependency=dependency,
            residual_route=residual,
            global_labels={
                "branch_count": selected_count,
                "initial_wave_size": 1,
                "later_wave_size": 2,
                "wave_count": 7,
                "task_complete": 0.0,
                "task_solvable": 1.0,
                "must_report_failure": 0.0,
                "takeover": 0.0,
                "halt": 0.0,
                "remaining_turns": 7.0,
            },
            expert_target=expert_profile(
                planning=1.0,
                scheduling=1.0,
                diversity=0.6,
                evidence=0.7,
                recovery=0.8,
                memory=0.9,
                halting=0.5,
                integration=1.0,
            ),
            masks={"coherence": [0.0] * count, "recovery": [0.0] * count},
            metadata={
                "platform": source["platform"],
                "sites": source["sites"],
                "time_limit": source["time_limit"],
                "eval_schema": source["eval_schema"],
            },
        )


def build_all(
    trajectory_source: Path,
    output: Path,
    *,
    trajectory_limit: int = 0,
    mmlu_limit: int = 1200,
    swe_limit: int = 300,
    longbench_limit: int = 250,
    clawbench_limit: int = 283,
    seed: int = 42,
) -> dict:
    rows: list[dict] = []
    trajectory = trajectory_events(trajectory_source)
    if trajectory_limit:
        for index, row in enumerate(trajectory):
            if index >= trajectory_limit:
                break
            rows.append(row)
    else:
        rows.extend(trajectory)
    rows.extend(mmlu_events(mmlu_limit, seed))
    rows.extend(swe_events(swe_limit, seed))
    rows.extend(longbench_events(longbench_limit, seed))
    rows.extend(clawbench_events(clawbench_limit, seed))
    random.Random(seed).shuffle(rows)
    return write_jsonl(output, rows)
