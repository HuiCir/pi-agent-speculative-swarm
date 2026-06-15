#!/usr/bin/env python3
"""Warm-start and jointly distill the trajectory-coupled RCG MoE OPD V1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from build_failure_aware_training_data import (
    FAILURE_TYPES,
    FAILURE_TYPE_TO_ID,
)
from dynamic_swarm_train import (
    empty_device_cache,
    load_base,
    peak_memory_gb,
    resolve_device,
    set_seed,
)
from rcg.moe_opd_controller import RcgMoeOpdV1, parameter_counts
from rcg.opd_serialization import (
    action_text,
    execution_text,
    query_text,
)


RECOVERY_TYPES = (
    "none",
    "repair_call",
    "wait_dependency",
    "abandon_tool",
    "replan_distinct_tool",
    "report_failure",
)
RECOVERY_TO_ID = {name: index for index, name in enumerate(RECOVERY_TYPES)}
FAILURE_TO_RECOVERY = {
    "success": "none",
    "timeout": "abandon_tool",
    "server_or_network": "abandon_tool",
    "auth_or_quota": "abandon_tool",
    "api_unavailable": "replan_distinct_tool",
    "empty_result": "abandon_tool",
    "invalid_call": "repair_call",
    "provenance_mismatch": "replan_distinct_tool",
    "blocked_dependency": "wait_dependency",
    "other_failure": "report_failure",
}
PHASES = ("planner", "outcome", "orchestration", "joint")


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("schema") != "rcg_moe_opd_v1":
                    raise ValueError(
                        f"{row.get('task_id')}: unexpected schema"
                    )
                rows.append(row)
    return rows


def stable_int(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8],
        "big",
    )


def trajectory_root(row: dict) -> str:
    return str(row["task_id"]).split("::snapshot_", 1)[0]


def split_rows(rows: list[dict], eval_ratio: float, seed: int):
    training = []
    evaluation = []
    for row in rows:
        trajectory_id = trajectory_root(row)
        bucket = stable_int(f"{seed}:{trajectory_id}") % 10000
        if bucket < int(eval_ratio * 10000):
            evaluation.append(row)
        else:
            training.append(row)
    if not training or not evaluation:
        raise ValueError("train/evaluation split is empty")
    return training, evaluation


def parameter_names(action: dict) -> list[str]:
    schema = action.get("parameters") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    return sorted(str(name) for name in (properties or {}))


def produced_names(action: dict, universe: list[str]) -> list[str]:
    description = str(action.get("description") or "").lower()
    output = []
    for name in universe:
        if re.search(
            rf"(?:produce|return|output|provide|yield)[a-z\s_-]{{0,32}}{re.escape(name.lower())}",
            description,
        ):
            output.append(name)
    return output


def need_text(action: dict) -> str:
    names = parameter_names(action)
    return " ; ".join(names) if names else "<NO_UNRESOLVED_INPUT>"


def produce_text(action: dict, universe: list[str]) -> str:
    names = produced_names(action, universe)
    return " ; ".join(names) if names else "<NO_DECLARED_OUTPUT>"


def all_texts(rows: list[dict]) -> dict[str, list[str]]:
    values = {
        "query": [],
        "action": [],
        "path": [],
        "need": [],
        "produce": [],
    }
    for row in rows:
        query = str(row["query"])
        universe = sorted(
            {
                name
                for candidate in row["candidates"]
                for name in parameter_names(candidate)
            }
        )
        values["query"].append(query_text(query))
        for candidate in row["candidates"]:
            values["action"].append(action_text(candidate))
            values["need"].append(need_text(candidate))
            values["produce"].append(produce_text(candidate, universe))
            if (candidate.get("execution") or {}).get("observed"):
                values["path"].append(execution_text(query, candidate))
    return {
        key: list(dict.fromkeys(texts))
        for key, texts in values.items()
    }


def cache_fingerprint(rows: list[dict], args) -> str:
    digest = hashlib.sha256()
    digest.update(str(args.encoder_mode).encode())
    digest.update(str(args.encoder_layers).encode())
    digest.update(str(args.hidden_size).encode())
    digest.update(str(args.target_path).encode())
    digest.update(str(args.query_tokens).encode())
    digest.update(str(args.action_tokens).encode())
    digest.update(str(args.path_tokens).encode())
    digest.update(str(args.parameter_tokens).encode())
    for row in rows:
        digest.update(str(row["task_id"]).encode())
        digest.update(str(row["query"]).encode())
        for candidate in row["candidates"]:
            digest.update(action_text(candidate).encode())
            if (candidate.get("execution") or {}).get("observed"):
                digest.update(
                    execution_text(row["query"], candidate).encode()
                )
    return digest.hexdigest()


@torch.inference_mode()
def qwen_encode(model, tokenizer, texts, max_length, batch_size, device):
    outputs = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = [texts[index] for index in indices]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        hidden = model.model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        mean = (
            (hidden * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )
        last_index = mask.long().sum(-1).clamp_min(1) - 1
        last = hidden[
            torch.arange(hidden.shape[0], device=device),
            last_index,
        ]
        pooled = (0.75 * mean + 0.25 * last).to(
            "cpu", dtype=torch.float16
        )
        for local_index, original_index in enumerate(indices):
            outputs[original_index] = pooled[local_index]
        del ids, mask, hidden, mean, last
    return torch.stack(outputs)


def hash_encode(texts: list[str], hidden_size: int) -> torch.Tensor:
    output = []
    for text in texts:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_int(text) % (2**63 - 1))
        vector = torch.randn(hidden_size, generator=generator)
        output.append(F.normalize(vector, dim=0).to(torch.float16))
    return torch.stack(output)


def build_or_load_cache(rows: list[dict], args, device: str):
    fingerprint = cache_fingerprint(rows, args)
    if args.cache.exists():
        payload = torch.load(
            args.cache, map_location="cpu", weights_only=False
        )
        if payload.get("fingerprint") == fingerprint:
            print(f"Loaded OPD encoder cache {args.cache}", flush=True)
            return payload
        print("Encoder cache fingerprint changed; rebuilding.", flush=True)

    texts = all_texts(rows)
    encoded = {}
    if args.encoder_mode == "qwen":
        load_args = SimpleNamespace(
            target_path=args.target_path,
            attn_implementation=args.attn_implementation,
            device=device,
        )
        base, tokenizer = load_base(load_args)
        original_layers = len(base.model.layers)
        base.model.layers = torch.nn.ModuleList(
            list(base.model.layers[: args.encoder_layers])
        )
        base.config.num_hidden_layers = args.encoder_layers
        args.hidden_size = int(base.config.hidden_size)
        print(
            f"Encoding OPD data with frozen Qwen layers "
            f"{original_layers}->{args.encoder_layers}",
            flush=True,
        )
        for key, items in texts.items():
            length = (
                args.query_tokens
                if key == "query"
                else args.path_tokens
                if key == "path"
                else args.parameter_tokens
                if key in {"need", "produce"}
                else args.action_tokens
            )
            encoded[key] = qwen_encode(
                base,
                tokenizer,
                items,
                length,
                (
                    args.path_encode_batch_size
                    if key == "path"
                    else args.encode_batch_size
                ),
                device,
            )
            print(f"Encoded {key}: {len(items)}", flush=True)
        del base, tokenizer
        gc.collect()
        empty_device_cache(torch.device(device))
    else:
        for key, items in texts.items():
            encoded[key] = hash_encode(items, args.hidden_size)

    payload = {
        "fingerprint": fingerprint,
        "hidden_size": args.hidden_size,
        "texts": texts,
        "encoded": encoded,
    }
    atomic_torch_save(args.cache, payload)
    return payload


class EncoderCache:
    def __init__(self, payload):
        self.encoded = payload["encoded"]
        self.indices = {
            key: {text: index for index, text in enumerate(texts)}
            for key, texts in payload["texts"].items()
        }

    def get(self, key: str, texts: list[str], device):
        indices = torch.tensor(
            [self.indices[key][text] for text in texts],
            dtype=torch.long,
        )
        return self.encoded[key].index_select(0, indices).to(
            device=device,
            dtype=torch.float32,
        )


def row_tensors(row: dict, cache: EncoderCache, device):
    query = str(row["query"])
    candidates = row["candidates"]
    universe = sorted(
        {
            name
            for candidate in candidates
            for name in parameter_names(candidate)
        }
    )
    outcome_indices = [
        index
        for index, candidate in enumerate(candidates)
        if (candidate.get("execution") or {}).get("observed")
    ]
    if not outcome_indices:
        raise ValueError(f"{row['task_id']}: no observed outcome paths")
    outcome_candidates = [candidates[index] for index in outcome_indices]
    return {
        "query": cache.get("query", [query_text(query)], device),
        "action": cache.get(
            "action", [action_text(item) for item in candidates], device
        ),
        "path": cache.get(
            "path",
            [execution_text(query, item) for item in outcome_candidates],
            device,
        ),
        "need": cache.get(
            "need", [need_text(item) for item in candidates], device
        ),
        "produce": cache.get(
            "produce",
            [produce_text(item, universe) for item in candidates],
            device,
        ),
        "outcome_indices": torch.tensor(
            outcome_indices, dtype=torch.long, device=device
        ),
    }


def target_tensors(row: dict, device):
    candidates = row["candidates"]
    names = [str(candidate["tool_name"]) for candidate in candidates]
    name_to_index = {name: index for index, name in enumerate(names)}
    selection_probability = torch.tensor(
        [
            float(candidate["teacher"]["selection_probability"])
            for candidate in candidates
        ],
        device=device,
    )
    selected = torch.tensor(
        [
            bool(candidate["teacher"]["selected"])
            for candidate in candidates
        ],
        dtype=torch.bool,
        device=device,
    )
    dependency = torch.zeros(
        len(candidates), len(candidates), device=device
    )
    for target, candidate in enumerate(candidates):
        for source_name in candidate["teacher"]["depends_on_tools"]:
            if source_name in name_to_index:
                dependency[target, name_to_index[source_name]] = 1.0
    failure = []
    recovery = []
    for candidate in candidates:
        calls = (candidate.get("execution") or {}).get("calls") or []
        failure_name = (
            str(calls[-1].get("failure_type") or "other_failure")
            if calls
            else "other_failure"
        )
        failure.append(FAILURE_TYPE_TO_ID.get(failure_name, 9))
        recovery.append(
            RECOVERY_TO_ID[
                FAILURE_TO_RECOVERY.get(
                    failure_name, "report_failure"
                )
            ]
        )
    teacher = row["teacher"]
    return {
        "selection_probability": selection_probability,
        "selected": selected,
        "dependency": dependency,
        "ready": torch.tensor(
            [
                float(candidate["teacher"]["initial_ready"])
                for candidate in candidates
            ],
            device=device,
        ),
        "stage": torch.tensor(
            [
                min(
                    int(candidate["teacher"]["stage"]),
                    int(teacher["wave_count"]),
                )
                for candidate in candidates
            ],
            dtype=torch.long,
            device=device,
        ),
        "coherence": torch.tensor(
            [
                float(candidate["teacher"]["coherence"])
                for candidate in candidates
            ],
            device=device,
        ),
        "contribution": torch.tensor(
            [
                float(candidate["teacher"]["contribution"])
                for candidate in candidates
            ],
            device=device,
        ),
        "continuation": torch.tensor(
            [
                float(candidate["teacher"]["continuation"])
                for candidate in candidates
            ],
            device=device,
        ),
        "failure": torch.tensor(
            failure, dtype=torch.long, device=device
        ),
        "recovery": torch.tensor(
            recovery, dtype=torch.long, device=device
        ),
        "branch_count": torch.tensor(
            int(teacher["branch_count"]), dtype=torch.long, device=device
        ),
        "initial_wave": torch.tensor(
            int(teacher["initial_wave_size"]),
            dtype=torch.long,
            device=device,
        ),
        "later_wave": torch.tensor(
            int(teacher["later_wave_size"]),
            dtype=torch.long,
            device=device,
        ),
        "wave_count": torch.tensor(
            int(teacher["wave_count"]), dtype=torch.long, device=device
        ),
        "task_complete": torch.tensor(
            float(teacher["task_complete"]), device=device
        ),
        "task_solvable": torch.tensor(
            float(teacher["task_solvable"]), device=device
        ),
        "task_solvability_known": bool(
            teacher.get("task_solvability_known", True)
        ),
        "takeover": torch.tensor(
            float(teacher["takeover"]), device=device
        ),
        "must_report_failure": torch.tensor(
            float(teacher["must_report_failure"]), device=device
        ),
        "teacher_weight": float(row.get("teacher_weight", 1.0)),
    }


def outcome_targets(targets: dict, indices: torch.Tensor) -> dict:
    vector_keys = (
        "selected",
        "coherence",
        "contribution",
        "continuation",
        "failure",
        "recovery",
    )
    output = {
        key: targets[key].index_select(0, indices)
        for key in vector_keys
    }
    output["dependency"] = targets["dependency"].index_select(
        0, indices
    ).index_select(1, indices)
    return output


def soft_binary_kl(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)


def balanced_binary_loss(logits, targets):
    positive = targets >= 0.5
    negative = ~positive
    losses = []
    if positive.any():
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[positive], targets[positive]
            )
        )
    if negative.any():
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[negative], targets[negative]
            )
        )
    return sum(losses) / max(len(losses), 1)


def shared_alignment_loss(plan: dict, targets: dict):
    query = F.normalize(plan["shared_query"], dim=-1, eps=1e-6)
    actions = F.normalize(
        plan["shared_actions"], dim=-1, eps=1e-6
    )
    logits = 8.0 * (query @ actions.T)
    return balanced_binary_loss(
        logits, targets["selection_probability"]
    )


def planner_loss(plan: dict, targets: dict):
    selected = targets["selected"]
    selection_kl = soft_binary_kl(
        plan["relevance"], targets["selection_probability"]
    )
    admission_kl = soft_binary_kl(
        plan["admission"], targets["selection_probability"]
    )
    teacher_distribution = selected.float()
    teacher_distribution = (
        teacher_distribution / teacher_distribution.sum().clamp_min(1.0)
    )
    set_kl = F.kl_div(
        F.log_softmax(plan["relevance"], dim=-1),
        teacher_distribution,
        reduction="batchmean",
    )
    admission_set_kl = F.kl_div(
        F.log_softmax(plan["admission"], dim=-1),
        teacher_distribution,
        reduction="batchmean",
    )
    dependency_loss = balanced_binary_loss(
        plan["dependency_logits"].flatten(),
        targets["dependency"].flatten(),
    )
    readiness_loss = balanced_binary_loss(
        plan["readiness"], targets["ready"]
    )
    stage_loss = F.cross_entropy(
        plan["stage_logits"], targets["stage"]
    )
    branch_count_loss = F.cross_entropy(
        plan["branch_count_logits"].unsqueeze(0),
        targets["branch_count"].unsqueeze(0),
    )
    initial_wave_loss = F.cross_entropy(
        plan["initial_wave_size_logits"].unsqueeze(0),
        targets["initial_wave"].unsqueeze(0),
    )
    later_wave_loss = F.cross_entropy(
        plan["later_wave_size_logits"].unsqueeze(0),
        targets["later_wave"].unsqueeze(0),
    )
    wave_count_loss = F.cross_entropy(
        plan["wave_count_logits"].unsqueeze(0),
        targets["wave_count"].unsqueeze(0),
    )
    budget_loss = F.smooth_l1_loss(
        plan["admission"].sigmoid().sum(),
        targets["branch_count"].float(),
    )
    count_values = torch.arange(
        plan["branch_count_logits"].numel(),
        dtype=plan["branch_count_logits"].dtype,
        device=plan["branch_count_logits"].device,
    )
    expected_count = (
        plan["branch_count_logits"].softmax(-1) * count_values
    ).sum()
    count_regression_loss = F.smooth_l1_loss(
        expected_count, targets["branch_count"].float()
    )
    count_agreement_loss = F.smooth_l1_loss(
        expected_count, plan["admission"].sigmoid().sum()
    )
    positives = plan["relevance"][selected]
    negatives = plan["relevance"][~selected]
    margin_loss = (
        F.relu(1.0 - positives[:, None] + negatives[None, :]).mean()
        if positives.numel() and negatives.numel()
        else selection_kl * 0.0
    )
    admission_positives = plan["admission"][selected]
    admission_negatives = plan["admission"][~selected]
    admission_margin_loss = (
        F.relu(
            1.0
            - admission_positives[:, None]
            + admission_negatives[None, :]
        ).mean()
        if admission_positives.numel() and admission_negatives.numel()
        else admission_kl * 0.0
    )
    selection_agreement_loss = F.smooth_l1_loss(
        plan["admission"].sigmoid(),
        plan["relevance"].sigmoid(),
    )
    confidence_loss = F.binary_cross_entropy_with_logits(
        plan["plan_confidence"], torch.ones_like(plan["plan_confidence"])
    )
    loss = (
        1.3 * selection_kl
        + 1.4 * admission_kl
        + 1.0 * set_kl
        + 0.8 * admission_set_kl
        + 0.8 * dependency_loss
        + 0.55 * readiness_loss
        + 0.7 * stage_loss
        + 1.1 * branch_count_loss
        + 0.8 * initial_wave_loss
        + 0.45 * later_wave_loss
        + 0.55 * wave_count_loss
        + 0.55 * budget_loss
        + 0.45 * count_regression_loss
        + 0.35 * count_agreement_loss
        + 0.45 * margin_loss
        + 0.65 * admission_margin_loss
        + 0.35 * selection_agreement_loss
        + 0.1 * confidence_loss
    )
    return loss, {
        "selection_kl": float(selection_kl.detach()),
        "admission_kl": float(admission_kl.detach()),
        "set_kl": float(set_kl.detach()),
        "admission_set_kl": float(admission_set_kl.detach()),
        "dependency_loss": float(dependency_loss.detach()),
        "stage_loss": float(stage_loss.detach()),
        "branch_count_loss": float(branch_count_loss.detach()),
        "count_regression_loss": float(count_regression_loss.detach()),
        "count_agreement_loss": float(count_agreement_loss.detach()),
        "selection_agreement_loss": float(
            selection_agreement_loss.detach()
        ),
        "initial_wave_loss": float(initial_wave_loss.detach()),
        "later_wave_loss": float(later_wave_loss.detach()),
        "wave_count_loss": float(wave_count_loss.detach()),
    }


def outcome_loss(outcome: dict, targets: dict):
    count = targets["selected"].numel()
    indices = torch.arange(count, device=targets["selected"].device)
    assignment_loss = 0.5 * (
        F.cross_entropy(outcome["assignment_logits"], indices)
        + F.cross_entropy(outcome["assignment_logits"].T, indices)
    )
    coherence_loss = balanced_binary_loss(
        outcome["coherence"], targets["coherence"]
    )
    contribution_loss = balanced_binary_loss(
        outcome["contribution"], targets["contribution"]
    )
    selected = targets["selected"]
    failure_loss = (
        F.cross_entropy(
            outcome["failure_type"][selected],
            targets["failure"][selected],
        )
        if selected.any()
        else assignment_loss * 0.0
    )
    recovery_loss = (
        F.cross_entropy(
            outcome["recovery"][selected],
            targets["recovery"][selected],
        )
        if selected.any()
        else assignment_loss * 0.0
    )
    continuation_loss = balanced_binary_loss(
        outcome["continuation"], targets["continuation"]
    )
    loss = (
        0.55 * assignment_loss
        + 1.25 * coherence_loss
        + 1.1 * contribution_loss
        + 0.7 * failure_loss
        + 0.55 * recovery_loss
        + 0.5 * continuation_loss
    )
    return loss, {
        "assignment_loss": float(assignment_loss.detach()),
        "coherence_loss": float(coherence_loss.detach()),
        "contribution_loss": float(contribution_loss.detach()),
        "failure_loss": float(failure_loss.detach()),
        "recovery_loss": float(recovery_loss.detach()),
        "continuation_loss": float(continuation_loss.detach()),
    }


def orchestration_loss(output: dict, targets: dict):
    outcome = output["outcome"]
    route = output["route"]
    complete_loss = F.binary_cross_entropy_with_logits(
        outcome["task_complete"], targets["task_complete"]
    )
    solvability_loss = (
        F.binary_cross_entropy_with_logits(
            outcome["task_solvability"], targets["task_solvable"]
        )
        if targets["task_solvability_known"]
        else outcome["task_solvability"] * 0.0
    )
    takeover_loss = F.binary_cross_entropy_with_logits(
        outcome["takeover"], targets["takeover"]
    )
    report_loss = F.binary_cross_entropy_with_logits(
        outcome["must_report_failure"],
        targets["must_report_failure"],
    )
    initial_route = output["model"].router_forward(
        output["query_hidden"],
        output["action_hidden"],
        None,
    )
    initial_route_loss = F.cross_entropy(
        initial_route["router_logits"].unsqueeze(0),
        torch.tensor(
            [0],
            dtype=torch.long,
            device=targets["selected"].device,
        ),
    )
    post_route_loss = F.cross_entropy(
        route["router_logits"].unsqueeze(0),
        torch.tensor(
            [2 if targets["task_complete"] >= 0.5 else 1],
            dtype=torch.long,
            device=targets["selected"].device,
        ),
    )
    consistency = (
        F.relu(outcome["takeover"].sigmoid() - outcome["task_complete"].sigmoid())
        + F.relu(
            outcome["must_report_failure"].sigmoid()
            + outcome["task_solvability"].sigmoid()
            - 1.0
        )
    )
    loss = (
        1.25 * complete_loss
        + 0.9 * solvability_loss
        + 1.0 * takeover_loss
        + 0.8 * report_loss
        + 0.35 * initial_route_loss
        + 0.5 * post_route_loss
        + 0.5 * consistency
    )
    return loss, {
        "task_complete_loss": float(complete_loss.detach()),
        "task_solvability_loss": float(solvability_loss.detach()),
        "takeover_loss": float(takeover_loss.detach()),
        "must_report_failure_loss": float(report_loss.detach()),
        "initial_route_loss": float(initial_route_loss.detach()),
        "post_route_loss": float(post_route_loss.detach()),
        "orchestration_consistency": float(consistency.detach()),
    }


def trajectory_loss(model, row, tensors, targets, phase):
    outcome_indices = tensors["outcome_indices"]
    count = int(outcome_indices.numel())
    critic_targets = outcome_targets(targets, outcome_indices)
    path_mask = torch.ones(
        count, 1, dtype=torch.bool, device=tensors["action"].device
    )
    output = model.trajectory_forward(
        tensors["action"],
        tensors["query"],
        tensors["path"],
        path_mask,
        outcome_action_hidden=tensors["action"].index_select(
            0, outcome_indices
        ),
        need_hidden=tensors["need"],
        produce_hidden=tensors["produce"],
        need_mask=torch.tensor(
            [bool(parameter_names(item)) for item in row["candidates"]],
            dtype=torch.bool,
            device=tensors["action"].device,
        ),
        ancestor_matrix=critic_targets["dependency"],
    )
    # These references let orchestration_loss evaluate the initial router
    # without duplicating the complete trajectory forward.
    output["model"] = model
    output["query_hidden"] = tensors["query"]
    output["action_hidden"] = tensors["action"]

    plan_part, plan_metrics = planner_loss(output["plan"], targets)
    outcome_part, outcome_metrics = outcome_loss(
        output["outcome"], critic_targets
    )
    orchestration_part, orchestration_metrics = orchestration_loss(
        output, targets
    )
    shared_part = shared_alignment_loss(output["plan"], targets)

    if phase == "planner":
        loss = plan_part + 0.75 * shared_part
    elif phase == "outcome":
        loss = outcome_part + 0.75 * shared_part
    elif phase == "orchestration":
        loss = orchestration_part + 0.75 * shared_part
    else:
        stage_probability = F.softmax(
            output["plan"]["stage_logits"], dim=-1
        )[:, 1]
        predicted_initial = (
            output["plan"]["admission"].sigmoid() * stage_probability
        ).sum()
        schedule_consistency = F.smooth_l1_loss(
            predicted_initial,
            targets["initial_wave"].float(),
        )
        retained_probability = (
            output["outcome"]["coherence"].sigmoid()
            * output["outcome"]["contribution"].sigmoid()
        )
        execution_coverage = F.smooth_l1_loss(
            retained_probability.sum(),
            critic_targets["contribution"].sum(),
        )
        loss = (
            plan_part
            + outcome_part
            + orchestration_part
            + 1.0 * shared_part
            + 0.65 * schedule_consistency
            + 0.65 * execution_coverage
        )
        orchestration_metrics.update(
            {
                "schedule_consistency": float(
                    schedule_consistency.detach()
                ),
                "execution_coverage": float(
                    execution_coverage.detach()
                ),
            }
        )
    loss = loss * targets["teacher_weight"]
    return loss, {
        **plan_metrics,
        **outcome_metrics,
        **orchestration_metrics,
        "shared_alignment_loss": float(shared_part.detach()),
        "teacher_weight": targets["teacher_weight"],
    }, output


def schedule_rows(
    training: list[dict],
    steps: int,
    seed: int,
    base_teacher_repeat: int,
    base_teacher_fraction: float,
    phase_mode: str = "curriculum",
):
    base_rows = [
        row
        for row in training
        if str(row.get("teacher_source", "")).startswith(
            "successful_qwen_base_trajectory"
        )
    ]
    base_rows_by_root = {}
    for row in base_rows:
        base_rows_by_root.setdefault(trajectory_root(row), []).append(row)
    gold_rows = [row for row in training if row not in base_rows]
    expanded = list(training)
    for _ in range(max(0, base_teacher_repeat - 1)):
        expanded.extend(base_rows)
    phase_rows = {}
    for phase_index, phase in enumerate(PHASES):
        rows = list(expanded)
        random.Random(seed + 1000 * phase_index).shuffle(rows)
        phase_rows[phase] = rows
    if not steps:
        return [
            (phase, row)
            for phase in PHASES
            for row in phase_rows[phase]
        ]

    if phase_mode == "curriculum":
        # Three 1/7 specialist warm starts followed by a 4/7 joint phase.
        warm_steps = max(1, steps // 7)
        allocation = {
            "planner": warm_steps,
            "outcome": warm_steps,
            "orchestration": warm_steps,
            "joint": max(0, steps - 3 * warm_steps),
        }
    elif phase_mode in PHASES:
        allocation = {
            phase: steps if phase == phase_mode else 0
            for phase in PHASES
        }
    else:
        raise ValueError(f"unknown phase mode: {phase_mode}")
    schedule = []
    for phase in PHASES:
        phase_steps = allocation[phase]
        if base_rows and base_teacher_fraction > 0:
            base_count = min(
                phase_steps,
                max(
                    len(base_rows),
                    round(phase_steps * base_teacher_fraction),
                ),
            )
            gold_count = phase_steps - base_count
            base_roots = sorted(base_rows_by_root)
            shuffled_base_by_root = {
                root: list(base_rows_by_root[root]) for root in base_roots
            }
            shuffled_gold = list(gold_rows)
            base_rng = random.Random(
                seed + 1000 * PHASES.index(phase) + 17
            )
            base_rng.shuffle(base_roots)
            for root in base_roots:
                base_rng.shuffle(shuffled_base_by_root[root])
            random.Random(seed + 1000 * PHASES.index(phase) + 31).shuffle(
                shuffled_gold
            )
            rows = [
                shuffled_base_by_root[
                    base_roots[index % len(base_roots)]
                ][
                    (index // len(base_roots))
                    % len(
                        shuffled_base_by_root[
                            base_roots[index % len(base_roots)]
                        ]
                    )
                ]
                for index in range(base_count)
            ] + [
                shuffled_gold[index % len(shuffled_gold)]
                for index in range(gold_count)
            ]
            random.Random(
                seed + 1000 * PHASES.index(phase) + 53
            ).shuffle(rows)
        else:
            source = phase_rows[phase]
            rows = [
                source[index % len(source)]
                for index in range(phase_steps)
            ]
        schedule.extend((phase, row) for row in rows)
    return schedule


@torch.inference_mode()
def evaluate(model, rows, cache, device, limit=0):
    model.eval()
    rows = rows[:limit] if limit else rows
    metrics = Counter()
    for row in rows:
        tensors = row_tensors(row, cache, device)
        targets = target_tensors(row, device)
        outcome_indices = tensors["outcome_indices"]
        outcome_count = int(outcome_indices.numel())
        plan_count = len(row["candidates"])
        critic_targets = outcome_targets(targets, outcome_indices)
        output = model.trajectory_forward(
            tensors["action"],
            tensors["query"],
            tensors["path"],
            torch.ones(
                outcome_count, 1, dtype=torch.bool, device=device
            ),
            outcome_action_hidden=tensors["action"].index_select(
                0, outcome_indices
            ),
            need_hidden=tensors["need"],
            produce_hidden=tensors["produce"],
            need_mask=torch.tensor(
                [
                    bool(parameter_names(item))
                    for item in row["candidates"]
                ],
                dtype=torch.bool,
                device=device,
            ),
            ancestor_matrix=critic_targets["dependency"],
        )
        predicted_count = int(
            output["plan"]["branch_count_logits"].argmax()
        )
        selected_indices = output["plan"]["admission"].topk(
            max(1, min(predicted_count, plan_count))
        ).indices
        predicted = torch.zeros(
            plan_count, dtype=torch.bool, device=device
        )
        predicted[selected_indices] = True
        gold = targets["selected"]
        true_positive = int((predicted & gold).sum())
        metrics["precision"] += true_positive / max(
            int(predicted.sum()), 1
        )
        metrics["recall"] += true_positive / max(int(gold.sum()), 1)
        metrics["exact"] += float(torch.equal(predicted, gold))
        metrics["count"] += float(
            predicted_count == int(targets["branch_count"])
        )
        metrics["initial_wave"] += float(
            int(output["plan"]["initial_wave_size_logits"].argmax())
            == int(targets["initial_wave"])
        )
        metrics["wave_count"] += float(
            int(output["plan"]["wave_count_logits"].argmax())
            == int(targets["wave_count"])
        )
        metrics["stage"] += float(
            output["plan"]["stage_logits"].argmax(-1).eq(
                targets["stage"]
            ).float().mean()
        )
        metrics["coherence"] += float(
            output["outcome"]["coherence"].sigmoid().ge(0.5).eq(
                critic_targets["coherence"].bool()
            ).float().mean()
        )
        metrics["contribution"] += float(
            output["outcome"]["contribution"].sigmoid().ge(0.5).eq(
                critic_targets["contribution"].bool()
            ).float().mean()
        )
        metrics["failure_type"] += float(
            output["outcome"]["failure_type"].argmax(-1).eq(
                critic_targets["failure"]
            ).float().mean()
        )
        metrics["recovery"] += float(
            output["outcome"]["recovery"].argmax(-1).eq(
                critic_targets["recovery"]
            ).float().mean()
        )
        metrics["continuation"] += float(
            output["outcome"]["continuation"].sigmoid().ge(0.5).eq(
                critic_targets["continuation"].bool()
            ).float().mean()
        )
        metrics["task_complete"] += float(
            output["outcome"]["task_complete"].sigmoid().ge(0.5)
            == targets["task_complete"].bool()
        )
        if targets["task_solvability_known"]:
            metrics["task_solvable"] += float(
                output["outcome"]["task_solvability"].sigmoid().ge(0.5)
                == targets["task_solvable"].bool()
            )
            metrics["task_solvable_count"] += 1
        metrics["takeover"] += float(
            output["outcome"]["takeover"].sigmoid().ge(0.5)
            == targets["takeover"].bool()
        )
        metrics["must_report_failure"] += float(
            output["outcome"]["must_report_failure"].sigmoid().ge(0.5)
            == targets["must_report_failure"].bool()
        )
        del tensors, targets, output
    total = max(len(rows), 1)
    result = {
        key: value / total for key, value in metrics.items()
    }
    if metrics.get("task_solvable_count", 0):
        result["task_solvable"] = (
            metrics["task_solvable"]
            / metrics["task_solvable_count"]
        )
    result["composite_score"] = (
        0.14 * result.get("precision", 0.0)
        + 0.14 * result.get("recall", 0.0)
        + 0.10 * result.get("exact", 0.0)
        + 0.07 * result.get("count", 0.0)
        + 0.06 * result.get("initial_wave", 0.0)
        + 0.04 * result.get("wave_count", 0.0)
        + 0.05 * result.get("stage", 0.0)
        + 0.09 * result.get("coherence", 0.0)
        + 0.06 * result.get("contribution", 0.0)
        + 0.05 * result.get("failure_type", 0.0)
        + 0.04 * result.get("recovery", 0.0)
        + 0.03 * result.get("continuation", 0.0)
        + 0.04 * result.get("task_complete", 0.0)
        + 0.03 * result.get("task_solvable", 0.0)
        + 0.03 * result.get("takeover", 0.0)
        + 0.03 * result.get("must_report_failure", 0.0)
    )
    model.train()
    return result


def checkpoint_payload(
    model,
    optimizer,
    step,
    args,
    history,
    validation,
    best_score,
):
    return {
        "version": 13 if args.coupled_selection else 12,
        "architecture": (
            "rcg_moe_opd_v1_coupled"
            if args.coupled_selection
            else "rcg_moe_opd_v1"
        ),
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "history_tail": history[-200:],
        "validation": validation,
        "best_score": best_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("data/rcg_moe_opd_v1.jsonl")
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("cache/rcg_moe_opd_v1.pt")
    )
    parser.add_argument(
        "--target-path",
        default=os.environ.get(
            "PI_LOCAL_QWEN_MODEL", "models/qwen3-8b"
        ),
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--encoder-mode", choices=("qwen", "hash"), default="qwen"
    )
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--action-tokens", type=int, default=320)
    parser.add_argument("--path-tokens", type=int, default=3072)
    parser.add_argument("--parameter-tokens", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--path-encode-batch-size", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--shared-layers", type=int, default=6)
    parser.add_argument("--planner-layers", type=int, default=2)
    parser.add_argument("--outcome-layers", type=int, default=2)
    parser.add_argument("--orchestration-layers", type=int, default=2)
    parser.add_argument("--set-heads", type=int, default=8)
    parser.add_argument("--max-branches", type=int, default=24)
    parser.add_argument("--max-waves", type=int, default=12)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--coupled-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train admission as a relevance residual under one set objective."
        ),
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--phase-mode",
        choices=("curriculum", *PHASES),
        default="curriculum",
        help="Run the full curriculum or isolate one trainable expert stage.",
    )
    parser.add_argument("--base-teacher-repeat", type=int, default=64)
    parser.add_argument(
        "--base-teacher-fraction",
        type=float,
        default=0.2,
        help="Guaranteed successful base-trajectory fraction per phase.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=0,
        help="Optional deterministic data prefix for smoke tests.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.08)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-lr", type=float, default=1e-4)
    parser.add_argument("--expert-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=7000)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rcg_moe_opd_v1_latest.pt"),
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=Path("checkpoints/rcg_moe_opd_v1_best.pt"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/rcg_moe_opd_v1.jsonl"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("eval_results/rcg_moe_opd_v1_validation.json"),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Initialize controller weights from a compatible V12 checkpoint.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    device = torch.device(args.device)

    rows = read_rows(args.data)
    if args.row_limit:
        rows = sorted(rows, key=lambda row: row["task_id"])[
            : args.row_limit
        ]
    training, evaluation = split_rows(
        rows, args.eval_ratio, args.seed
    )
    schedule = schedule_rows(
        training,
        args.steps,
        args.seed,
        args.base_teacher_repeat,
        args.base_teacher_fraction,
        args.phase_mode,
    )
    print(
        f"OPD V1 rows={len(rows)} train={len(training)} "
        f"eval={len(evaluation)} schedule={len(schedule)}",
        flush=True,
    )
    cache_payload = build_or_load_cache(rows, args, args.device)
    cache = EncoderCache(cache_payload)
    hidden_size = int(cache_payload["hidden_size"])
    model = RcgMoeOpdV1(
        hidden_size=hidden_size,
        latent_dim=args.latent_dim,
        shared_layers=args.shared_layers,
        planner_layers=args.planner_layers,
        outcome_layers=args.outcome_layers,
        orchestration_layers=args.orchestration_layers,
        set_heads=args.set_heads,
        failure_classes=len(FAILURE_TYPES),
        recovery_classes=len(RECOVERY_TYPES),
        max_branches=args.max_branches,
        max_waves=args.max_waves,
        max_concurrency=args.max_concurrency,
        coupled_selection=args.coupled_selection,
    ).to(device)
    if args.resume_checkpoint is not None:
        resumed = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if int(resumed.get("version", 0)) < 12:
            raise ValueError("resume checkpoint is not RCG MoE OPD V1")
        model.load_state_dict(resumed["model"], strict=True)
        print(
            f"Initialized from {args.resume_checkpoint} "
            f"step={resumed.get('step')}",
            flush=True,
        )
    counts = parameter_counts(model)
    print(f"RCG-MoE-OPD V1 parameters={counts}", flush=True)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.shared_parameters()),
                "lr": args.shared_lr,
                "name": "shared",
            },
            {
                "params": list(model.planner_parameters()),
                "lr": args.expert_lr,
                "name": "planner",
            },
            {
                "params": list(model.outcome_parameters()),
                "lr": args.expert_lr,
                "name": "outcome",
            },
            {
                "params": list(model.orchestration_parameters()),
                "lr": args.expert_lr,
                "name": "orchestration",
            },
        ],
        weight_decay=args.weight_decay,
    )

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.unlink(missing_ok=True)
    history = []
    raw_validation = evaluate(
        model,
        evaluation,
        cache,
        device,
        args.eval_limit,
    )
    raw_validation["step"] = 0
    raw_validation["phase"] = "raw"
    validation_history = [raw_validation]
    atomic_json(args.validation_file, validation_history)
    print(
        f"Raw validation score={raw_validation['composite_score']:.4f} "
        f"plan={raw_validation.get('precision', 0):.4f}/"
        f"{raw_validation.get('recall', 0):.4f} "
        f"coherence={raw_validation.get('coherence', 0):.4f} "
        f"failure={raw_validation.get('failure_type', 0):.4f}",
        flush=True,
    )
    best_score = -math.inf
    started = time.time()
    current_phase = None
    for step, (phase, row) in enumerate(schedule, 1):
        if phase != current_phase:
            model.set_active_stage(phase)
            current_phase = phase
            print(f"Entering phase={phase} step={step}", flush=True)
        tensors = row_tensors(row, cache, device)
        targets = target_tensors(row, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, output = trajectory_loss(
            model, row, tensors, targets, phase
        )
        loss.backward()
        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, args.max_grad_norm
        )
        optimizer.step()
        record = {
            "step": step,
            "phase": phase,
            "task_id": row["task_id"],
            "teacher_source": row["teacher_source"],
            "candidate_count": len(row["candidates"]),
            "teacher_branch_count": row["teacher"]["branch_count"],
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "elapsed_seconds": time.time() - started,
            "peak_memory_gb": peak_memory_gb(device),
            **metrics,
        }
        history.append(record)
        append_jsonl(args.log_file, record)
        del loss, output, tensors, targets

        if step % args.log_every == 0:
            recent = history[-args.log_every :]
            print(
                f"Step {step}/{len(schedule)} phase={phase} "
                f"loss={sum(item['loss'] for item in recent) / len(recent):.4f} "
                f"grad={sum(item['grad_norm'] for item in recent) / len(recent):.3f} "
                f"peak={record['peak_memory_gb']:.2f}GB",
                flush=True,
            )
        should_validate = (
            step % args.validate_every == 0
            or step == len(schedule)
            or (
                step < len(schedule)
                and schedule[step][0] != phase
            )
        )
        validation = (
            validation_history[-1] if validation_history else {}
        )
        if should_validate:
            validation = evaluate(
                model,
                evaluation,
                cache,
                device,
                args.eval_limit,
            )
            validation["step"] = step
            validation["phase"] = phase
            validation_history.append(validation)
            atomic_json(args.validation_file, validation_history)
            print(
                f"Validation step={step} score="
                f"{validation['composite_score']:.4f} "
                f"plan={validation.get('precision', 0):.4f}/"
                f"{validation.get('recall', 0):.4f}/"
                f"{validation.get('exact', 0):.4f} "
                f"count={validation.get('count', 0):.4f} "
                f"wave={validation.get('initial_wave', 0):.4f} "
                f"complete={validation.get('task_complete', 0):.4f}",
                flush=True,
            )
            if validation["composite_score"] > best_score:
                best_score = validation["composite_score"]
                atomic_torch_save(
                    args.best_checkpoint,
                    checkpoint_payload(
                        model,
                        optimizer,
                        step,
                        args,
                        history,
                        validation,
                        best_score,
                    ),
                )
                print(
                    f"Updated best checkpoint step={step} "
                    f"score={best_score:.4f}",
                    flush=True,
                )
        should_archive = step % args.save_every == 0
        should_save_latest = (
            should_validate
            or should_archive
            or step == len(schedule)
        )
        if should_save_latest:
            atomic_torch_save(
                args.checkpoint,
                checkpoint_payload(
                    model,
                    optimizer,
                    step,
                    args,
                    history,
                    validation,
                    best_score,
                ),
            )
        if should_archive:
            archive = args.checkpoint.with_name(
                f"rcg_moe_opd_v1_step_{step}.pt"
            )
            atomic_torch_save(
                archive,
                checkpoint_payload(
                    model,
                    optimizer,
                    step,
                    args,
                    history,
                    validation,
                    best_score,
                ),
            )
        if step % 100 == 0:
            gc.collect()
            empty_device_cache(device)

    print(
        f"Finished RCG-MoE-OPD V1 steps={len(schedule)} "
        f"best={best_score:.4f} elapsed={(time.time() - started) / 3600:.2f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
