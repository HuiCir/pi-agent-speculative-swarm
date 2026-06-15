"""Train a shared-trunk RCG MoE from scratch for long-horizon agents."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from build_failure_aware_training_data import (
    FAILURE_TYPES,
    FAILURE_TYPE_TO_ID,
)
from dynamic_swarm_train import (
    balanced_bce,
    coherence_ranking_loss,
    empty_device_cache,
    load_base,
    peak_memory_gb,
    read_rows,
    resolve_device,
    select_nodes,
    set_seed,
    stable_int,
)
from rcg.dual_controller import DualTowerRCG, parameter_counts


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
FULL_DATA_PHASES = (
    "planner",
    "critic_core",
    "quality",
    "failure",
    "control",
    "joint",
)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def branch_key(task_id, branch):
    return f"{task_id}::{branch['branch_id']}"


def action_text(branch):
    return (
        "<TOOL_ACTION>\n"
        f"TOOL: {branch['tool_name']}\n"
        f"DESCRIPTION: {branch['tool_description']}\n"
        "REQUIRED_PARAMETERS: "
        f"{json.dumps(branch['required_parameters'], ensure_ascii=False)}\n"
        "OPTIONAL_PARAMETERS: "
        f"{json.dumps(branch['optional_parameters'], ensure_ascii=False)}\n"
        f"PRODUCES: {json.dumps(branch.get('produces'), ensure_ascii=False)}\n"
        "</TOOL_ACTION>"
    )


def query_text(row):
    return f"<TASK_QUERY>\n{row['query']}\n</TASK_QUERY>"


def observed_result(branch):
    replay = branch.get("live_replay") or {}
    preview = str(replay.get("live_output_preview") or "").strip()
    if preview:
        return preview
    failure = str(branch.get("failure_type") or "other_failure")
    if failure == "timeout":
        return "ERROR: tool call timed out after the bounded execution window."
    if failure == "blocked_dependency":
        blocked = ", ".join(branch.get("blocked_by") or [])
        return (
            "UPSTREAM_BLOCKED: this action could not make valid progress "
            f"because required branches failed: {blocked}."
        )
    if failure == "api_unavailable":
        return "ERROR: requested API or service is unavailable."
    if failure == "server_or_network":
        return "ERROR: provider or network failed during tool execution."
    if failure == "auth_or_quota":
        return "ERROR: authentication, subscription, or quota rejected the call."
    if failure == "invalid_call":
        return "ERROR: tool arguments do not satisfy the requested call."
    if failure == "empty_result":
        return "ERROR: tool returned an empty or unusable result."
    return str(branch.get("executed_output") or "")


def path_text(row, branch):
    return (
        "<ACTION_EXECUTION>\n"
        f"LOCAL_OBJECTIVE: {branch['tool_description']}\n"
        f"ACTUAL_TOOL: {branch['tool_name']}\n"
        "ACTUAL_PARAMETERS: "
        f"{json.dumps(branch['required_parameters'], ensure_ascii=False)}\n"
        f"RESULT: {observed_result(branch)[:2600]}\n"
        "</ACTION_EXECUTION>"
    )


def need_text(branch):
    names = sorted(
        {
            str(edge.get("parameter", "")).strip()
            for edge in branch.get("depends_on", [])
            if str(edge.get("parameter", "")).strip()
        }
    )
    return " ; ".join(names) if names else "<NO_UNRESOLVED_INPUT>"


def produce_text(branch):
    names = sorted(
        {
            str(item.get("parameter", "")).strip()
            for item in branch.get("produces", []) or []
            if str(item.get("parameter", "")).strip()
        }
    )
    return " ; ".join(names) if names else "<NO_DECLARED_OUTPUT>"


def validate_training_rows(rows):
    errors = []
    seen_tasks = set()
    service_failures = {
        "timeout",
        "server_or_network",
        "auth_or_quota",
        "api_unavailable",
        "empty_result",
        "provenance_mismatch",
        "other_failure",
    }
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen_tasks:
            errors.append(f"invalid or duplicate task_id={task_id!r}")
        seen_tasks.add(task_id)
        branches = row.get("branches") or []
        if int(row.get("branch_count_target", len(branches))) != len(branches):
            errors.append(f"{task_id}: branch_count_target mismatch")
        branch_ids = [str(branch.get("branch_id") or "") for branch in branches]
        if not all(branch_ids) or len(branch_ids) != len(set(branch_ids)):
            errors.append(f"{task_id}: invalid or duplicate branch_id")
        branch_id_set = set(branch_ids)
        graph = {branch_id: [] for branch_id in branch_ids}
        for branch in branches:
            branch_id = str(branch.get("branch_id") or "")
            failure_type = str(branch.get("failure_type") or "")
            if failure_type not in FAILURE_TYPE_TO_ID:
                errors.append(f"{task_id}/{branch_id}: unknown failure_type")
            elif int(branch.get("failure_type_id", -1)) != FAILURE_TYPE_TO_ID[failure_type]:
                errors.append(f"{task_id}/{branch_id}: failure_type_id mismatch")
            continuation = float(branch.get("continuation_target", 0.0))
            if failure_type in service_failures and continuation != 0.0:
                errors.append(
                    f"{task_id}/{branch_id}: service failure cannot continue"
                )
            if continuation and failure_type not in {
                "invalid_call",
                "blocked_dependency",
            }:
                errors.append(
                    f"{task_id}/{branch_id}: unsupported continuation target"
                )
            if float(branch.get("main_candidate_target", 0.0)) and (
                failure_type != "success"
                or not float(branch.get("effective_coherence_target", 0.0))
            ):
                errors.append(
                    f"{task_id}/{branch_id}: invalid main candidate"
                )
            for edge in branch.get("depends_on") or []:
                source = str(edge.get("source_branch_id") or "")
                if source not in branch_id_set or source == branch_id:
                    errors.append(
                        f"{task_id}/{branch_id}: invalid dependency {source!r}"
                    )
                else:
                    graph[branch_id].append(source)
        visiting = set()
        visited = set()

        def visit(node):
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            if not all(visit(parent) for parent in graph[node]):
                return False
            visiting.remove(node)
            visited.add(node)
            return True

        if not all(visit(branch_id) for branch_id in branch_ids):
            errors.append(f"{task_id}: cyclic dependency graph")
        solvable = bool(float(row.get("task_solvability_target", 0.0)))
        if bool(float(row.get("takeover_target", 0.0))) and not solvable:
            errors.append(f"{task_id}: takeover requires solvable task")
    if errors:
        preview = "\n".join(errors[:30])
        raise ValueError(
            f"training data validation failed ({len(errors)} errors):\n{preview}"
        )
    return {
        "tasks": len(rows),
        "branches": sum(len(row.get("branches") or []) for row in rows),
        "dependency_edges": sum(
            len(branch.get("depends_on") or [])
            for row in rows
            for branch in row.get("branches") or []
        ),
    }


def build_schedule(rows, steps, eval_per_stratum, seed):
    pools = defaultdict(list)
    for row in rows:
        pools[
            (
                str(row["trajectory_type"]),
                int(row["task_solvability_target"]),
            )
        ].append(row)
    evaluation = []
    training_pools = {}
    for index, key in enumerate(sorted(pools)):
        items = list(pools[key])
        random.Random(seed + index).shuffle(items)
        evaluation.extend(items[:eval_per_stratum])
        training_pools[key] = items[eval_per_stratum:]
    random.Random(seed + 20).shuffle(evaluation)

    training = [
        row
        for key in sorted(training_pools)
        for row in training_pools[key]
    ]
    schedule = []
    for phase_index, phase in enumerate(FULL_DATA_PHASES):
        segment = list(training)
        random.Random(seed + 1000 * (phase_index + 1)).shuffle(segment)
        schedule.extend(
            {"phase": phase, "task_id": row["task_id"]}
            for row in segment
        )
    if steps:
        schedule = schedule[:steps]
    return schedule, evaluation


@torch.inference_mode()
def encode_pooled(
    model,
    tokenizer,
    texts,
    max_length,
    batch_size,
    device,
    combine_last=False,
):
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        ids = tokenized["input_ids"].to(device)
        mask = tokenized["attention_mask"].to(device)
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
        if combine_last:
            last_index = mask.long().sum(-1).clamp_min(1) - 1
            last = hidden[
                torch.arange(hidden.shape[0], device=device), last_index
            ]
            mean = 0.75 * mean + 0.25 * last
        outputs.append(mean.to("cpu", dtype=torch.float16))
        del ids, mask, hidden, mean
    return torch.cat(outputs)


def training_data_fingerprint(rows_by_id, task_ids, max_nodes):
    digest = hashlib.sha256()
    for task_id in sorted(task_ids):
        row = rows_by_id[task_id]
        digest.update(task_id.encode("utf-8"))
        digest.update(query_text(row).encode("utf-8"))
        for branch in select_nodes(row, max_nodes):
            for text in (
                action_text(branch),
                path_text(row, branch),
                need_text(branch),
                produce_text(branch),
            ):
                digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def cache_signature(args, task_ids, rows_by_id):
    payload = {
        "target_path": str(args.target_path),
        "task_ids": sorted(task_ids),
        "max_nodes": args.max_nodes,
        "query_tokens": args.query_tokens,
        "action_tokens": args.action_tokens,
        "path_tokens": args.path_tokens,
        "parameter_tokens": args.parameter_tokens,
        "encoder_layers": args.encoder_layers,
        "data_fingerprint": training_data_fingerprint(
            rows_by_id, task_ids, args.max_nodes
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def encode_deduplicated(
    model,
    tokenizer,
    texts,
    max_length,
    batch_size,
    device,
    combine_last=False,
):
    unique = list(dict.fromkeys(texts))
    index = {text: position for position, text in enumerate(unique)}
    encoded = encode_pooled(
        model,
        tokenizer,
        unique,
        max_length,
        batch_size,
        device,
        combine_last,
    )
    return encoded[[index[text] for text in texts]]


def deterministic_parameter_vectors(texts, hidden_size):
    vectors = {}

    def name_vector(name):
        if name not in vectors:
            seed = int(
                hashlib.sha256(name.encode("utf-8")).hexdigest()[:16],
                16,
            ) % (2**63 - 1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            vectors[name] = F.normalize(
                torch.randn(hidden_size, generator=generator),
                dim=0,
            ).to(torch.float16)
        return vectors[name]

    output = []
    for text in texts:
        if text.startswith("<NO_"):
            output.append(torch.zeros(hidden_size, dtype=torch.float16))
            continue
        names = [item.strip() for item in text.split(";") if item.strip()]
        output.append(
            torch.stack([name_vector(name) for name in names]).mean(0)
        )
    return torch.stack(output)


def build_or_load_cache(
    args,
    rows_by_id,
    schedule,
    evaluation,
    device,
):
    task_ids = {
        item["task_id"] for item in schedule
    } | {row["task_id"] for row in evaluation}
    signature = cache_signature(args, task_ids, rows_by_id)
    cache_path = Path(args.cache)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("signature") == signature:
            print(
                f"Loaded encoder cache {cache_path} "
                f"tasks={len(cached['query_ids'])} "
                f"branches={len(cached['branch_keys'])}",
                flush=True,
            )
            return cached

    query_ids = sorted(task_ids)
    branch_items = []
    for task_id in query_ids:
        row = rows_by_id[task_id]
        branch_items.extend(
            (task_id, branch)
            for branch in select_nodes(row, args.max_nodes)
        )
    branch_items.sort(key=lambda item: branch_key(item[0], item[1]))
    branch_keys = [branch_key(task_id, branch) for task_id, branch in branch_items]
    print(
        f"Building encoder cache tasks={len(query_ids)} "
        f"branches={len(branch_keys)} on {device}.",
        flush=True,
    )
    load_args = argparse.Namespace(
        target_path=args.target_path,
        attn_implementation=args.attn_implementation,
        device=str(device),
    )
    base, tokenizer = load_base(load_args)
    if args.encoder_layers:
        original_layers = len(base.model.layers)
        base.model.layers = torch.nn.ModuleList(
            list(base.model.layers[: args.encoder_layers])
        )
        base.config.num_hidden_layers = args.encoder_layers
        print(
            f"Truncated frozen Qwen encoder layers "
            f"{original_layers}->{args.encoder_layers}.",
            flush=True,
        )
        gc.collect()
        empty_device_cache(device)
    query = encode_deduplicated(
        base,
        tokenizer,
        [query_text(rows_by_id[task_id]) for task_id in query_ids],
        args.query_tokens,
        args.encode_batch_size,
        device,
    )
    action = encode_deduplicated(
        base,
        tokenizer,
        [action_text(branch) for _, branch in branch_items],
        args.action_tokens,
        args.encode_batch_size,
        device,
    )
    path = encode_deduplicated(
        base,
        tokenizer,
        [
            path_text(rows_by_id[task_id], branch)
            for task_id, branch in branch_items
        ],
        args.path_tokens,
        args.encode_batch_size,
        device,
        combine_last=True,
    )
    hidden_size = query.shape[-1]
    need = deterministic_parameter_vectors(
        [need_text(branch) for _, branch in branch_items],
        hidden_size,
    )
    produce = deterministic_parameter_vectors(
        [produce_text(branch) for _, branch in branch_items],
        hidden_size,
    )
    cached = {
        "signature": signature,
        "query_ids": query_ids,
        "branch_keys": branch_keys,
        "query": query,
        "action": action,
        "path": path,
        "need": need,
        "produce": produce,
    }
    atomic_torch_save(cache_path, cached)
    print(
        f"Saved encoder cache {cache_path} "
        f"size={cache_path.stat().st_size / 2**30:.2f}GB.",
        flush=True,
    )
    del base, tokenizer
    gc.collect()
    empty_device_cache(device)
    return cached


class CacheView:
    def __init__(self, cache):
        self.cache = cache
        self.query_index = {
            task_id: index
            for index, task_id in enumerate(cache["query_ids"])
        }
        self.branch_index = {
            key: index
            for index, key in enumerate(cache["branch_keys"])
        }

    def query(self, task_id, device):
        return self.cache["query"][self.query_index[task_id]].to(
            device=device, dtype=torch.float32
        ).unsqueeze(0)

    def branches(self, keys, field, device):
        indices = [self.branch_index[key] for key in keys]
        return self.cache[field][indices].to(
            device=device, dtype=torch.float32
        )


def branch_signature(branch):
    return (
        branch["tool_name"],
        json.dumps(
            branch.get("required_parameters", []),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def candidate_branches(
    row,
    rows_by_id,
    train_branch_pool,
    max_nodes,
    distractors,
    step,
):
    gold = select_nodes(row, max_nodes)
    gold_keys = {branch_key(row["task_id"], branch) for branch in gold}
    gold_tools = {branch["tool_name"] for branch in gold}
    rng = random.Random(stable_int(f"{row['task_id']}:{step}:distractors"))
    candidates = list(train_branch_pool)
    rng.shuffle(candidates)
    selected_distractors = []
    for task_id, branch in candidates:
        key = branch_key(task_id, branch)
        if key in gold_keys or task_id == row["task_id"]:
            continue
        if (
            branch["tool_name"] in gold_tools
            and len(selected_distractors) < distractors // 2
        ):
            continue
        selected_distractors.append((task_id, branch))
        if len(selected_distractors) >= distractors:
            break
    combined = [(row["task_id"], branch, True) for branch in gold]
    combined.extend(
        (task_id, branch, False)
        for task_id, branch in selected_distractors
    )
    rng.shuffle(combined)
    return combined


def dependency_matrix(items, device):
    count = len(items)
    by_id = {
        branch["branch_id"]: index
        for index, (_, branch, gold) in enumerate(items)
        if gold
    }
    target = torch.zeros(count, count, device=device)
    for target_index, (_, branch, gold) in enumerate(items):
        if not gold:
            continue
        for edge in branch.get("depends_on", []):
            source = by_id.get(edge.get("source_branch_id"))
            if source is not None and source != target_index:
                target[target_index, source] = 1.0
    return target


def transitive_closure(dependency):
    closure = dependency.bool().clone()
    for middle in range(closure.shape[0]):
        closure |= closure[:, middle : middle + 1] & closure[middle : middle + 1]
    return closure.float()


def execution_state(row, items, dependency, device, step):
    count = len(items)
    levels = [0] * count
    for _ in range(count):
        changed = False
        for target in range(count):
            sources = dependency[target].nonzero().flatten().tolist()
            value = 0 if not sources else max(levels[source] + 1 for source in sources)
            if value > levels[target]:
                levels[target] = value
                changed = True
        if not changed:
            break
    maximum = max(levels, default=0)
    stage = stable_int(f"{row['task_id']}:{step}:stage") % (maximum + 1)
    completed = torch.tensor(
        [
            bool(gold and levels[index] < stage)
            for index, (_, _, gold) in enumerate(items)
        ],
        dtype=torch.bool,
        device=device,
    )
    readiness = torch.zeros(count, device=device)
    mask = torch.tensor(
        [gold for _, _, gold in items],
        dtype=torch.bool,
        device=device,
    )
    for index, (_, _, gold) in enumerate(items):
        if not gold or completed[index]:
            continue
        sources = dependency[index].bool()
        readiness[index] = float(
            not sources.any() or completed[sources].all()
        )
    return completed, readiness, mask


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = torch.as_tensor(labels).bool()
    predictions = torch.as_tensor(probabilities) >= threshold
    tp = int((predictions & labels).sum())
    tn = int((~predictions & ~labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": tp / max(tp + fp, 1),
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def masked_balanced_bce(logits, targets, mask=None):
    if mask is not None and not mask.bool().any():
        return logits.sum() * 0.0
    if mask is None and not targets.numel():
        return logits.sum() * 0.0
    return balanced_bce(logits, targets, mask)


def balanced_cross_entropy(logits, targets):
    counts = torch.bincount(
        targets, minlength=logits.shape[-1]
    ).float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = (
        targets.numel()
        / present.sum().clamp_min(1)
        / counts[present]
    )
    return F.cross_entropy(logits, targets, weight=weights)


def planner_tensors(row, items, cache, device):
    keys = [branch_key(task_id, branch) for task_id, branch, _ in items]
    return {
        "query": cache.query(row["task_id"], device),
        "action": cache.branches(keys, "action", device),
        "need": cache.branches(keys, "need", device),
        "produce": cache.branches(keys, "produce", device),
    }


def shared_loss(model, row, items, tensors):
    actions, _ = model.shared_encode_actions(tensors["action"])
    query = model.shared_encode_query(tensors["query"])
    logits = 10.0 * (
        F.normalize(query, dim=-1, eps=1e-6)
        @ F.normalize(actions, dim=-1, eps=1e-6).T
    )
    target = torch.tensor(
        [float(gold) for _, _, gold in items],
        device=logits.device,
    )
    classification = balanced_bce(logits, target)
    positives = logits[target.bool()]
    negatives = logits[~target.bool()]
    ranking = (
        F.relu(0.5 - positives[:, None] + negatives[None, :]).mean()
        if positives.numel() and negatives.numel()
        else logits.sum() * 0.0
    )
    loss = classification + 0.5 * ranking
    stats = binary_metrics(
        target.cpu(), logits.sigmoid().detach().cpu()
    )
    return loss, {
        "shared_loss": float(loss.detach()),
        "shared_balanced_accuracy": stats["balanced_accuracy"],
    }


def planner_loss(model, row, items, tensors, step):
    device = tensors["action"].device
    gold_mask = torch.tensor(
        [gold for _, _, gold in items],
        dtype=torch.bool,
        device=device,
    )
    dependency = dependency_matrix(items, device)
    completed, readiness_target, readiness_mask = execution_state(
        row, items, dependency, device, step
    )
    need_mask = torch.tensor(
        [
            bool(gold and branch.get("depends_on"))
            for _, branch, gold in items
        ],
        dtype=torch.bool,
        device=device,
    )
    output = model.planner_forward(
        tensors["action"],
        tensors["query"],
        tensors["need"],
        tensors["produce"],
        need_mask,
        completed,
    )
    relevance_target = gold_mask.float()
    admission_target = torch.tensor(
        [
            float(
                gold
                and branch.get("admission_target", 1.0)
            )
            for _, branch, gold in items
        ],
        device=device,
    )
    relevance_loss = balanced_bce(
        output["relevance"], relevance_target
    )
    admission_loss = balanced_bce(
        output["admission"], admission_target
    )
    branch_count_target = torch.tensor(
        [
            min(
                int(row.get("branch_count_target", gold_mask.sum())),
                model.max_branches,
            )
        ],
        dtype=torch.long,
        device=device,
    )
    branch_count_loss = F.cross_entropy(
        output["branch_count_logits"].unsqueeze(0),
        branch_count_target,
    )
    predicted_admission_count = output["admission"].sigmoid().sum()
    target_admission_count = branch_count_target.float().squeeze(0)
    admission_budget_loss = F.smooth_l1_loss(
        predicted_admission_count,
        target_admission_count,
    )
    over_admission_loss = (
        F.relu(predicted_admission_count - target_admission_count)
        / max(len(items), 1)
    ).square()
    positives = output["relevance"][gold_mask]
    negatives = output["relevance"][~gold_mask]
    coverage_loss = (
        F.relu(0.75 - positives[:, None] + negatives[None, :]).mean()
        if positives.numel() and negatives.numel()
        else relevance_loss * 0.0
    )

    diagonal = torch.eye(
        len(items), dtype=torch.bool, device=device
    )
    dependency_mask = (
        gold_mask[:, None] & gold_mask[None, :] & ~diagonal
    )
    dependency_loss = masked_balanced_bce(
        output["dependency_logits"],
        dependency,
        dependency_mask,
    )
    readiness_loss = masked_balanced_bce(
        output["readiness"],
        readiness_target,
        readiness_mask & ~completed,
    )
    signatures = [branch_signature(branch) for _, branch, _ in items]
    seen = Counter()
    novelty_target = []
    for signature, (_, _, gold) in zip(signatures, items):
        novelty_target.append(float(gold and seen[signature] == 0))
        seen[signature] += 1
    novelty_target = torch.tensor(novelty_target, device=device)
    novelty_loss = balanced_bce(
        output["novelty"], novelty_target
    )

    distinct = torch.zeros_like(dependency_mask)
    for left in range(len(items)):
        for right in range(len(items)):
            distinct[left, right] = (
                gold_mask[left]
                and gold_mask[right]
                and signatures[left] != signatures[right]
            )
    pair_mask = torch.triu(distinct, diagonal=1)
    steering_similarity = (
        output["steering"] @ output["steering"].T
    )
    repulsive_values = steering_similarity[pair_mask]
    repulsive_loss = (
        F.relu(repulsive_values - 0.1).square().mean()
        if repulsive_values.numel()
        else relevance_loss * 0.0
    )
    probability = output["dependency_probability"]
    identity = torch.eye(len(items), device=device)
    cycle_loss = (
        torch.diagonal(
            torch.linalg.matrix_power(
                identity + probability.square() / max(len(items), 1),
                len(items),
            )
        ).sum()
        - len(items)
    ) / max(len(items), 1)
    loss = (
        1.2 * relevance_loss
        + 0.8 * admission_loss
        + 0.6 * branch_count_loss
        + 0.7 * admission_budget_loss
        + 0.8 * over_admission_loss
        + 0.8 * coverage_loss
        + 0.8 * dependency_loss
        + 0.6 * readiness_loss
        + 0.35 * novelty_loss
        + 0.2 * repulsive_loss
        + 0.03 * cycle_loss
    )
    relevance_stats = binary_metrics(
        relevance_target.cpu(),
        output["relevance"].sigmoid().detach().cpu(),
    )
    top = output["relevance"].topk(int(gold_mask.sum())).indices
    selected = torch.zeros_like(gold_mask)
    selected[top] = True
    recall = float((selected & gold_mask).sum() / gold_mask.sum().clamp_min(1))
    exact = float((selected == gold_mask).all())
    return loss, {
        "planner_loss": float(loss.detach()),
        "relevance_loss": float(relevance_loss.detach()),
        "admission_loss": float(admission_loss.detach()),
        "branch_count_loss": float(branch_count_loss.detach()),
        "admission_budget_loss": float(admission_budget_loss.detach()),
        "over_admission_loss": float(over_admission_loss.detach()),
        "predicted_admission_count": float(
            predicted_admission_count.detach()
        ),
        "target_admission_count": float(target_admission_count),
        "coverage_loss": float(coverage_loss.detach()),
        "dependency_loss": float(dependency_loss.detach()),
        "readiness_loss": float(readiness_loss.detach()),
        "novelty_loss": float(novelty_loss.detach()),
        "repulsive_loss": float(repulsive_loss.detach()),
        "planner_balanced_accuracy": relevance_stats["balanced_accuracy"],
        "planner_recall": recall,
        "planner_exact": exact,
    }


def critic_tensors(row, branches, cache, device):
    keys = [branch_key(row["task_id"], branch) for branch in branches]
    return {
        "query": cache.query(row["task_id"], device),
        "action": cache.branches(keys, "action", device),
        "path": cache.branches(keys, "path", device),
    }


def main_candidate_targets(row, branches, coherence_target):
    targets = [
        float(branch.get("main_candidate_target", 0.0))
        for branch in branches
    ]
    if all("main_candidate_target" in branch for branch in branches):
        return targets
    depended_on = {
        str(edge.get("source_branch_id") or "")
        for branch in branches
        for edge in branch.get("depends_on", [])
    }
    for index, branch in enumerate(branches):
        branch_id = str(branch.get("branch_id"))
        targets[index] = float(
            branch_id not in depended_on
            and coherence_target[index] >= 0.5
        )
    return targets


def critic_loss(model, row, branches, tensors, objective="all"):
    device = tensors["action"].device
    items = [(row["task_id"], branch, True) for branch in branches]
    dependency = dependency_matrix(items, device)
    ancestor = transitive_closure(dependency)
    path_mask = torch.ones(
        len(branches), 1, dtype=torch.bool, device=device
    )
    output = model.critic_forward(
        tensors["action"],
        tensors["path"],
        path_mask,
        query_hidden=tensors["query"],
        ancestor_matrix=ancestor,
    )
    indices = torch.arange(len(branches), device=device)
    assignment_loss = 0.5 * (
        F.cross_entropy(output["assignment_logits"], indices)
        + F.cross_entropy(output["assignment_logits"].T, indices)
    )
    coherence_target = torch.tensor(
        [
            float(branch["effective_coherence_target"])
            for branch in branches
        ],
        device=device,
    )
    coherence_loss = (
        0.4
        * F.binary_cross_entropy_with_logits(
            output["coherence"], coherence_target
        )
        + 0.6
        * balanced_bce(output["coherence"], coherence_target)
        + 0.2
        * coherence_ranking_loss(
            output["coherence"], coherence_target, 0.5
        )
    )
    source_target = torch.tensor(
        [
            float(branch["failure_source_target"])
            for branch in branches
        ],
        device=device,
    )
    failure_source_loss = balanced_bce(
        output["failure_source"], source_target
    )
    failure_target = torch.tensor(
        [int(branch["failure_type_id"]) for branch in branches],
        dtype=torch.long,
        device=device,
    )
    failure_type_loss = balanced_cross_entropy(
        output["failure_type"], failure_target
    )
    recovery_target = torch.tensor(
        [
            RECOVERY_TO_ID[
                FAILURE_TO_RECOVERY[branch["failure_type"]]
            ]
            for branch in branches
        ],
        dtype=torch.long,
        device=device,
    )
    recovery_loss = balanced_cross_entropy(
        output["recovery"], recovery_target
    )
    utility_target = torch.tensor(
        [
            float(branch["execution_coherence_target"])
            for branch in branches
        ],
        device=device,
    )
    utility_loss = balanced_bce(
        output["utility"], utility_target
    )
    explicit_main = main_candidate_targets(
        row, branches, coherence_target
    )
    main_target = torch.tensor(explicit_main, device=device)
    main_score_loss = balanced_bce(
        output["main_score"], main_target
    )
    continuation_target = torch.tensor(
        [
            float(branch.get("continuation_target", 0.0))
            for branch in branches
        ],
        device=device,
    )
    continuation_loss = balanced_bce(
        output["continuation"], continuation_target
    )
    task_target = torch.tensor(
        float(row["task_solvability_target"]), device=device
    )
    task_loss = F.binary_cross_entropy_with_logits(
        output["task_solvability"].reshape(()), task_target
    )
    takeover_target = torch.tensor(
        float(
            row.get(
                "takeover_target",
                bool(row["task_solvability_target"])
                and bool(main_target.any()),
            )
        ),
        device=device,
    )
    takeover_loss = F.binary_cross_entropy_with_logits(
        output["takeover"].reshape(()), takeover_target
    )
    quality_loss = (
        1.2 * coherence_loss
        + 0.25 * utility_loss
        + 0.55 * main_score_loss
    )
    failure_loss = (
        0.65 * failure_source_loss
        + 0.8 * failure_type_loss
        + 0.8 * recovery_loss
        + 0.75 * continuation_loss
    )
    control_loss = (
        + 1.0 * task_loss
        + 0.8 * takeover_loss
    )
    objective_losses = {
        "all": (
            assignment_loss
            + quality_loss
            + failure_loss
            + control_loss
        ),
        "critic_core": assignment_loss,
        "quality": quality_loss,
        "failure": failure_loss,
        "control": control_loss,
    }
    if objective not in objective_losses:
        raise ValueError(f"unknown critic objective: {objective}")
    loss = objective_losses[objective]
    coherence_stats = binary_metrics(
        coherence_target.cpu(),
        output["coherence"].sigmoid().detach().cpu(),
    )
    task_correct = float(
        (output["task_solvability"].sigmoid() >= 0.5)
        == task_target.bool()
    )
    return loss, {
        "critic_loss": float(loss.detach()),
        "critic_objective": objective,
        "quality_expert_loss": float(quality_loss.detach()),
        "failure_expert_loss": float(failure_loss.detach()),
        "control_expert_loss": float(control_loss.detach()),
        "assignment_loss": float(assignment_loss.detach()),
        "coherence_loss": float(coherence_loss.detach()),
        "failure_source_loss": float(failure_source_loss.detach()),
        "failure_type_loss": float(failure_type_loss.detach()),
        "recovery_loss": float(recovery_loss.detach()),
        "task_loss": float(task_loss.detach()),
        "main_score_loss": float(main_score_loss.detach()),
        "continuation_loss": float(continuation_loss.detach()),
        "takeover_loss": float(takeover_loss.detach()),
        "coherence_balanced_accuracy": coherence_stats["balanced_accuracy"],
        "failure_type_accuracy": float(
            output["failure_type"].argmax(-1).eq(
                failure_target
            ).float().mean()
        ),
        "recovery_accuracy": float(
            output["recovery"].argmax(-1).eq(
                recovery_target
            ).float().mean()
        ),
        "task_accuracy": task_correct,
    }, output


def router_loss(model, row, branches, tensors):
    device = tensors["action"].device
    items = [(row["task_id"], branch, True) for branch in branches]
    dependency = dependency_matrix(items, device)
    ancestor = transitive_closure(dependency)
    path_mask = torch.ones(
        len(branches), 1, dtype=torch.bool, device=device
    )
    with torch.no_grad():
        critic = model.critic_forward(
            tensors["action"],
            tensors["path"],
            path_mask,
            query_hidden=tensors["query"],
            ancestor_matrix=ancestor,
        )
    initial = model.router_forward(
        tensors["query"], tensors["action"], None
    )
    post = model.router_forward(
        tensors["query"], tensors["action"], critic["recovery"].detach()
    )
    initial_target = torch.tensor([0], device=device)
    post_target_value = 2 if row["task_solvability_target"] else 1
    post_target = torch.tensor([post_target_value], device=device)
    route_loss = (
        F.cross_entropy(
            initial["router_logits"].unsqueeze(0), initial_target
        )
        + F.cross_entropy(
            post["router_logits"].unsqueeze(0), post_target
        )
    )
    failure_target = torch.tensor(
        [
            float(
                branch["failure_source_target"]
                or not branch["effective_coherence_target"]
            )
            for branch in branches
        ],
        device=device,
    )
    bridge_loss = balanced_bce(
        post["recovery_relevance"], failure_target
    )
    gate_target = torch.tensor(
        float(not row["task_solvability_target"]), device=device
    )
    gate_loss = F.binary_cross_entropy(
        post["bridge_gate"], gate_target
    )
    loss = route_loss + 0.7 * bridge_loss + 0.5 * gate_loss
    return loss, {
        "router_loss": float(loss.detach()),
        "route_classification_loss": float(route_loss.detach()),
        "bridge_loss": float(bridge_loss.detach()),
        "bridge_gate_loss": float(gate_loss.detach()),
        "router_accuracy": 0.5
        * (
            float(initial["router_logits"].argmax() == 0)
            + float(post["router_logits"].argmax() == post_target_value)
        ),
    }


def choose_distractor_pool(schedule, rows_by_id, max_nodes):
    unique = sorted({item["task_id"] for item in schedule})
    return [
        (task_id, branch)
        for task_id in unique
        for branch in select_nodes(rows_by_id[task_id], max_nodes)
    ]


def apply_update(model, loss, optimizers, active, max_grad_norm):
    optimizer_names = {
        "planner": ("shared", "planner"),
        "critic_core": ("critic_core",),
        "quality": ("quality",),
        "failure": ("failure",),
        "control": ("control", "router"),
        "joint": (
            "shared",
            "planner",
            "critic_core",
            "quality",
            "failure",
            "control",
            "router",
        ),
    }[active]
    if active == "planner":
        model.set_active_group("joint_planner")
    elif active == "control":
        model.set_active_group("control_router")
    elif active == "joint":
        model.set_active_group("all")
    else:
        model.set_active_group(active)
    for name in optimizer_names:
        optimizers[name].zero_grad(set_to_none=True)
    loss.backward()
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_grad_norm
    )
    for name in optimizer_names:
        optimizers[name].step()
    return float(grad_norm)


@torch.inference_mode()
def evaluate(
    model,
    evaluation,
    rows_by_id,
    cache,
    distractor_pool,
    args,
    device,
    step,
):
    model.eval()
    planner_labels = []
    planner_probabilities = []
    admission_probabilities = []
    admission_labels = []
    branch_count_correct = 0
    plan_recalls = []
    plan_exacts = []
    dependency_labels = []
    dependency_probabilities = []
    readiness_labels = []
    readiness_probabilities = []
    coherence_labels = []
    coherence_probabilities = []
    task_labels = []
    task_probabilities = []
    main_labels = []
    main_probabilities = []
    takeover_labels = []
    takeover_probabilities = []
    continuation_labels = []
    continuation_probabilities = []
    recovery_correct = 0
    recovery_total = 0
    failure_correct = 0
    failure_total = 0
    router_correct = 0
    router_total = 0
    for eval_index, row in enumerate(evaluation):
        items = candidate_branches(
            row,
            rows_by_id,
            distractor_pool,
            args.max_nodes,
            args.distractors,
            step + eval_index,
        )
        tensors = planner_tensors(row, items, cache, device)
        gold_mask = torch.tensor(
            [gold for _, _, gold in items],
            dtype=torch.bool,
            device=device,
        )
        dependency = dependency_matrix(items, device)
        completed, readiness_target, readiness_mask = execution_state(
            row, items, dependency, device, step + eval_index
        )
        need_mask = torch.tensor(
            [
                bool(gold and branch.get("depends_on"))
                for _, branch, gold in items
            ],
            dtype=torch.bool,
            device=device,
        )
        planner = model.planner_forward(
            tensors["action"],
            tensors["query"],
            tensors["need"],
            tensors["produce"],
            need_mask,
            completed,
        )
        planner_labels.extend(gold_mask.cpu().tolist())
        planner_probabilities.extend(
            planner["relevance"].sigmoid().cpu().tolist()
        )
        admission_probabilities.extend(
            planner["admission"].sigmoid().cpu().tolist()
        )
        admission_labels.extend(
            [
                int(
                    gold
                    and branch.get("admission_target", 1.0)
                )
                for _, branch, gold in items
            ]
        )
        branch_count_correct += int(
            planner["branch_count_logits"].argmax()
            == min(
                int(row.get("branch_count_target", gold_mask.sum())),
                model.max_branches,
            )
        )
        top = planner["relevance"].topk(int(gold_mask.sum())).indices
        selected = torch.zeros_like(gold_mask)
        selected[top] = True
        plan_recalls.append(
            float((selected & gold_mask).sum() / gold_mask.sum())
        )
        plan_exacts.append(float((selected == gold_mask).all()))
        pair_mask = (
            gold_mask[:, None]
            & gold_mask[None, :]
            & ~torch.eye(
                len(items), dtype=torch.bool, device=device
            )
        )
        dependency_labels.extend(dependency[pair_mask].cpu().tolist())
        dependency_probabilities.extend(
            planner["dependency_probability"][pair_mask].cpu().tolist()
        )
        ready_mask = readiness_mask & ~completed
        readiness_labels.extend(readiness_target[ready_mask].cpu().tolist())
        readiness_probabilities.extend(
            planner["readiness"].sigmoid()[ready_mask].cpu().tolist()
        )

        branches = select_nodes(row, args.max_nodes)
        critic_inputs = critic_tensors(row, branches, cache, device)
        critic_items = [
            (row["task_id"], branch, True) for branch in branches
        ]
        critic_dependency = dependency_matrix(critic_items, device)
        critic = model.critic_forward(
            critic_inputs["action"],
            critic_inputs["path"],
            torch.ones(
                len(branches), 1, dtype=torch.bool, device=device
            ),
            query_hidden=critic_inputs["query"],
            ancestor_matrix=transitive_closure(critic_dependency),
        )
        coherence_target = [
            int(branch["effective_coherence_target"])
            for branch in branches
        ]
        coherence_labels.extend(coherence_target)
        coherence_probabilities.extend(
            critic["coherence"].sigmoid().cpu().tolist()
        )
        task_labels.append(int(row["task_solvability_target"]))
        task_probabilities.append(
            float(critic["task_solvability"].sigmoid())
        )
        main_target = main_candidate_targets(
            row, branches, coherence_target
        )
        main_labels.extend(main_target)
        main_probabilities.extend(
            critic["main_score"].sigmoid().cpu().tolist()
        )
        takeover_labels.append(
            int(
                row.get(
                    "takeover_target",
                    bool(row["task_solvability_target"])
                    and any(main_target),
                )
            )
        )
        takeover_probabilities.append(
            float(critic["takeover"].sigmoid())
        )
        continuation_labels.extend(
            [
                int(branch.get("continuation_target", 0.0))
                for branch in branches
            ]
        )
        continuation_probabilities.extend(
            critic["continuation"].sigmoid().cpu().tolist()
        )
        failure_target = torch.tensor(
            [branch["failure_type_id"] for branch in branches],
            device=device,
        )
        recovery_target = torch.tensor(
            [
                RECOVERY_TO_ID[
                    FAILURE_TO_RECOVERY[branch["failure_type"]]
                ]
                for branch in branches
            ],
            device=device,
        )
        failure_correct += int(
            critic["failure_type"].argmax(-1).eq(
                failure_target
            ).sum()
        )
        recovery_correct += int(
            critic["recovery"].argmax(-1).eq(
                recovery_target
            ).sum()
        )
        failure_total += len(branches)
        recovery_total += len(branches)
        initial = model.router_forward(
            critic_inputs["query"], critic_inputs["action"], None
        )
        post = model.router_forward(
            critic_inputs["query"],
            critic_inputs["action"],
            critic["recovery"],
        )
        router_correct += int(initial["router_logits"].argmax() == 0)
        router_correct += int(
            post["router_logits"].argmax()
            == (2 if row["task_solvability_target"] else 1)
        )
        router_total += 2

    planner_metrics = binary_metrics(
        planner_labels, planner_probabilities
    )
    admission_metrics = binary_metrics(
        admission_labels, admission_probabilities
    )
    dependency_metrics = binary_metrics(
        dependency_labels, dependency_probabilities
    )
    readiness_metrics = binary_metrics(
        readiness_labels, readiness_probabilities
    )
    coherence_metrics = binary_metrics(
        coherence_labels, coherence_probabilities
    )
    task_metrics = binary_metrics(task_labels, task_probabilities)
    main_metrics = binary_metrics(main_labels, main_probabilities)
    takeover_metrics = binary_metrics(
        takeover_labels, takeover_probabilities
    )
    continuation_metrics = binary_metrics(
        continuation_labels, continuation_probabilities
    )
    result = {
        "step": step,
        "planner": planner_metrics,
        "admission": admission_metrics,
        "branch_count_accuracy": branch_count_correct / len(evaluation),
        "plan_recall": sum(plan_recalls) / len(plan_recalls),
        "plan_exact": sum(plan_exacts) / len(plan_exacts),
        "dependency": dependency_metrics,
        "readiness": readiness_metrics,
        "coherence": coherence_metrics,
        "task_solvability": task_metrics,
        "main_candidate": main_metrics,
        "takeover": takeover_metrics,
        "continuation": continuation_metrics,
        "failure_type_accuracy": failure_correct / max(failure_total, 1),
        "recovery_accuracy": recovery_correct / max(recovery_total, 1),
        "router_accuracy": router_correct / max(router_total, 1),
    }
    result["composite_score"] = (
        0.16 * result["plan_recall"]
        + 0.07 * result["plan_exact"]
        + 0.07 * result["admission"]["balanced_accuracy"]
        + 0.05 * result["branch_count_accuracy"]
        + 0.08 * result["dependency"]["balanced_accuracy"]
        + 0.05 * result["readiness"]["balanced_accuracy"]
        + 0.12 * result["coherence"]["balanced_accuracy"]
        + 0.10 * result["task_solvability"]["balanced_accuracy"]
        + 0.08 * result["main_candidate"]["balanced_accuracy"]
        + 0.07 * result["takeover"]["balanced_accuracy"]
        + 0.05 * result["continuation"]["balanced_accuracy"]
        + 0.07 * result["recovery_accuracy"]
        + 0.03 * result["router_accuracy"]
    )
    return result


def optimizer_payload(optimizers):
    return {
        name: optimizer.state_dict()
        for name, optimizer in optimizers.items()
    }


def checkpoint_payload(
    model,
    optimizers,
    step,
    args,
    history,
    schedule,
    evaluation,
    validation,
):
    return {
        "version": 11,
        "training_mode": (
            "full_data_moe_specialists_with_joint_calibration"
        ),
        "random_init": True,
        "step": step,
        "model": model.state_dict(),
        "optimizers": optimizer_payload(optimizers),
        "args": vars(args),
        "history": history,
        "schedule": schedule,
        "eval_task_ids": [row["task_id"] for row in evaluation],
        "validation": validation,
        "recovery_types": RECOVERY_TYPES,
        "parameter_counts": parameter_counts(model),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-path",
        default=os.environ.get(
            "PI_LOCAL_QWEN_MODEL", "models/qwen3-8b"
        ),
    )
    parser.add_argument(
        "--data", default="data/traject_swarm_training_failure_aware_v11.jsonl"
    )
    parser.add_argument(
        "--cache", default="cache/rcg_swarm_v11_encoder_cache.pt"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Optional prefix of the full curriculum; 0 uses every row per expert.",
    )
    parser.add_argument("--eval-per-stratum", type=int, default=75)
    parser.add_argument("--max-nodes", type=int, default=13)
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--query-tokens", type=int, default=128)
    parser.add_argument("--action-tokens", type=int, default=128)
    parser.add_argument("--path-tokens", type=int, default=128)
    parser.add_argument("--parameter-tokens", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--shared-layers", type=int, default=2)
    parser.add_argument("--planner-layers", type=int, default=2)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--set-heads", type=int, default=8)
    parser.add_argument("--max-branches", type=int, default=16)
    parser.add_argument("--shared-lr", type=float, default=5e-5)
    parser.add_argument("--planner-lr", type=float, default=8e-5)
    parser.add_argument("--critic-lr", type=float, default=8e-5)
    parser.add_argument("--router-lr", type=float, default=1e-4)
    parser.add_argument("--joint-lr-scale", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rcg_swarm_v11_local_latest.pt",
    )
    parser.add_argument(
        "--checkpoint-prefix", default="rcg_swarm_v11_local"
    )
    parser.add_argument(
        "--best-checkpoint",
        default="checkpoints/rcg_swarm_v11_local_best.pt",
    )
    parser.add_argument(
        "--log-file", default="logs/rcg_swarm_v11_full.jsonl"
    )
    parser.add_argument(
        "--validation-file",
        default="eval_results/rcg_swarm_v11_validation_history.json",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.device = resolve_device(args.device)
    device = torch.device(args.device)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    rows = read_rows(args.data)
    data_stats = validate_training_rows(rows)
    print(f"Validated training data: {data_stats}", flush=True)
    rows_by_id = {row["task_id"]: row for row in rows}
    schedule, evaluation = build_schedule(
        rows, args.steps, args.eval_per_stratum, args.seed
    )
    args.steps = len(schedule)
    schedule_counts = Counter(
        (
            rows_by_id[item["task_id"]]["trajectory_type"],
            int(rows_by_id[item["task_id"]]["task_solvability_target"]),
        )
        for item in schedule
    )
    print(
        f"V11 schedule steps={len(schedule)} eval={len(evaluation)} "
        f"strata={dict(schedule_counts)}",
        flush=True,
    )
    run_schedule = (
        schedule[: args.stop_after]
        if args.stop_after
        else schedule
    )
    cache_payload = build_or_load_cache(
        args, rows_by_id, run_schedule, evaluation, device
    )
    cache = CacheView(cache_payload)
    distractor_pool = choose_distractor_pool(
        run_schedule, rows_by_id, args.max_nodes
    )

    model = DualTowerRCG(
        hidden_size=cache_payload["query"].shape[-1],
        latent_dim=args.latent_dim,
        shared_layers=args.shared_layers,
        planner_layers=args.planner_layers,
        critic_layers=args.critic_layers,
        set_heads=args.set_heads,
        max_path_tokens=args.path_tokens,
        failure_classes=len(FAILURE_TYPES),
        recovery_classes=len(RECOVERY_TYPES),
        max_branches=args.max_branches,
    ).to(device)
    counts = parameter_counts(model)
    print(f"Random-init RCG Swarm V11 parameters={counts}", flush=True)
    optimizers = {
        "shared": torch.optim.AdamW(
            model.shared_parameters(),
            lr=args.shared_lr,
            weight_decay=args.weight_decay,
        ),
        "planner": torch.optim.AdamW(
            model.planner_parameters(),
            lr=args.planner_lr,
            weight_decay=args.weight_decay,
        ),
        "critic_core": torch.optim.AdamW(
            model.critic_core_parameters(),
            lr=args.critic_lr,
            weight_decay=args.weight_decay,
        ),
        "quality": torch.optim.AdamW(
            model.quality_parameters(),
            lr=args.critic_lr,
            weight_decay=args.weight_decay,
        ),
        "failure": torch.optim.AdamW(
            model.failure_parameters(),
            lr=args.critic_lr,
            weight_decay=args.weight_decay,
        ),
        "control": torch.optim.AdamW(
            model.control_parameters(),
            lr=args.critic_lr,
            weight_decay=args.weight_decay,
        ),
        "router": torch.optim.AdamW(
            model.router_parameters(),
            lr=args.router_lr,
            weight_decay=args.weight_decay,
        ),
    }
    history = []
    validation_history = []
    best_score = -math.inf
    joint_lr_scaled = False
    started = time.time()
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.log_file).exists():
        Path(args.log_file).unlink()

    for step, schedule_item in enumerate(run_schedule, 1):
        row = rows_by_id[schedule_item["task_id"]]
        phase = schedule_item["phase"]
        if phase == "joint" and not joint_lr_scaled:
            for optimizer in optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] *= args.joint_lr_scale
            joint_lr_scaled = True
            print(
                f"Joint calibration learning rates scaled by "
                f"{args.joint_lr_scale:.3f}.",
                flush=True,
            )
        if phase == "planner":
            items = candidate_branches(
                row,
                rows_by_id,
                distractor_pool,
                args.max_nodes,
                args.distractors,
                step,
            )
            tensors = planner_tensors(row, items, cache, device)
            model.set_active_group("joint_planner")
            loss, metrics = planner_loss(
                model, row, items, tensors, step
            )
        elif phase == "joint":
            items = candidate_branches(
                row,
                rows_by_id,
                distractor_pool,
                args.max_nodes,
                args.distractors,
                step,
            )
            planner_inputs = planner_tensors(row, items, cache, device)
            branches = select_nodes(row, args.max_nodes)
            critic_inputs = critic_tensors(
                row, branches, cache, device
            )
            model.set_active_group("all")
            planner_part, planner_metrics = planner_loss(
                model, row, items, planner_inputs, step
            )
            critic_part, critic_metrics, _ = critic_loss(
                model, row, branches, critic_inputs, objective="all"
            )
            route_part, route_metrics = router_loss(
                model, row, branches, critic_inputs
            )
            loss = planner_part + critic_part + route_part
            metrics = {
                **planner_metrics,
                **critic_metrics,
                **route_metrics,
                "joint_planner_loss": float(planner_part.detach()),
                "joint_critic_loss": float(critic_part.detach()),
                "joint_router_loss": float(route_part.detach()),
            }
            tensors = {
                "planner": planner_inputs,
                "critic": critic_inputs,
            }
        else:
            branches = select_nodes(row, args.max_nodes)
            tensors = critic_tensors(row, branches, cache, device)
            if phase in {"critic_core", "quality", "failure"}:
                model.set_active_group(phase)
                loss, metrics, _ = critic_loss(
                    model, row, branches, tensors, objective=phase
                )
            else:
                model.set_active_group("control_router")
                control_loss, control_metrics, _ = critic_loss(
                    model, row, branches, tensors, objective="control"
                )
                route_loss, route_metrics = router_loss(
                    model, row, branches, tensors
                )
                loss = control_loss + route_loss
                metrics = {**control_metrics, **route_metrics}
        grad_norm = apply_update(
            model,
            loss,
            optimizers,
            phase,
            args.max_grad_norm,
        )
        metrics.update(
            {
                "step": step,
                "phase": phase,
                "task_id": row["task_id"],
                "trajectory_type": row["trajectory_type"],
                "task_solvability_target": row["task_solvability_target"],
                "loss": float(loss.detach()),
                "grad_norm": grad_norm,
                "elapsed_seconds": time.time() - started,
                "peak_memory_gb": peak_memory_gb(device),
            }
        )
        history.append(metrics)
        append_jsonl(args.log_file, metrics)
        del loss, tensors

        if step % args.log_every == 0:
            recent = history[-args.log_every :]
            print(
                f"Step {step}/{args.steps} phase={phase} "
                f"loss={sum(row['loss'] for row in recent) / len(recent):.4f} "
                f"grad={sum(row['grad_norm'] for row in recent) / len(recent):.3f} "
                f"peak={metrics['peak_memory_gb']:.2f}GB "
                f"elapsed={(time.time() - started) / 3600:.2f}h",
                flush=True,
            )

        next_phase = (
            run_schedule[step]["phase"]
            if step < len(run_schedule)
            else None
        )
        phase_complete = next_phase != phase
        should_evaluate = (
            not args.skip_eval
            and (
                phase_complete
                or (
                    args.eval_every
                    and step % args.eval_every == 0
                )
            )
        )
        validation = validation_history[-1] if validation_history else None
        if should_evaluate:
            validation = evaluate(
                model,
                evaluation,
                rows_by_id,
                cache,
                distractor_pool,
                args,
                device,
                step,
            )
            validation_history.append(validation)
            atomic_json(args.validation_file, validation_history)
            print(
                f"Validation step={step} "
                f"score={validation['composite_score']:.4f} "
                f"plan={validation['plan_recall']:.4f}/"
                f"{validation['plan_exact']:.4f} "
                f"count={validation['branch_count_accuracy']:.4f} "
                f"coh={validation['coherence']['balanced_accuracy']:.4f} "
                f"takeover={validation['takeover']['balanced_accuracy']:.4f} "
                f"continue={validation['continuation']['balanced_accuracy']:.4f} "
                f"task={validation['task_solvability']['balanced_accuracy']:.4f} "
                f"recovery={validation['recovery_accuracy']:.4f}",
                flush=True,
            )
            if validation["composite_score"] > best_score:
                best_score = validation["composite_score"]
                best_payload = checkpoint_payload(
                    model,
                    optimizers,
                    step,
                    args,
                    history,
                    schedule,
                    evaluation,
                    validation,
                )
                best_payload["best_score"] = best_score
                atomic_torch_save(args.best_checkpoint, best_payload)
                print(
                    f"Updated best checkpoint at step {step}: "
                    f"{best_score:.4f}",
                    flush=True,
                )
            latest = checkpoint_payload(
                model,
                optimizers,
                step,
                args,
                history,
                schedule,
                evaluation,
                validation,
            )
            latest["best_score"] = best_score
            atomic_torch_save(args.checkpoint, latest)

        should_archive = phase_complete or (
            args.save_every and step % args.save_every == 0
        )
        if should_archive:
            payload = checkpoint_payload(
                model,
                optimizers,
                step,
                args,
                history,
                schedule,
                evaluation,
                validation,
            )
            payload["best_score"] = best_score
            archive = (
                Path(args.checkpoint).parent
                / f"{args.checkpoint_prefix}_step_{step}.pt"
            )
            atomic_torch_save(archive, payload)
            atomic_torch_save(args.checkpoint, payload)
            print(f"Archived checkpoint {archive}", flush=True)
        if step % 50 == 0:
            empty_device_cache(device)

    print(
        f"Finished V11 training steps={len(run_schedule)} "
        f"best_score={best_score:.4f} "
        f"elapsed={(time.time() - started) / 3600:.2f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
