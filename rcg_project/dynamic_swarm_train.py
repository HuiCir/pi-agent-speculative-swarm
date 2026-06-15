"""Train a unified RCG action-DAG policy without serial/parallel modes."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_failure_aware_training_data import FAILURE_TYPE_TO_ID, FAILURE_TYPES
from rcg.dynamic_controller import DynamicRCG, dynamic_parameter_count


WORD_PATTERN = re.compile(r"[a-z0-9_\u0080-\uffff]+", re.IGNORECASE)
ERROR_MARKERS = (
    "error:",
    "failed after",
    "timeout after",
    "api is unreachable",
    "api (not working)",
    "internal server error",
    "bad gateway",
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def balanced_split(rows, train_count, eval_count, seed):
    def edge_bearing(row):
        return any(branch.get("depends_on") for branch in row["branches"])

    def domain_balanced_order(items, class_seed):
        groups = defaultdict(list)
        for row in items:
            source_family = str(row.get("source", "")).split("/", 1)[-1]
            solvability = str(
                (row.get("task_outcome") or {}).get("status", "unknown")
            )
            groups[(row["domain"], source_family, solvability)].append(row)
        for key, group in groups.items():
            random.Random(class_seed ^ stable_int("/".join(key))).shuffle(group)
        positions = {key: 0 for key in groups}
        ordered = []
        while True:
            progressed = False
            for key in sorted(groups):
                index = positions[key]
                if index >= len(groups[key]):
                    continue
                ordered.append(groups[key][index])
                positions[key] = index + 1
                progressed = True
            if not progressed:
                return ordered

    pools = {
        state: domain_balanced_order(
            [row for row in rows if edge_bearing(row) == state],
            seed ^ stable_int(f"edge-bearing={state}"),
        )
        for state in (False, True)
    }
    train_each = {False: train_count // 2, True: train_count - train_count // 2}
    eval_each = {False: eval_count // 2, True: eval_count - eval_count // 2}
    training = []
    evaluation = []
    for state in (False, True):
        train_stop = train_each[state]
        eval_stop = train_stop + eval_each[state]
        training.extend(pools[state][:train_stop])
        evaluation.extend(pools[state][train_stop:eval_stop])
    random.Random(seed).shuffle(training)
    random.Random(seed + 1).shuffle(evaluation)
    return training, evaluation


def ratio_split(
    rows,
    train_count,
    eval_count,
    positive_ratio,
    seed,
    negative_parallel_fraction=0.2,
):
    positive = [
        row for row in rows if float(row.get("task_solvability_target", 1.0)) == 1.0
    ]
    negative = [
        row for row in rows if float(row.get("task_solvability_target", 1.0)) == 0.0
    ]
    random.Random(seed).shuffle(positive)
    random.Random(seed + 1).shuffle(negative)

    def counts(total):
        positives = round(total * positive_ratio)
        return positives, total - positives

    train_positive, train_negative = counts(train_count)
    eval_positive, eval_negative = counts(eval_count)
    required_positive = train_positive + eval_positive
    required_negative = train_negative + eval_negative
    if len(positive) < required_positive or len(negative) < required_negative:
        raise RuntimeError(
            "Insufficient ratio-split rows: "
            f"positive={len(positive)}/{required_positive}, "
            f"negative={len(negative)}/{required_negative}"
        )
    train_positive_rows = positive[:train_positive]
    eval_positive_rows = positive[train_positive:required_positive]

    def branch_counts(row):
        positives = sum(
            float(branch.get("effective_coherence_target", 1.0)) == 1.0
            for branch in row["branches"]
        )
        return positives, len(row["branches"]) - positives

    branch_ratio = positive_ratio / max(1.0 - positive_ratio, 1e-6)

    def select_negatives(candidates, count, positive_rows):
        if not count:
            return []
        positive_base = sum(
            branch_counts(row)[0] for row in positive_rows
        )
        parallel_count = min(
            round(count * negative_parallel_fraction),
            sum(row.get("trajectory_type") == "parallel" for row in candidates),
        )
        parallel = [
            row for row in candidates if row.get("trajectory_type") == "parallel"
        ]
        parallel.sort(
            key=lambda row: (
                branch_ratio * branch_counts(row)[1]
                - branch_counts(row)[0]
            ),
            reverse=True,
        )
        selected = parallel[:parallel_count]
        selected_ids = {id(row) for row in selected}
        remaining_count = count - len(selected)
        score_so_far = sum(
            branch_ratio * branch_counts(row)[1]
            - branch_counts(row)[0]
            for row in selected
        )
        target_average = (
            (positive_base - score_so_far) / remaining_count
            if remaining_count
            else 0.0
        )
        remaining = [
            row
            for row in candidates
            if id(row) not in selected_ids
            and row.get("trajectory_type") != "parallel"
        ]
        if len(remaining) < remaining_count:
            remaining.extend(
                row
                for row in candidates
                if id(row) not in selected_ids
                and row not in remaining
            )
        remaining.sort(
            key=lambda row: abs(
                (
                    branch_ratio * branch_counts(row)[1]
                    - branch_counts(row)[0]
                )
                - target_average
            )
        )
        return selected + remaining[:remaining_count]

    train_negative_rows = select_negatives(
        negative, train_negative, train_positive_rows
    )
    used = {id(row) for row in train_negative_rows}
    eval_negative_rows = select_negatives(
        [row for row in negative if id(row) not in used],
        eval_negative,
        eval_positive_rows,
    )
    training = train_positive_rows + train_negative_rows
    evaluation = eval_positive_rows + eval_negative_rows
    random.Random(seed + 2).shuffle(training)
    random.Random(seed + 3).shuffle(evaluation)
    return training, evaluation


def select_nodes(row, max_nodes):
    branches = list(row["branches"])
    if len(branches) <= max_nodes:
        return branches
    by_id = {branch["branch_id"]: branch for branch in branches}
    selected = []
    queue = sorted(
        branches,
        key=lambda branch: (
            len(branch.get("depends_on", [])),
            branch.get("earliest_commit_stage", 0),
            branch["branch_id"],
        ),
    )
    for branch in queue:
        required = [
            edge.get("source_branch_id")
            for edge in branch.get("depends_on", [])
            if edge.get("source_branch_id") in by_id
        ]
        for source_id in required:
            source = by_id[source_id]
            if source not in selected and len(selected) < max_nodes:
                selected.append(source)
        if branch not in selected and len(selected) < max_nodes:
            selected.append(branch)
        if len(selected) >= max_nodes:
            break
    order = {branch["branch_id"]: index for index, branch in enumerate(branches)}
    return sorted(selected, key=lambda branch: order[branch["branch_id"]])


def render_action(row, branch):
    return (
        "<ACTION_NODE>\n"
        f"GLOBAL_QUERY: {row['query']}\n"
        f"TOOL: {branch['tool_name']}\n"
        f"LOCAL_OBJECTIVE: {branch['tool_description']}\n"
        f"REQUIRED_PARAMETERS: "
        f"{json.dumps(branch['required_parameters'], ensure_ascii=False)}\n"
        f"OPTIONAL_PARAMETERS: "
        f"{json.dumps(branch['optional_parameters'], ensure_ascii=False)}\n"
        f"PRODUCES: {json.dumps(branch.get('produces'), ensure_ascii=False)}\n"
        "</ACTION_NODE>"
    )


def render_path(row, branch):
    output = str(branch.get("executed_output", ""))
    if len(output) > 2200:
        output = output[:2200]
    return (
        "<ACTION_EXECUTION>\n"
        f"LOCAL_OBJECTIVE: {branch['tool_description']}\n"
        f"ACTUAL_TOOL: {branch['tool_name']}\n"
        f"ACTUAL_PARAMETERS: "
        f"{json.dumps(branch['required_parameters'], ensure_ascii=False)}\n"
        f"STATUS: {branch.get('execution_status', 'unknown')}\n"
        f"LIVE_STATUS: {(branch.get('live_replay') or {}).get('status', 'unknown')}\n"
        f"FAILURE_TYPE: {branch.get('failure_type', 'unknown')}\n"
        f"BLOCKED_BY: {json.dumps(branch.get('blocked_by', []), ensure_ascii=False)}\n"
        f"RESULT: {output}\n"
        f"ADAPTATION: {json.dumps(branch.get('adaptation'), ensure_ascii=False)}\n"
        "</ACTION_EXECUTION>"
    )


def unresolved_need_text(branch):
    parameters = sorted(
        {
            str(edge.get("parameter", "")).strip()
            for edge in branch.get("depends_on", [])
            if str(edge.get("parameter", "")).strip()
        }
    )
    return " ; ".join(parameters) if parameters else "<NO_UNRESOLVED_INPUT>"


def produced_value_text(branch):
    parameters = sorted(
        {
            str(item.get("parameter", "")).strip()
            for item in branch.get("produces", []) or []
            if str(item.get("parameter", "")).strip()
        }
    )
    return " ; ".join(parameters) if parameters else "<NO_DECLARED_OUTPUT>"


def lexical_coverage(output, answer):
    output_words = set(WORD_PATTERN.findall(str(output).lower()))
    answer_words = set(WORD_PATTERN.findall(str(answer).lower()))
    if not output_words or not answer_words:
        return 0.0
    return len(output_words & answer_words) / max(min(len(output_words), len(answer_words)), 1)


def obvious_bad_result(output):
    text = str(output or "").strip()
    lowered = text.lower()
    if not text or any(marker in lowered for marker in ERROR_MARKERS):
        return True
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False
    if value in (None, [], {}):
        return True
    if isinstance(value, dict):
        if value.get("success") is False or value.get("found") is False:
            return True
        for key in ("results", "result", "items", "articles", "list", "value"):
            if key in value and value[key] in (None, [], {}):
                return True
    return False


def strict_coherence(branch):
    if "effective_coherence_target" in branch:
        return float(branch["effective_coherence_target"])
    replay = branch.get("live_replay") or {}
    if replay.get("status"):
        return float(replay["status"] == "live_valid")
    return float(
        branch.get("execution_status") == "success"
        and not obvious_bad_result(branch.get("executed_output"))
    )


def final_answer_text(value):
    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values())
    return str(value or "")


def dependency_targets(branches, device):
    count = len(branches)
    by_id = {branch["branch_id"]: index for index, branch in enumerate(branches)}
    target = torch.zeros((count, count), device=device)
    explicit = torch.zeros((count, count), device=device)
    for target_index, branch in enumerate(branches):
        for edge in branch.get("depends_on", []):
            source_index = by_id.get(edge.get("source_branch_id"))
            if source_index is None or source_index == target_index:
                continue
            target[target_index, source_index] = 1.0
            explicit[target_index, source_index] = (
                1.0 if edge.get("annotation_source") == "dataset" else 0.35
            )
    return target, explicit


def graph_levels(dependency):
    count = dependency.shape[0]
    levels = [0] * count
    for _ in range(count):
        changed = False
        for target in range(count):
            sources = dependency[target].nonzero(as_tuple=False).flatten().tolist()
            level = 0 if not sources else max(levels[source] + 1 for source in sources)
            if level > levels[target]:
                levels[target] = level
                changed = True
        if not changed:
            break
    return levels


def execution_snapshot(row, branches, dependency, device):
    levels = graph_levels(dependency.detach().cpu())
    max_level = max(levels, default=0)
    stage = stable_int(f"{row['task_id']}:dag-stage") % (max_level + 1)
    completed = torch.tensor(
        [level < stage for level in levels], dtype=torch.bool, device=device
    )
    readiness = torch.zeros(len(branches), device=device)
    for index in range(len(branches)):
        if completed[index]:
            continue
        sources = dependency[index].bool()
        readiness[index] = float(not sources.any() or completed[sources].all())
    return completed, readiness, stage


def normalized_parameters(branch):
    return tuple(
        sorted(
            (
                str(item.get("name", "")).lower(),
                json.dumps(item.get("value"), ensure_ascii=False, sort_keys=True),
            )
            for item in branch.get("required_parameters", [])
        )
    )


def action_signature(branch):
    return branch["tool_name"], normalized_parameters(branch)


def build_result_donor_pool(rows, max_nodes):
    donors = []
    for row in rows:
        for branch in select_nodes(row, max_nodes):
            if strict_coherence(branch):
                item = copy.deepcopy(branch)
                item["_task_id"] = row["task_id"]
                donors.append(item)
    return donors


def mutate_required_parameters(parameters):
    mutated = copy.deepcopy(parameters)
    if not mutated:
        return [{"name": "__unexpected_parameter__", "value": "__wrong__"}]
    mutated[0]["value"] = f"__wrong__{mutated[0].get('value', '')}"
    return mutated


def choose_result_donor(row, branch, donor_pool, same_tool):
    candidates = [
        donor
        for donor in donor_pool
        if donor.get("_task_id") != row["task_id"]
        and str(donor.get("executed_output", "")).strip()
        and (
            donor.get("tool_name") == branch.get("tool_name")
            if same_tool
            else donor.get("tool_name") != branch.get("tool_name")
        )
        and (
            not same_tool
            or normalized_parameters(donor)
            != normalized_parameters(branch)
        )
    ]
    if not candidates:
        candidates = [
            donor
            for donor in donor_pool
            if donor.get("_task_id") != row["task_id"]
            and str(donor.get("executed_output", "")).strip()
        ]
    if not candidates:
        return None
    index = stable_int(
        f"{row['task_id']}:{branch['branch_id']}:"
        f"{'same' if same_tool else 'other'}-donor"
    )
    return candidates[index % len(candidates)]


def corrupt_results(row, branches, donor_pool=None, synthetic_faults=True):
    donor_pool = donor_pool or branches
    corrupted = copy.deepcopy(branches)
    targets = []
    for index, branch in enumerate(corrupted):
        target = strict_coherence(branch)
        branch["_fault_type"] = "natural"
        branch["_training_failure_type_id"] = int(
            branch.get(
                "failure_type_id",
                FAILURE_TYPE_TO_ID["success" if target else "other_failure"],
            )
        )
        mode = (
            stable_int(f"{row['task_id']}:{branch['branch_id']}:fault-v2")
            % 100
        )
        if synthetic_faults and target == 1.0 and mode < 20:
            donor = choose_result_donor(
                row, branch, donor_pool, same_tool=True
            )
            if donor is not None:
                branch["executed_output"] = donor["executed_output"]
                branch["execution_status"] = "success"
                branch["_fault_type"] = "stale_same_tool_result"
                branch["_training_failure_type_id"] = FAILURE_TYPE_TO_ID[
                    "provenance_mismatch"
                ]
                target = 0.0
        elif synthetic_faults and target == 1.0 and mode < 35:
            branch["required_parameters"] = mutate_required_parameters(
                branch.get("required_parameters", [])
            )
            branch["_fault_type"] = "wrong_parameters"
            branch["_training_failure_type_id"] = FAILURE_TYPE_TO_ID[
                "invalid_call"
            ]
            target = 0.0
        elif synthetic_faults and target == 1.0 and mode < 45:
            donor = choose_result_donor(
                row, branch, donor_pool, same_tool=False
            )
            if donor is not None:
                branch["tool_name"] = donor["tool_name"]
                branch["executed_output"] = donor["executed_output"]
                branch["_fault_type"] = "wrong_tool"
                branch["_training_failure_type_id"] = FAILURE_TYPE_TO_ID[
                    "provenance_mismatch"
                ]
                target = 0.0
        elif synthetic_faults and target == 1.0 and mode < 55:
            branch["executed_output"] = (
                "ERROR: upstream API returned 503 Service Unavailable"
            )
            branch["execution_status"] = "failed"
            branch["_fault_type"] = "api_error"
            branch["_training_failure_type_id"] = FAILURE_TYPE_TO_ID[
                "server_or_network"
            ]
            target = 0.0
        elif synthetic_faults and target == 1.0 and mode < 65:
            donor = choose_result_donor(
                row, branch, donor_pool, same_tool=False
            )
            if donor is not None:
                branch["executed_output"] = donor["executed_output"]
                branch["_fault_type"] = "wrong_result_other_tool"
                branch["_training_failure_type_id"] = FAILURE_TYPE_TO_ID[
                    "provenance_mismatch"
                ]
                target = 0.0
        elif synthetic_faults and target == 1.0 and mode < 75:
            branch["executed_output"] = (
                str(branch.get("executed_output", ""))
                + "\nUNTRUSTED_DIAGNOSTIC_NOISE: retry_count=2 "
                "trace_id=abc unrelated cache warning"
            )
            branch["_fault_type"] = "benign_noise"
        targets.append(target)
    return corrupted, torch.tensor(targets, dtype=torch.float32)


@torch.inference_mode()
def encode_texts(model, tokenizer, texts, max_length, batch_size, device=None):
    device = device or next(model.parameters()).device
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    hidden_parts = []
    mask_parts = []
    last_parts = []
    mean_parts = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        ids = tokenized["input_ids"][start:stop].to(device)
        mask = tokenized["attention_mask"][start:stop].to(device)
        output = model.model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state.detach()
        last_index = mask.long().sum(-1).clamp_min(1) - 1
        last = hidden[
            torch.arange(hidden.shape[0], device=device), last_index
        ]
        mean = (
            (hidden * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )
        hidden_parts.append(hidden)
        mask_parts.append(mask.bool())
        last_parts.append(last)
        mean_parts.append(mean)
    return {
        "hidden": torch.cat(hidden_parts),
        "mask": torch.cat(mask_parts),
        "last": torch.cat(last_parts),
        "mean": torch.cat(mean_parts),
    }


def balanced_bce(logits, targets, mask=None):
    logits = logits.flatten()
    targets = targets.float().flatten()
    if mask is not None:
        mask = mask.bool().flatten()
        logits = logits[mask]
        targets = targets[mask]
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives.item() and negatives.item():
        weights = torch.where(
            targets.bool(),
            0.5 * targets.numel() / positives,
            0.5 * targets.numel() / negatives,
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets, weight=weights
        )
    return F.binary_cross_entropy_with_logits(logits, targets)


def coherence_ranking_loss(logits, targets, margin):
    positives = logits[targets.bool()]
    negatives = logits[~targets.bool()]
    if not positives.numel() or not negatives.numel():
        return logits.sum() * 0.0
    return F.relu(
        margin - positives[:, None] + negatives[None, :]
    ).mean()


def dependency_ranking_loss(logits, targets):
    losses = []
    nodes = targets.shape[0]
    for target_index in range(nodes):
        sources = targets[target_index].nonzero(as_tuple=False).flatten()
        if not sources.numel():
            continue
        candidates = torch.ones(nodes, dtype=torch.bool, device=targets.device)
        candidates[target_index] = False
        candidate_indices = candidates.nonzero(as_tuple=False).flatten()
        remapped = (candidate_indices[:, None] == sources[None, :]).any(-1)
        target_distribution = remapped.float()
        target_distribution = target_distribution / target_distribution.sum()
        losses.append(
            -(target_distribution * F.log_softmax(
                logits[target_index, candidate_indices], dim=-1
            )).sum()
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def binary_stats(logits, targets, mask=None):
    prediction = logits.sigmoid() >= 0.5
    target = targets.bool()
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
    tp = (prediction & target).sum().item()
    tn = (~prediction & ~target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": tp / max(tp + fp, 1),
        "recall": recall,
        "specificity": specificity,
    }


def controller_loss(
    controller,
    encoded_action,
    encoded_path,
    encoded_need,
    encoded_produce,
    row,
    branches,
    coherence_target,
    args,
):
    device = encoded_action["last"].device
    dependency, dependency_weight = dependency_targets(branches, device)
    completed, readiness_target, stage = execution_snapshot(
        row, branches, dependency, device
    )
    output = controller(
        encoded_action["mean"],
        encoded_path["hidden"],
        encoded_path["mask"],
        completed_mask=completed,
        need_hidden=encoded_need["mean"],
        produce_hidden=encoded_produce["mean"],
        need_mask=torch.tensor(
            [bool(branch.get("depends_on")) for branch in branches],
            device=device,
        ),
    )
    nodes = len(branches)
    indices = torch.arange(nodes, device=device)
    assignment_loss = 0.5 * (
        F.cross_entropy(output["assignment_logits"], indices)
        + F.cross_entropy(output["assignment_logits"].T, indices)
    )
    assignment_accuracy = (
        output["assignment_logits"].argmax(-1).eq(indices).float().mean()
    )
    assignment_coverage = (
        output["assignment_logits"].argmax(-1).unique().numel() / nodes
    )

    diagonal = torch.eye(nodes, dtype=torch.bool, device=device)
    edge_mask = ~diagonal
    dependency_loss = balanced_bce(
        output["dependency_logits"], dependency, mask=edge_mask
    )
    dependency_rank_loss = dependency_ranking_loss(
        output["dependency_logits"], dependency
    )
    if dependency.sum().item():
        dependency_loss = 0.55 * dependency_loss + 0.45 * dependency_rank_loss
    else:
        dependency_loss = 0.15 * dependency_loss
    readiness_loss = balanced_bce(output["readiness"], readiness_target)

    coherence_target = coherence_target.to(device)
    coherence_logits = output["coherence"]
    coherence_unweighted = F.binary_cross_entropy_with_logits(
        coherence_logits, coherence_target
    )
    coherence_balanced = balanced_bce(
        coherence_logits, coherence_target
    )
    coherence_rank_loss = coherence_ranking_loss(
        coherence_logits, coherence_target, args.coherence_margin
    )
    coherence_brier_loss = (
        coherence_logits.sigmoid() - coherence_target
    ).square().mean()
    coherence_loss = (
        (1.0 - args.coherence_balanced_mix) * coherence_unweighted
        + args.coherence_balanced_mix * coherence_balanced
        + args.coherence_rank_weight * coherence_rank_loss
        + args.coherence_brier_weight * coherence_brier_loss
    )

    failure_source_target = torch.tensor(
        [
            float(
                branch.get("_fault_type") != "natural"
                and coherence_target[index].item() == 0.0
            )
            if branch.get("_fault_type") != "natural"
            else float(branch.get("failure_source_target", 1.0 - coherence_target[index].item()))
            for index, branch in enumerate(branches)
        ],
        device=device,
    )
    failure_source_loss = balanced_bce(
        output["failure_source"], failure_source_target
    )
    failure_type_target = torch.tensor(
        [
            int(
                branch.get(
                    "_training_failure_type_id",
                    FAILURE_TYPE_TO_ID[
                        "success" if coherence_target[index].item() else "other_failure"
                    ],
                )
            )
            for index, branch in enumerate(branches)
        ],
        dtype=torch.long,
        device=device,
    )
    failure_type_loss = F.cross_entropy(
        output["failure_type"], failure_type_target
    )
    task_solvability_target = torch.tensor(
        min(
            float(row.get("task_solvability_target", 1.0)),
            float(coherence_target.bool().all().item()),
        ),
        device=device,
    )
    task_solvability_loss = F.binary_cross_entropy_with_logits(
        output["task_solvability"].reshape(()),
        task_solvability_target,
    )

    answer_text = final_answer_text(row.get("final_answer"))
    utility_target = torch.tensor(
        [
            min(
                1.0,
                0.15 * coherence_target[index].item()
                + 0.85
                * lexical_coverage(branch.get("executed_output"), answer_text),
            )
            for index, branch in enumerate(branches)
        ],
        device=device,
    )
    utility_loss = F.binary_cross_entropy_with_logits(
        output["utility"], utility_target
    )

    signatures = [action_signature(branch) for branch in branches]
    counts = Counter(signatures)
    seen = Counter()
    novelty_values = []
    for signature in signatures:
        seen[signature] += 1
        structurally_novel = counts[signature] == 1 or seen[signature] == 1
        novelty_values.append(
            float(structurally_novel and coherence_target[len(novelty_values)].item())
        )
    novelty_target = torch.tensor(novelty_values, device=device)
    novelty_loss = balanced_bce(output["novelty"], novelty_target)

    unrelated = edge_mask & ~dependency.bool() & ~dependency.bool().T
    distinct = torch.zeros_like(unrelated)
    for left in range(nodes):
        for right in range(nodes):
            distinct[left, right] = signatures[left] != signatures[right]
    coherent_pairs = coherence_target.bool()[:, None] & coherence_target.bool()[None, :]
    repulsive_mask = torch.triu(
        unrelated & distinct & coherent_pairs, diagonal=1
    )
    pair_similarity = output["path_similarity"][repulsive_mask]
    if pair_similarity.numel():
        repulsive_loss = F.relu(pair_similarity - args.repulsive_margin).square().mean()
        mean_similarity = pair_similarity.mean().detach()
    else:
        repulsive_loss = assignment_loss * 0.0
        mean_similarity = torch.zeros((), device=device)

    dependency_prob = output["dependency_prob"]
    identity = torch.eye(nodes, device=device, dtype=dependency_prob.dtype)
    cycle_loss = (
        torch.diagonal(
            torch.linalg.matrix_power(
                identity
                + (dependency_prob * dependency_prob) / max(nodes, 1),
                nodes,
            )
        ).sum()
        - nodes
    ) / max(nodes, 1)

    total = (
        args.assignment_weight * assignment_loss
        + args.dependency_weight * dependency_loss
        + args.readiness_weight * readiness_loss
        + args.coherence_weight * coherence_loss
        + args.failure_source_weight * failure_source_loss
        + args.failure_type_weight * failure_type_loss
        + args.task_solvability_weight * task_solvability_loss
        + args.utility_weight * utility_loss
        + args.novelty_weight * novelty_loss
        + args.repulsive_weight * repulsive_loss
        + args.cycle_weight * cycle_loss
    )
    dep_stats = binary_stats(
        output["dependency_logits"], dependency, mask=edge_mask
    )
    ready_stats = binary_stats(output["readiness"], readiness_target)
    coh_stats = binary_stats(output["coherence"], coherence_target)
    failure_source_stats = binary_stats(
        output["failure_source"], failure_source_target
    )
    fault_counts = Counter(
        str(branch.get("_fault_type", "natural"))
        for branch in branches
    )
    return total, {
        "loss": total.detach().item(),
        "assignment_loss": assignment_loss.detach().item(),
        "dependency_loss": dependency_loss.detach().item(),
        "dependency_rank_loss": dependency_rank_loss.detach().item(),
        "readiness_loss": readiness_loss.detach().item(),
        "coherence_loss": coherence_loss.detach().item(),
        "coherence_unweighted_loss": coherence_unweighted.detach().item(),
        "coherence_balanced_loss": coherence_balanced.detach().item(),
        "coherence_rank_loss": coherence_rank_loss.detach().item(),
        "coherence_brier_loss": coherence_brier_loss.detach().item(),
        "failure_source_loss": failure_source_loss.detach().item(),
        "failure_type_loss": failure_type_loss.detach().item(),
        "task_solvability_loss": task_solvability_loss.detach().item(),
        "utility_loss": utility_loss.detach().item(),
        "novelty_loss": novelty_loss.detach().item(),
        "repulsive_loss": repulsive_loss.detach().item(),
        "cycle_loss": cycle_loss.detach().item(),
        "assignment_accuracy": assignment_accuracy.detach().item(),
        "assignment_coverage": assignment_coverage,
        "dependency_balanced_accuracy": dep_stats["balanced_accuracy"],
        "dependency_recall": dep_stats["recall"],
        "readiness_balanced_accuracy": ready_stats["balanced_accuracy"],
        "readiness_recall": ready_stats["recall"],
        "coherence_balanced_accuracy": coh_stats["balanced_accuracy"],
        "coherence_recall": coh_stats["recall"],
        "failure_source_balanced_accuracy": failure_source_stats[
            "balanced_accuracy"
        ],
        "failure_type_accuracy": output["failure_type"].argmax(-1).eq(
            failure_type_target
        ).float().mean().item(),
        "task_solvability_accuracy": float(
            (output["task_solvability"].sigmoid() >= 0.5).eq(
                task_solvability_target.bool()
            ).item()
        ),
        "task_solvability_target": task_solvability_target.item(),
        "mean_similarity": mean_similarity.item(),
        "coherence_positive_rate": coherence_target.mean().item(),
        "stale_result_negatives": fault_counts["stale_same_tool_result"],
        "wrong_parameter_negatives": fault_counts["wrong_parameters"],
        "wrong_tool_negatives": fault_counts["wrong_tool"],
        "api_error_negatives": fault_counts["api_error"],
        "benign_noise_positives": fault_counts["benign_noise"],
        "ready_positive_rate": readiness_target.mean().item(),
        "dependency_edges": int(dependency.sum().item()),
        "explicit_dependency_weight": dependency_weight.sum().item(),
        "completed_nodes": int(completed.sum().item()),
        "stage": stage,
        "nodes": nodes,
    }


def load_base(args):
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_memory_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0


def empty_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def atomic_save(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(path, controller, optimizer, step, args, history, train_ids, eval_ids):
    payload = {
        "version": 9,
        "training_mode": "failure_aware_dynamic_action_dag_no_eagle",
        "step": step,
        "model": controller.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "history": history,
        "train_task_ids": train_ids,
        "eval_task_ids": eval_ids,
    }
    atomic_save(path, payload)
    archive = os.path.join(
        os.path.dirname(path), f"{args.checkpoint_prefix}_step_{step}.pt"
    )
    atomic_save(archive, payload)


def append_jsonl(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def average(history, window):
    rows = history[-window:]
    keys = [
        key
        for key, value in rows[-1].items()
        if isinstance(value, (int, float))
        and key not in {"step", "elapsed_seconds", "peak_memory_gb"}
    ]
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
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
        "--data", default="data/traject_swarm_training_failure_aware.jsonl"
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--eval-samples", type=int, default=120)
    parser.add_argument("--positive-ratio", type=float, default=0.75)
    parser.add_argument(
        "--negative-parallel-fraction", type=float, default=0.2
    )
    parser.add_argument("--max-nodes", type=int, default=13)
    parser.add_argument("--action-tokens", type=int, default=128)
    parser.add_argument("--path-tokens", type=int, default=160)
    parser.add_argument("--path-encode-tokens", type=int)
    parser.add_argument("--parameter-tokens", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--set-heads", type=int, default=8)
    parser.add_argument("--assignment-weight", type=float, default=1.0)
    parser.add_argument("--dependency-weight", type=float, default=0.9)
    parser.add_argument("--readiness-weight", type=float, default=0.8)
    parser.add_argument("--coherence-weight", type=float, default=0.8)
    parser.add_argument("--coherence-balanced-mix", type=float, default=0.2)
    parser.add_argument("--coherence-rank-weight", type=float, default=0.2)
    parser.add_argument("--coherence-brier-weight", type=float, default=0.1)
    parser.add_argument("--coherence-margin", type=float, default=0.5)
    parser.add_argument("--failure-source-weight", type=float, default=0.6)
    parser.add_argument("--failure-type-weight", type=float, default=0.35)
    parser.add_argument("--task-solvability-weight", type=float, default=0.8)
    parser.add_argument("--utility-weight", type=float, default=0.25)
    parser.add_argument("--novelty-weight", type=float, default=0.35)
    parser.add_argument("--repulsive-weight", type=float, default=0.08)
    parser.add_argument("--cycle-weight", type=float, default=0.03)
    parser.add_argument("--repulsive-margin", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmstart-lr-scale", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--checkpoint-prefix", default="rcg_failure_aware_v9")
    parser.add_argument(
        "--checkpoint", default="checkpoints/rcg_failure_aware_v9_latest.pt"
    )
    parser.add_argument("--log-file", default="logs/rcg_failure_aware_v9.jsonl")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--synthetic-faults", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.path_encode_tokens = args.path_encode_tokens or args.path_tokens
    args.device = resolve_device(args.device)
    device = torch.device(args.device)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rows = read_rows(args.data)
    training, evaluation = ratio_split(
        rows,
        args.samples,
        args.eval_samples,
        args.positive_ratio,
        args.seed,
        args.negative_parallel_fraction,
    )
    resume_payload = None
    if args.resume_checkpoint:
        resume_payload = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        by_id = {row["task_id"]: row for row in rows}
        training = [
            by_id[task_id] for task_id in resume_payload["train_task_ids"]
        ]
        evaluation = [
            by_id[task_id] for task_id in resume_payload["eval_task_ids"]
        ]
    if len(training) != args.samples:
        raise RuntimeError(f"requested {args.samples}, found {len(training)}")
    train_task_counts = Counter(
        int(row["task_solvability_target"]) for row in training
    )
    train_branch_counts = Counter(
        int(branch["effective_coherence_target"])
        for row in training
        for branch in row["branches"]
    )
    print(
        f"Training composition tasks +:{train_task_counts[1]} "
        f"-:{train_task_counts[0]}; branches +:{train_branch_counts[1]} "
        f"-:{train_branch_counts[0]}; "
        f"branch_ratio={train_branch_counts[1] / max(train_branch_counts[0], 1):.3f}",
        flush=True,
    )

    print(
        f"Loading frozen Qwen encoder on {device}; "
        "no EAGLE and no mode label.",
        flush=True,
    )
    base, tokenizer = load_base(args)
    controller = DynamicRCG(
        hidden_size=base.config.hidden_size,
        latent_dim=args.latent_dim,
        set_layers=args.set_layers,
        set_heads=args.set_heads,
        max_chain_tokens=args.path_tokens,
        failure_classes=len(FAILURE_TYPES),
    ).to(device).train()
    if resume_payload is not None:
        controller.load_state_dict(resume_payload["model"], strict=True)
        print(
            f"Resuming {args.resume_checkpoint} at "
            f"step {resume_payload['step']}.",
            flush=True,
        )
    elif args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        incompatible = controller.load_state_dict(
            initial["model"], strict=False
        )
        expected_missing = {
            name
            for name in controller.state_dict()
            if name.startswith(
                ("failure_source_head.", "failure_type_head.", "task_solvability_head.")
            )
        }
        unexpected_missing = set(incompatible.missing_keys) - expected_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected warm-start mismatch: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        print(
            f"Warm-started {args.init_checkpoint}; "
            f"initialized {len(incompatible.missing_keys)} new-head tensors.",
            flush=True,
        )
    new_head_prefixes = (
        "failure_source_head.",
        "failure_type_head.",
        "task_solvability_head.",
    )
    old_parameters = []
    new_parameters = []
    for name, parameter in controller.named_parameters():
        (
            new_parameters
            if name.startswith(new_head_prefixes)
            else old_parameters
        ).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": old_parameters,
                "lr": args.lr * args.warmstart_lr_scale,
            },
            {"params": new_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    print(
        f"DynamicRCG parameters={dynamic_parameter_count(controller):,}; "
        f"train={len(training)} eval={len(evaluation)}",
        flush=True,
    )

    train_ids = [row["task_id"] for row in training]
    eval_ids = [row["task_id"] for row in evaluation]
    result_donor_pool = build_result_donor_pool(
        training, args.max_nodes
    )
    history = list(resume_payload.get("history", [])) if resume_payload else []
    initial_step = int(resume_payload.get("step", 0)) if resume_payload else 0
    prior_elapsed = (
        float(history[-1].get("elapsed_seconds", 0.0)) if history else 0.0
    )
    started = time.time() - prior_elapsed
    optimizer.zero_grad(set_to_none=True)
    for step, row in enumerate(
        training[initial_step:], start=initial_step + 1
    ):
        branches = select_nodes(row, args.max_nodes)
        path_branches, coherence_target = corrupt_results(
            row,
            branches,
            result_donor_pool,
            synthetic_faults=args.synthetic_faults,
        )
        encoded_action = encode_texts(
            base,
            tokenizer,
            [render_action(row, branch) for branch in branches],
            args.action_tokens,
            args.encode_batch_size,
            device,
        )
        encoded_path = encode_texts(
            base,
            tokenizer,
            [render_path(row, branch) for branch in path_branches],
            args.path_encode_tokens,
            args.encode_batch_size,
            device,
        )
        encoded_need = encode_texts(
            base,
            tokenizer,
            [unresolved_need_text(branch) for branch in branches],
            args.parameter_tokens,
            args.encode_batch_size,
            device,
        )
        encoded_produce = encode_texts(
            base,
            tokenizer,
            [produced_value_text(branch) for branch in branches],
            args.parameter_tokens,
            args.encode_batch_size,
            device,
        )
        total, metrics = controller_loss(
            controller,
            encoded_action,
            encoded_path,
            encoded_need,
            encoded_produce,
            row,
            path_branches,
            coherence_target,
            args,
        )
        (total / args.grad_accum).backward()
        if step % args.grad_accum == 0 or step == len(training):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                controller.parameters(), args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metrics["grad_norm"] = float(grad_norm)
            metrics["optimizer_update"] = True
        else:
            metrics["grad_norm"] = 0.0
            metrics["optimizer_update"] = False

        metrics.update(
            {
                "step": step,
                "task_id": row["task_id"],
                "domain": row["domain"],
                "source_family": str(row.get("source", "")).split("/", 1)[-1],
                "elapsed_seconds": time.time() - started,
                "peak_memory_gb": peak_memory_gb(device),
            }
        )
        history.append(metrics)
        append_jsonl(args.log_file, metrics)
        del total, encoded_action, encoded_path, encoded_need, encoded_produce

        if step % args.log_every == 0:
            recent = average(history, args.log_every)
            print(
                f"Step {step}: loss={recent['loss']:.4f} "
                f"assign={recent['assignment_accuracy']:.3f}/"
                f"{recent['assignment_coverage']:.3f} "
                f"dep_bal={recent['dependency_balanced_accuracy']:.3f} "
                f"ready_bal={recent['readiness_balanced_accuracy']:.3f} "
                f"coh_bal={recent['coherence_balanced_accuracy']:.3f} "
                f"fail_src={recent['failure_source_balanced_accuracy']:.3f} "
                f"solv={recent['task_solvability_accuracy']:.3f} "
                f"sim={recent['mean_similarity']:.3f} "
                f"peak={metrics['peak_memory_gb']:.2f}GB",
                flush=True,
            )
        if step % args.save_every == 0 or step == len(training):
            save_checkpoint(
                args.checkpoint,
                controller,
                optimizer,
                step,
                args,
                history,
                train_ids,
                eval_ids,
            )
            print(f"Saved checkpoint at step {step}", flush=True)
        if step % 25 == 0:
            empty_device_cache(device)

    print(f"Finished {len(training)} unified DAG steps.", flush=True)


if __name__ == "__main__":
    main()
