from __future__ import annotations

import torch
import torch.nn.functional as F


def _tensor(value, device, dtype=torch.float32):
    return torch.tensor(value, device=device, dtype=dtype)


def balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        active = mask > 0.5
        logits = logits[active]
        targets = targets[active]
    if not logits.numel():
        return targets.sum() * 0.0
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
    return sum(losses) / len(losses)


def pair_loss(logits: torch.Tensor, values: list[list[float]]) -> torch.Tensor:
    target = _tensor(values, logits.device)
    count = target.shape[0]
    diagonal = torch.eye(count, dtype=torch.bool, device=target.device)
    return balanced_bce(logits[~diagonal], target[~diagonal])


def fascia_loss(output: dict, row: dict) -> tuple[torch.Tensor, dict]:
    device = output["admission"].device
    labels = row["labels"]
    node = labels["node"]
    pair = labels["pair"]
    global_labels = labels["global"]
    masks = labels.get("mask") or {}
    admission_target = _tensor(node["admission"], device)
    coherence_target = _tensor(node["coherence"], device)
    recovery_target = _tensor(node["recovery"], device, torch.long)
    coherence_mask = _tensor(
        masks.get("coherence", [1.0] * len(admission_target)), device
    )
    recovery_mask = _tensor(
        masks.get("recovery", [1.0] * len(admission_target)), device
    )
    recovery_active = recovery_mask > 0.5

    parts = {
        "admission": balanced_bce(output["admission"], admission_target),
        "priority": F.smooth_l1_loss(
            output["priority"].sigmoid(), _tensor(node["priority"], device)
        ),
        "stage": F.cross_entropy(
            output["stage_logits"], _tensor(node["stage"], device, torch.long)
        ),
        "coherence": balanced_bce(
            output["coherence"], coherence_target, coherence_mask
        ),
        "contribution": balanced_bce(
            output["contribution"], _tensor(node["contribution"], device)
        ),
        "novelty": balanced_bce(
            output["novelty"], _tensor(node["novelty"], device)
        ),
        "memory_keep": balanced_bce(
            output["memory_keep"], _tensor(node["memory_keep"], device)
        ),
        "recovery": (
            F.cross_entropy(
                output["recovery_logits"][recovery_active],
                recovery_target[recovery_active],
            )
            if recovery_active.any()
            else output["recovery_logits"].sum() * 0.0
        ),
        "dependency": pair_loss(output["dependency"], pair["dependency"]),
        "residual_route": pair_loss(
            output["residual_route"], pair["residual_route"]
        ),
        "branch_count": F.cross_entropy(
            output["branch_count_logits"].unsqueeze(0),
            _tensor(
                [global_labels["branch_count"]], device, torch.long
            ),
        ),
        "initial_wave": F.cross_entropy(
            output["initial_wave_logits"].unsqueeze(0),
            _tensor(
                [global_labels["initial_wave_size"]], device, torch.long
            ),
        ),
        "later_wave": F.cross_entropy(
            output["later_wave_logits"].unsqueeze(0),
            _tensor(
                [global_labels["later_wave_size"]], device, torch.long
            ),
        ),
        "wave_count": F.cross_entropy(
            output["wave_count_logits"].unsqueeze(0),
            _tensor([global_labels["wave_count"]], device, torch.long),
        ),
    }
    for name in (
        "task_complete",
        "task_solvable",
        "must_report_failure",
        "takeover",
        "halt",
    ):
        if name == "task_solvable" and not masks.get(
            "task_solvable", 1.0
        ):
            parts[name] = output[name].sum() * 0.0
        else:
            parts[name] = F.binary_cross_entropy_with_logits(
                output[name],
                _tensor(global_labels[name], device),
            )
    parts["remaining_turns"] = F.smooth_l1_loss(
        output["remaining_turns"],
        _tensor(global_labels["remaining_turns"], device),
    )
    expert_target = _tensor(labels["expert_target"], device)
    expert_target = expert_target / expert_target.sum().clamp_min(1e-6)
    parts["router"] = F.kl_div(
        F.log_softmax(output["router_logits"], dim=-1),
        expert_target,
        reduction="sum",
    )
    parts["router_z"] = torch.logsumexp(
        output["router_logits"], dim=-1
    ).square()
    predicted_count = output["admission"].sigmoid().sum()
    parts["count_consistency"] = F.smooth_l1_loss(
        predicted_count,
        _tensor(global_labels["branch_count"], device),
    )
    parts["halt_consistency"] = (
        F.relu(
            output["halt"].sigmoid()
            - output["task_complete"].sigmoid()
        )
        + F.relu(
            output["takeover"].sigmoid()
            - output["task_complete"].sigmoid()
        )
        + F.relu(
            output["must_report_failure"].sigmoid()
            + output["task_solvable"].sigmoid()
            - 1.0
        )
    )
    weights = {
        "admission": 1.4,
        "priority": 0.3,
        "stage": 0.7,
        "coherence": 1.2,
        "contribution": 1.0,
        "novelty": 0.5,
        "memory_keep": 0.7,
        "recovery": 0.7,
        "dependency": 0.9,
        "residual_route": 0.8,
        "branch_count": 0.8,
        "initial_wave": 0.6,
        "later_wave": 0.4,
        "wave_count": 0.5,
        "task_complete": 0.8,
        "task_solvable": 0.7,
        "must_report_failure": 0.7,
        "takeover": 0.7,
        "halt": 0.8,
        "remaining_turns": 0.4,
        "router": 0.5,
        "router_z": 0.001,
        "count_consistency": 0.5,
        "halt_consistency": 0.5,
    }
    metadata_weights = row.get("metadata", {}).get("loss_weights") or {}
    for name, value in metadata_weights.items():
        if name in weights:
            weights[name] = float(value)
    loss = sum(weights[name] * value for name, value in parts.items())
    teacher_weight = float(row.get("metadata", {}).get("teacher_weight", 1.0))
    loss = loss * teacher_weight
    metrics = {
        f"loss_{name}": float(value.detach()) for name, value in parts.items()
    }
    metrics["loss"] = float(loss.detach())
    return loss, metrics
