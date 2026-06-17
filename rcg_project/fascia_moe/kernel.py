from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KernelLimits:
    max_concurrency: int = 8
    max_branches: int = 24
    admission_threshold: float = 0.5
    coherence_threshold: float = 0.5


def _acyclic_edges(scores: torch.Tensor, selected: list[int]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    reachability: dict[int, set[int]] = {index: set() for index in selected}
    candidates = []
    for target in selected:
        for source in selected:
            if source != target:
                candidates.append(
                    (
                        float(scores[target, source].detach()),
                        source,
                        target,
                    )
                )
    for score, source, target in sorted(candidates, reverse=True):
        if score < 0.5 or source in reachability[target]:
            continue
        edges.append((source, target))
        downstream = {target, *reachability[target]}
        upstream = [node for node in selected if source == node or source in reachability[node]]
        for node in upstream:
            reachability[node].update(downstream)
    return edges


def compile_policy(output: dict[str, torch.Tensor], limits: KernelLimits) -> dict:
    admission = output["admission"].sigmoid()
    predicted_count = int(output["branch_count_logits"].argmax().item())
    count = min(limits.max_branches, max(1, predicted_count))
    ranked = admission.argsort(descending=True).tolist()
    selected = [
        index for index in ranked if admission[index] >= limits.admission_threshold
    ][:count]
    if not selected:
        selected = ranked[:1]
    wave_size = int(output["initial_wave_logits"].argmax().item())
    wave_size = min(limits.max_concurrency, max(1, wave_size))
    priority = output["priority"].sigmoid()
    selected.sort(
        key=lambda index: float(priority[index].detach()), reverse=True
    )
    dependencies = _acyclic_edges(output["dependency"].sigmoid(), selected)
    coherent = [
        index
        for index in selected
        if output["coherence"][index].sigmoid() >= limits.coherence_threshold
    ]
    return {
        "selected": selected,
        "initial_wave": selected[:wave_size],
        "dependencies": dependencies,
        "coherent": coherent,
        "halt": bool(output["halt"].sigmoid() >= 0.5),
        "takeover": bool(output["takeover"].sigmoid() >= 0.5),
        "must_report_failure": bool(
            output["must_report_failure"].sigmoid() >= 0.5
        ),
        "active_experts": output["active_experts"].tolist(),
    }
