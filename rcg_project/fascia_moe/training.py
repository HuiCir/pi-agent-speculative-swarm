from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .config import FasciaConfig
from .encoding import EncoderCache, encode_row
from .losses import fascia_loss
from .model import FasciaMoE
from .schema import validate_event


def atomic_save(path: Path, value) -> None:
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


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_rows(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_event(row)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def stable_bucket(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10000


def split_rows(
    rows: list[dict], eval_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    training, evaluation = [], []
    for row in rows:
        target = (
            evaluation
            if stable_bucket(row["group_id"], seed) < int(eval_ratio * 10000)
            else training
        )
        target.append(row)
    if not training or not evaluation:
        raise ValueError("empty train/evaluation split")
    return training, evaluation


def sequence_order(row: dict) -> tuple[int, str]:
    task_id = row["task_id"]
    if "::plan" in task_id:
        return (0, task_id)
    if "::snapshot_" in task_id:
        try:
            return (1 + int(task_id.rsplit("::snapshot_", 1)[1]), task_id)
        except ValueError:
            return (1, task_id)
    if "::resolved" in task_id:
        return (1000, task_id)
    return (999, task_id)


def group_sequences(rows: list[dict]) -> dict[str, list[list[dict]]]:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["source_type"]][row["group_id"]].append(row)
    return {
        source: [
            sorted(sequence, key=sequence_order)
            for _, sequence in sorted(source_groups.items())
        ]
        for source, source_groups in groups.items()
    }


DEFAULT_SOURCE_WEIGHTS = {
    "trajectory": 0.44,
    "mmlu": 0.20,
    "swe_bench": 0.12,
    "longbench": 0.12,
    "clawbench": 0.12,
    "tau_bench": 0.18,
    "tau2_airline": 0.10,
    "tau2_retail": 0.14,
    "tau2_telecom": 0.22,
    "tau2_banking": 0.08,
    "long_horizon_v2": 0.16,
    "swarm_protocol_v3": 0.30,
    "hotpotqa": 0.14,
    "gaia": 0.08,
    "gaia_text": 0.08,
}


def build_schedule(
    rows: list[dict],
    steps: int,
    seed: int,
    source_weights: dict[str, float] | None = None,
) -> list[dict]:
    sequences = group_sequences(rows)
    available = sorted(sequences)
    merged_weights = dict(DEFAULT_SOURCE_WEIGHTS)
    if source_weights:
        merged_weights.update(
            {
                str(source): max(0.0, float(weight))
                for source, weight in source_weights.items()
            }
        )
    weights = [
        merged_weights.get(source, 0.1) for source in available
    ]
    if not any(weight > 0 for weight in weights):
        raise ValueError("at least one source weight must be positive")
    randomizer = random.Random(seed)
    positions = {source: 0 for source in available}
    for source in available:
        randomizer.shuffle(sequences[source])
    schedule = []
    while len(schedule) < steps:
        source = randomizer.choices(available, weights=weights, k=1)[0]
        source_sequences = sequences[source]
        position = positions[source]
        if position >= len(source_sequences):
            randomizer.shuffle(source_sequences)
            position = 0
        sequence = source_sequences[position]
        positions[source] = position + 1
        for row in sequence:
            schedule.append(row)
            if len(schedule) >= steps:
                break
    return schedule


def _binary_counts(
    logits: torch.Tensor,
    targets: list[float],
    mask: list[float] | None = None,
) -> Counter:
    predicted = logits.sigmoid().detach().cpu() >= 0.5
    truth = torch.tensor(targets) >= 0.5
    active = (
        torch.tensor(mask) >= 0.5
        if mask is not None
        else torch.ones_like(truth)
    )
    predicted = predicted[active]
    truth = truth[active]
    return Counter(
        {
            "tp": int((predicted & truth).sum()),
            "tn": int((~predicted & ~truth).sum()),
            "fp": int((predicted & ~truth).sum()),
            "fn": int((~predicted & truth).sum()),
        }
    )


def _binary_metrics(
    counts: Counter,
) -> dict[str, float | bool | None]:
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    if total == 0:
        return {
            "available": False,
            "accuracy": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    class_recalls = []
    if tp + fn:
        class_recalls.append(recall)
    if tn + fp:
        class_recalls.append(specificity)
    return {
        "available": True,
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": sum(class_recalls) / len(class_recalls),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }


@torch.inference_mode()
def evaluate(
    model: FasciaMoE,
    rows: list[dict],
    cache: EncoderCache,
    device: torch.device,
    limit: int = 0,
) -> dict:
    model.eval()
    selected_rows = rows[:limit] if limit else rows
    counters = {
        source: {
            "admission": Counter(),
            "coherence": Counter(),
            "halt": Counter(),
            "takeover": Counter(),
            "report": Counter(),
            "exact": 0,
            "count": 0,
            "stage_correct": 0,
            "stage_total": 0,
            "branch_correct": 0,
            "router_hit": 0,
        }
        for source in sorted({row["source_type"] for row in selected_rows})
    }
    state_by_group: dict[str, torch.Tensor] = {}
    total_loss = 0.0
    for row in sorted(
        selected_rows,
        key=lambda item: (item["group_id"], sequence_order(item)),
    ):
        tensors = encode_row(row, cache, device)
        previous = state_by_group.get(row["group_id"])
        output = model(**tensors, previous_state=previous)
        state_by_group[row["group_id"]] = output["recurrent_state"].detach()
        loss, _ = fascia_loss(output, row)
        total_loss += float(loss)
        source = row["source_type"]
        store = counters[source]
        labels = row["labels"]
        mask = labels.get("mask") or {}
        store["admission"].update(
            _binary_counts(
                output["admission"], labels["node"]["admission"]
            )
        )
        store["coherence"].update(
            _binary_counts(
                output["coherence"],
                labels["node"]["coherence"],
                mask.get("coherence"),
            )
        )
        for key, target_name in (
            ("halt", "halt"),
            ("takeover", "takeover"),
            ("report", "must_report_failure"),
        ):
            store[key].update(
                _binary_counts(
                    output[target_name].reshape(1),
                    [labels["global"][target_name]],
                )
            )
        admission = output["admission"].sigmoid() >= 0.5
        truth = torch.tensor(
            labels["node"]["admission"], device=device
        ) >= 0.5
        store["exact"] += int(torch.equal(admission, truth))
        store["count"] += 1
        stage = output["stage_logits"].argmax(-1).cpu()
        stage_target = torch.tensor(labels["node"]["stage"])
        store["stage_correct"] += int((stage == stage_target).sum())
        store["stage_total"] += len(stage_target)
        store["branch_correct"] += int(
            int(output["branch_count_logits"].argmax())
            == int(labels["global"]["branch_count"])
        )
        target_experts = torch.tensor(labels["expert_target"]).topk(
            model.config.top_k
        ).indices.tolist()
        active = set(output["active_experts"].cpu().tolist())
        store["router_hit"] += int(bool(active & set(target_experts)))
    per_source = {}
    composite_values = []
    for source, store in counters.items():
        admission = _binary_metrics(store["admission"])
        coherence = _binary_metrics(store["coherence"])
        halt = _binary_metrics(store["halt"])
        takeover = _binary_metrics(store["takeover"])
        report = _binary_metrics(store["report"])
        count = max(store["count"], 1)
        summary = {
            "rows": store["count"],
            "admission": admission,
            "coherence": coherence,
            "halt": halt,
            "takeover": takeover,
            "must_report_failure": report,
            "exact_set": store["exact"] / count,
            "branch_count_accuracy": store["branch_correct"] / count,
            "stage_accuracy": store["stage_correct"]
            / max(store["stage_total"], 1),
            "router_topk_hit": store["router_hit"] / count,
        }
        components = [
            (0.25, admission["f1"]),
            (0.15, admission["recall"]),
            (0.12, coherence["balanced_accuracy"]),
            (0.10, summary["exact_set"]),
            (0.08, summary["branch_count_accuracy"]),
            (0.08, summary["stage_accuracy"]),
            (0.07, halt["balanced_accuracy"]),
            (0.05, takeover["balanced_accuracy"]),
            (0.05, report["balanced_accuracy"]),
            (0.05, summary["router_topk_hit"]),
        ]
        available_components = [
            (weight, value)
            for weight, value in components
            if value is not None
        ]
        summary["composite"] = sum(
            weight * value for weight, value in available_components
        ) / max(
            sum(weight for weight, _ in available_components),
            1e-9,
        )
        per_source[source] = summary
        composite_values.append(summary["composite"])
    model.train()
    return {
        "loss": total_loss / max(len(selected_rows), 1),
        "composite": sum(composite_values) / max(len(composite_values), 1),
        "per_source": per_source,
        "rows": len(selected_rows),
    }


class TopCheckpointManager:
    def __init__(self, directory: Path, keep: int = 3):
        self.directory = directory
        self.keep = keep
        self.entries: list[dict] = []
        self.manifest = directory / "best_manifest.json"
        if self.manifest.exists():
            self.entries = json.loads(
                self.manifest.read_text(encoding="utf-8")
            ).get("checkpoints", [])

    def consider(
        self,
        score: float,
        step: int,
        payload: dict,
    ) -> bool:
        if len(self.entries) >= self.keep and score <= min(
            entry["score"] for entry in self.entries
        ):
            return False
        path = self.directory / f"fascia_best_step_{step}_{score:.5f}.pt"
        model_payload = {
            key: value
            for key, value in payload.items()
            if key != "optimizer"
        }
        atomic_save(path, model_payload)
        self.entries.append(
            {"score": score, "step": step, "path": str(path)}
        )
        self.entries.sort(key=lambda item: item["score"], reverse=True)
        while len(self.entries) > self.keep:
            removed = self.entries.pop()
            Path(removed["path"]).unlink(missing_ok=True)
        atomic_json(
            self.manifest,
            {"checkpoints": self.entries, "updated_step": step},
        )
        return True


def checkpoint_payload(
    model: FasciaMoE,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_config: dict,
    validation: dict,
) -> dict:
    return {
        "version": str(train_config.get("checkpoint_version", "fascia_moe_v1")),
        "step": step,
        "model_config": model.config.to_dict(),
        "train_config": train_config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validation": validation,
        "parameter_summary": model.parameter_summary(),
    }


def train(
    *,
    model: FasciaMoE,
    training_rows: list[dict],
    evaluation_rows: list[dict],
    cache: EncoderCache,
    device: torch.device,
    train_config: dict,
    output_dir: Path,
    resume_payload: dict | None = None,
) -> dict:
    steps = int(train_config["steps"])
    gradient_accumulation = int(
        train_config.get("gradient_accumulation", 1)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    start_step = 0
    resumed_from = ""
    if resume_payload is not None:
        if "optimizer" not in resume_payload:
            raise ValueError(
                "resume checkpoint must include optimizer state; use "
                "init_checkpoint for model-only warm starts"
            )
        start_step = max(0, min(int(resume_payload.get("step", 0)), steps))
        optimizer.load_state_dict(resume_payload["optimizer"])
        resumed_from = str(resume_payload.get("checkpoint_path") or "")
    schedule = build_schedule(
        training_rows,
        steps,
        int(train_config.get("seed", 42)),
        train_config.get("source_weights"),
    )
    history_path = output_dir / "logs" / "training.jsonl"
    if start_step <= 0:
        history_path.unlink(missing_ok=True)
    latest_path = output_dir / "checkpoints" / "fascia_latest.pt"
    manager = TopCheckpointManager(output_dir / "checkpoints", keep=3)
    validate_every = int(train_config.get("validate_every", 200))
    save_every = int(train_config.get("save_every", 700))
    warmup_steps = int(train_config.get("router_teacher_steps", steps // 10))
    eval_limit = int(train_config.get("eval_limit", 300))
    max_grad_norm = float(train_config.get("max_grad_norm", 1.0))
    regression_grace = int(train_config.get("regression_grace_steps", 800))
    regression_margin = float(train_config.get("regression_margin", 0.12))
    regression_patience = int(train_config.get("regression_patience", 3))
    regression_count = 0
    best_score = max(
        [float(entry.get("score", -1.0)) for entry in manager.entries]
        or [-1.0]
    )
    start = time.time()
    state_by_group: dict[str, torch.Tensor] = {}

    baseline = evaluate(
        model, evaluation_rows, cache, device, limit=eval_limit
    )
    atomic_json(
        output_dir
        / "eval"
        / ("baseline_resume.json" if start_step else "baseline_untrained.json"),
        baseline,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    validation_history_path = output_dir / "eval" / "validation_history.json"
    validations = []
    if start_step and validation_history_path.exists():
        validations = json.loads(validation_history_path.read_text(encoding="utf-8"))
    aborted = False
    abort_reason = ""
    if start_step >= steps:
        final_validation = baseline
        final_validation["step"] = start_step
        summary = {
            "aborted": False,
            "abort_reason": "",
            "step": start_step,
            "best_score": best_score,
            "baseline": baseline,
            "final_validation": final_validation,
            "top_checkpoints": manager.entries,
            "elapsed_seconds": time.time() - start,
            "resumed_from": resumed_from,
            "start_step": start_step,
            "already_complete": True,
        }
        atomic_json(output_dir / "training_summary.json", summary)
        return summary
    for step, row in enumerate(schedule[start_step:], start=start_step + 1):
        tensors = encode_row(row, cache, device)
        previous = state_by_group.get(row["group_id"])
        expert_override = None
        if step <= warmup_steps:
            expert_override = torch.tensor(
                row["labels"]["expert_target"],
                device=device,
            ).topk(model.config.top_k).indices
        output = model(
            **tensors,
            previous_state=previous,
            expert_override=expert_override,
        )
        next_state = output["recurrent_state"].detach()
        if torch.isfinite(next_state).all():
            state_by_group[row["group_id"]] = next_state
        else:
            state_by_group.pop(row["group_id"], None)
        loss, metrics = fascia_loss(output, row)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            state_by_group.pop(row["group_id"], None)
            metrics.update(
                {
                    "step": step,
                    "source_type": row["source_type"],
                    "task_id": row["task_id"],
                    "grad_norm": float("nan"),
                    "elapsed_seconds": time.time() - start,
                    "active_experts": output["active_experts"]
                    .detach()
                    .cpu()
                    .tolist(),
                    "skipped_update": True,
                    "skip_reason": "nonfinite_loss",
                }
            )
            append_jsonl(history_path, metrics)
            continue
        (loss / gradient_accumulation).backward()
        grad_norm = 0.0
        skipped_update = False
        skip_reason = ""
        if step % gradient_accumulation == 0 or step == steps:
            grad_norm_value = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            grad_norm = float(grad_norm_value)
            if torch.isfinite(grad_norm_value):
                optimizer.step()
            else:
                skipped_update = True
                skip_reason = "nonfinite_grad"
                state_by_group.clear()
            optimizer.zero_grad(set_to_none=True)
        metrics.update(
            {
                "step": step,
                "source_type": row["source_type"],
                "task_id": row["task_id"],
                "grad_norm": grad_norm,
                "elapsed_seconds": time.time() - start,
                "active_experts": output["active_experts"]
                .detach()
                .cpu()
                .tolist(),
                "skipped_update": skipped_update,
                "skip_reason": skip_reason,
            }
        )
        append_jsonl(history_path, metrics)

        should_validate = (
            step == 1 or step % validate_every == 0 or step == steps
        )
        if should_validate:
            validation = evaluate(
                model,
                evaluation_rows,
                cache,
                device,
                limit=eval_limit,
            )
            validation["step"] = step
            validations.append(validation)
            score = float(validation["composite"])
            best_score = max(best_score, score)
            payload = checkpoint_payload(
                model, optimizer, step, train_config, validation
            )
            if resumed_from:
                payload["resumed_from"] = resumed_from
                payload["start_step"] = start_step
            manager.consider(score, step, payload)
            atomic_json(validation_history_path, validations)
            print(
                f"step={step} loss={metrics['loss']:.4f} "
                f"val={score:.4f} best={best_score:.4f}",
                flush=True,
            )
            if (
                step >= regression_grace
                and score < best_score - regression_margin
            ):
                regression_count += 1
            else:
                regression_count = 0
            if regression_count >= regression_patience:
                aborted = True
                abort_reason = (
                    f"validation composite regressed by more than "
                    f"{regression_margin:.3f} for {regression_count} checks"
                )
                atomic_save(latest_path, payload)
                break

        if step % save_every == 0:
            validation = validations[-1] if validations else baseline
            payload = checkpoint_payload(
                model, optimizer, step, train_config, validation
            )
            if resumed_from:
                payload["resumed_from"] = resumed_from
                payload["start_step"] = start_step
            atomic_save(latest_path, payload)
    final_step = validations[-1]["step"] if validations else start_step
    final_validation = validations[-1] if validations else baseline
    if not latest_path.exists() or not aborted:
        payload = checkpoint_payload(
            model,
            optimizer,
            final_step,
            train_config,
            final_validation,
        )
        if resumed_from:
            payload["resumed_from"] = resumed_from
            payload["start_step"] = start_step
        atomic_save(latest_path, payload)
    summary = {
        "aborted": aborted,
        "abort_reason": abort_reason,
        "step": final_step,
        "best_score": best_score,
        "baseline": baseline,
        "final_validation": final_validation,
        "top_checkpoints": manager.entries,
        "elapsed_seconds": time.time() - start,
        "resumed_from": resumed_from,
        "start_step": start_step,
    }
    atomic_json(output_dir / "training_summary.json", summary)
    return summary
