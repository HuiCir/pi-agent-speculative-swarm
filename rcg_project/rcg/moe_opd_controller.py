"""Trajectory-coupled RCG MoE controller for OPD training."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def set_encoder(width: int, layers: int, heads: int) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * 4,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,
        dtype=torch.float32,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=layers,
        enable_nested_tensor=False,
    )


def projection(input_width: int, output_width: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_width, output_width, bias=False, dtype=torch.float32),
        nn.LayerNorm(output_width, dtype=torch.float32),
    )


def residual_expert(input_width: int, width: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_width, width * 2, dtype=torch.float32),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(width * 2, width, dtype=torch.float32),
        nn.LayerNorm(width, dtype=torch.float32),
    )


class RcgMoeOpdV1(nn.Module):
    """Large shared trajectory expert with three small execution-stage experts.

    The shared expert is deliberately used by every forward path. Warm-start
    phases keep it trainable while activating one specialist at a time; joint
    OPD then optimizes the complete trajectory decision.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        latent_dim: int = 768,
        shared_layers: int = 6,
        planner_layers: int = 2,
        outcome_layers: int = 2,
        orchestration_layers: int = 2,
        set_heads: int = 8,
        failure_classes: int = 10,
        recovery_classes: int = 6,
        max_branches: int = 16,
        max_waves: int = 8,
        max_concurrency: int = 8,
        coupled_selection: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.failure_classes = failure_classes
        self.recovery_classes = recovery_classes
        self.max_branches = max_branches
        self.max_waves = max_waves
        self.max_concurrency = max_concurrency
        self.coupled_selection = coupled_selection

        # The shared expert owns the dominant representation capacity.
        self.shared_projection = projection(hidden_size, latent_dim)
        self.shared_query_projection = projection(hidden_size, latent_dim)
        self.shared_action_encoder = set_encoder(
            latent_dim, shared_layers, set_heads
        )
        self.shared_path_adapter = residual_expert(latent_dim, latent_dim)
        self.shared_query_adapter = residual_expert(latent_dim, latent_dim)

        self.planner_projection = projection(hidden_size, latent_dim)
        self.planner_encoder = set_encoder(
            latent_dim, planner_layers, set_heads
        )
        self.planner_query = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.planner_key = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.planner_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )
        self.dependency_query = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.dependency_key = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.dependency_pair = residual_expert(latent_dim * 4, latent_dim)
        self.dependency_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )
        self.parameter_scale = nn.Parameter(
            torch.tensor(math.log(6.0), dtype=torch.float32)
        )
        plan_width = latent_dim * 4
        self.admission_head = nn.Sequential(
            nn.Linear(plan_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.readiness_head = nn.Sequential(
            nn.Linear(plan_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.stage_head = nn.Sequential(
            nn.Linear(plan_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, max_waves + 1, dtype=torch.float32),
        )
        global_plan_width = latent_dim * 5
        self.branch_count_head = nn.Sequential(
            nn.Linear(global_plan_width, latent_dim * 2, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, max_branches + 1, dtype=torch.float32),
        )
        self.initial_wave_head = nn.Sequential(
            nn.Linear(global_plan_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, max_concurrency + 1, dtype=torch.float32),
        )
        self.later_wave_head = nn.Sequential(
            nn.Linear(global_plan_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, max_concurrency + 1, dtype=torch.float32),
        )
        self.wave_count_head = nn.Sequential(
            nn.Linear(global_plan_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, max_waves + 1, dtype=torch.float32),
        )
        self.plan_confidence_head = nn.Sequential(
            nn.Linear(global_plan_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )

        self.outcome_action_projection = projection(hidden_size, latent_dim)
        self.outcome_path_projection = projection(hidden_size, latent_dim)
        self.outcome_action_encoder = set_encoder(
            latent_dim, outcome_layers, set_heads
        )
        outcome_width = latent_dim * 8
        self.outcome_expert = residual_expert(outcome_width, latent_dim)
        self.coherence_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )
        self.contribution_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )
        self.failure_type_head = nn.Linear(
            latent_dim, failure_classes, dtype=torch.float32
        )
        self.recovery_head = nn.Linear(
            latent_dim, recovery_classes, dtype=torch.float32
        )
        self.continuation_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )

        self.orchestration_encoder = set_encoder(
            latent_dim, orchestration_layers, set_heads
        )
        global_outcome_width = latent_dim * 5
        self.task_complete_head = nn.Sequential(
            nn.Linear(global_outcome_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.task_solvability_head = nn.Sequential(
            nn.Linear(global_outcome_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.takeover_head = nn.Sequential(
            nn.Linear(global_outcome_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.must_report_failure_head = nn.Sequential(
            nn.Linear(global_outcome_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.recovery_embedding = nn.Embedding(
            recovery_classes, latent_dim, dtype=torch.float32
        )
        self.route_head = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 3, dtype=torch.float32),
        )

    def shared_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("shared_"):
                yield parameter

    def planner_parameters(self):
        prefixes = (
            "planner_",
            "dependency_",
            "parameter_scale",
            "admission_",
            "readiness_",
            "stage_",
            "branch_count_",
            "initial_wave_",
            "later_wave_",
            "wave_count_",
            "plan_confidence_",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(prefixes):
                yield parameter

    def outcome_parameters(self):
        prefixes = (
            "outcome_",
            "coherence_",
            "contribution_",
            "failure_type_",
            "recovery_head.",
            "continuation_",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(prefixes):
                yield parameter

    def orchestration_parameters(self):
        prefixes = (
            "orchestration_",
            "task_complete_",
            "task_solvability_",
            "takeover_",
            "must_report_failure_",
            "recovery_embedding.",
            "route_",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(prefixes):
                yield parameter

    def set_active_stage(self, stage: str) -> None:
        if stage not in {"planner", "outcome", "orchestration", "joint"}:
            raise ValueError(f"unknown training stage: {stage}")
        specialist_prefixes = {
            "planner": (
                "planner_",
                "dependency_",
                "parameter_scale",
                "admission_",
                "readiness_",
                "stage_",
                "branch_count_",
                "initial_wave_",
                "later_wave_",
                "wave_count_",
                "plan_confidence_",
            ),
            "outcome": (
                "outcome_",
                "coherence_",
                "contribution_",
                "failure_type_",
                "recovery_head.",
                "continuation_",
            ),
            "orchestration": (
                "orchestration_",
                "task_complete_",
                "task_solvability_",
                "takeover_",
                "must_report_failure_",
                "recovery_embedding.",
                "route_",
            ),
            "joint": ("",),
        }[stage]
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith("shared_")
                or name.startswith(specialist_prefixes)
            )
        self.train()
        if stage not in {"planner", "joint"}:
            self.planner_encoder.eval()
        if stage not in {"outcome", "joint"}:
            self.outcome_action_encoder.eval()
        if stage not in {"orchestration", "joint"}:
            self.orchestration_encoder.eval()

    def encode_shared_actions(self, hidden: torch.Tensor):
        semantic = self.shared_projection(hidden.float())
        contextual = semantic + self.shared_action_encoder(
            semantic.unsqueeze(0)
        ).squeeze(0)
        return contextual, semantic

    def encode_shared_query(self, hidden: torch.Tensor):
        projected = self.shared_query_projection(hidden.float())
        if projected.ndim == 2:
            projected = projected.mean(dim=0)
        return projected + self.shared_query_adapter(projected)

    def dependency_logits(self, actions: torch.Tensor):
        count = actions.shape[0]
        query = self.dependency_query(actions)
        key = self.dependency_key(actions)
        left = query[:, None].expand(count, count, -1)
        right = key[None, :].expand(count, count, -1)
        features = torch.cat(
            [left, right, left - right, left * right],
            dim=-1,
        )
        logits = self.dependency_head(
            self.dependency_pair(features)
        ).squeeze(-1)
        diagonal = torch.eye(
            count, dtype=torch.bool, device=actions.device
        )
        return logits.masked_fill(diagonal, -20.0)

    def planner_forward(
        self,
        action_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        need_hidden: torch.Tensor | None = None,
        produce_hidden: torch.Tensor | None = None,
        need_mask: torch.Tensor | None = None,
        completed_mask: torch.Tensor | None = None,
    ):
        shared_actions, shared_semantic = self.encode_shared_actions(
            action_hidden
        )
        specialist_semantic = self.planner_projection(
            action_hidden.float()
        )
        actions = shared_actions + self.planner_encoder(
            (shared_semantic + specialist_semantic).unsqueeze(0)
        ).squeeze(0)
        query = self.encode_shared_query(query_hidden)
        query = query + self.planner_projection(
            query_hidden.float()
        ).mean(dim=0)
        relevance = (
            self.planner_query(query)
            @ self.planner_key(actions).T
            / math.sqrt(self.latent_dim)
        )
        relevance = relevance + self.planner_scale.exp().clamp(max=100.0) * (
            F.normalize(query, dim=-1, eps=1e-6)
            @ F.normalize(
                shared_semantic + specialist_semantic,
                dim=-1,
                eps=1e-6,
            ).T
        )

        dependency_logits = self.dependency_logits(actions)
        if need_hidden is not None and produce_hidden is not None:
            need = self.shared_projection(need_hidden.float())
            produce = self.shared_projection(produce_hidden.float())
            parameter_flow = (
                F.normalize(need, dim=-1, eps=1e-6)
                @ F.normalize(produce, dim=-1, eps=1e-6).T
            )
            dependency_logits = dependency_logits + (
                self.parameter_scale.exp().clamp(max=100.0)
                * parameter_flow
            )
        count = actions.shape[0]
        diagonal = torch.eye(
            count, dtype=torch.bool, device=actions.device
        )
        if need_mask is not None:
            dependency_logits = dependency_logits.masked_fill(
                ~need_mask.to(actions.device).bool()[:, None],
                -20.0,
            )
        dependency_logits = dependency_logits.masked_fill(diagonal, -20.0)
        dependency_probability = dependency_logits.sigmoid().masked_fill(
            diagonal, 0.0
        )
        dependency_context = (
            dependency_probability
            / dependency_probability.sum(-1, keepdim=True).clamp_min(1e-6)
        ) @ actions
        if completed_mask is None:
            completed_mask = torch.zeros(
                count, dtype=torch.bool, device=actions.device
            )
        completed_mask = completed_mask.to(actions.device).bool()
        completed_weights = completed_mask.float()
        completed_context = (
            (completed_weights[:, None] * actions).sum(0)
            / completed_weights.sum().clamp_min(1.0)
        ).expand_as(actions)
        query_context = query.expand_as(actions)
        plan_features = torch.cat(
            [
                actions,
                dependency_context,
                completed_context + query_context,
                actions * (dependency_context + query_context),
            ],
            dim=-1,
        )
        admission_residual = self.admission_head(plan_features).squeeze(-1)
        admission = (
            relevance + admission_residual
            if self.coupled_selection
            else admission_residual
        )
        readiness = self.readiness_head(plan_features).squeeze(-1)
        unresolved = (~completed_mask).float()
        structural_ready = (
            1.0
            - dependency_probability * unresolved.unsqueeze(0)
        ).clamp(1e-6, 1.0).prod(-1)
        readiness = readiness + torch.logit(
            structural_ready.clamp(1e-5, 1.0 - 1e-5)
        )
        readiness = readiness.masked_fill(completed_mask, -20.0)
        stage_logits = self.stage_head(plan_features)

        normalized = F.normalize(actions, dim=-1, eps=1e-6)
        similarity = normalized @ normalized.T
        sibling_similarity = similarity.masked_fill(diagonal, -1.0)
        closest = sibling_similarity.max(-1).values
        novelty = (1.0 - closest).clamp(0.0, 2.0)
        if count == 1:
            novelty = torch.ones_like(novelty)
        sibling_index = sibling_similarity.argmax(-1)
        steering = F.normalize(
            actions - actions[sibling_index],
            dim=-1,
            eps=1e-6,
        )
        if count == 1:
            steering = F.normalize(actions, dim=-1, eps=1e-6)

        global_features = torch.cat(
            [
                query,
                actions.mean(0),
                actions.amin(0),
                actions.amax(0),
                (actions * admission.sigmoid().unsqueeze(-1)).mean(0),
            ],
            dim=-1,
        )
        return {
            "actions": actions,
            "action_semantic": shared_semantic + specialist_semantic,
            "shared_actions": shared_actions,
            "shared_query": query,
            "relevance": relevance,
            "admission": admission,
            "admission_residual": admission_residual,
            "dependency_logits": dependency_logits,
            "dependency_probability": dependency_probability,
            "readiness": readiness,
            "stage_logits": stage_logits,
            "branch_count_logits": self.branch_count_head(global_features),
            "initial_wave_size_logits": self.initial_wave_head(
                global_features
            ),
            "later_wave_size_logits": self.later_wave_head(global_features),
            "wave_count_logits": self.wave_count_head(global_features),
            "plan_confidence": self.plan_confidence_head(
                global_features
            ).reshape(()),
            "novelty": novelty,
            "steering": steering,
            "action_similarity": similarity,
        }

    def critic_forward(
        self,
        action_hidden: torch.Tensor,
        path_hidden: torch.Tensor,
        path_mask: torch.Tensor,
        query_hidden: torch.Tensor | None = None,
        ancestor_matrix: torch.Tensor | None = None,
    ):
        shared_actions, shared_action_semantic = self.encode_shared_actions(
            action_hidden
        )
        action_semantic = shared_action_semantic + (
            self.outcome_action_projection(action_hidden.float())
        )
        actions = shared_actions + self.outcome_action_encoder(
            action_semantic.unsqueeze(0)
        ).squeeze(0)
        if path_hidden.ndim == 3:
            mask = path_mask.bool()
            pooled_path = (
                (path_hidden.float() * mask.unsqueeze(-1)).sum(1)
                / mask.sum(1, keepdim=True).clamp_min(1)
            )
        else:
            pooled_path = path_hidden.float()
        path_semantic = (
            self.shared_projection(pooled_path)
            + self.outcome_path_projection(pooled_path)
        )
        paths = path_semantic + self.shared_path_adapter(path_semantic)
        count = actions.shape[0]
        query = (
            self.encode_shared_query(query_hidden)
            if query_hidden is not None
            else torch.zeros(
                self.latent_dim,
                dtype=actions.dtype,
                device=actions.device,
            )
        )
        assignment = (
            F.normalize(paths, dim=-1, eps=1e-6)
            @ F.normalize(actions, dim=-1, eps=1e-6).T
        )
        assigned = F.softmax(assignment * 10.0, dim=-1) @ actions
        if ancestor_matrix is None:
            ancestor_matrix = torch.zeros(
                count, count, device=actions.device
            )
        ancestor_matrix = ancestor_matrix.float().to(actions.device)
        ancestor_context = (
            ancestor_matrix
            / ancestor_matrix.sum(-1, keepdim=True).clamp_min(1.0)
        ) @ paths
        query_context = query.expand_as(paths)
        features = torch.cat(
            [
                paths,
                assigned,
                (paths - assigned).abs(),
                paths * assigned,
                ancestor_context,
                paths * ancestor_context,
                query_context,
                paths * query_context,
            ],
            dim=-1,
        )
        outcome_features = self.outcome_expert(features)
        orchestrated = outcome_features + self.orchestration_encoder(
            outcome_features.unsqueeze(0)
        ).squeeze(0)
        global_features = torch.cat(
            [
                query,
                orchestrated.mean(0),
                orchestrated.amin(0),
                orchestrated.amax(0),
                (
                    orchestrated
                    * self.coherence_head(outcome_features).sigmoid()
                ).mean(0),
            ],
            dim=-1,
        )
        failure_type = self.failure_type_head(outcome_features)
        failure_source = (
            torch.logsumexp(failure_type[:, 1:], dim=-1)
            - failure_type[:, 0]
        )
        contribution = self.contribution_head(
            outcome_features
        ).squeeze(-1)
        return {
            "actions": actions,
            "paths": paths,
            "assignment_logits": assignment * 10.0,
            "coherence": self.coherence_head(
                outcome_features
            ).squeeze(-1),
            "contribution": contribution,
            # Compatibility aliases are derived, not separately parameterized.
            "utility": contribution,
            "main_score": contribution,
            "failure_source": failure_source,
            "failure_type": failure_type,
            "recovery": self.recovery_head(outcome_features),
            "continuation": self.continuation_head(
                outcome_features
            ).squeeze(-1),
            "task_complete": self.task_complete_head(
                global_features
            ).reshape(()),
            "task_solvability": self.task_solvability_head(
                global_features
            ).reshape(()),
            "takeover": self.takeover_head(global_features).reshape(()),
            "must_report_failure": self.must_report_failure_head(
                global_features
            ).reshape(()),
            "outcome_features": outcome_features,
            "orchestration_features": orchestrated,
            "shared_actions": shared_actions,
        }

    def router_forward(
        self,
        query_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        recovery_logits: torch.Tensor | None = None,
    ):
        shared_actions, _ = self.encode_shared_actions(action_hidden)
        query = self.encode_shared_query(query_hidden)
        if recovery_logits is None:
            recovery_context = torch.zeros_like(query)
        else:
            probability = F.softmax(recovery_logits.float(), dim=-1)
            recovery_context = (
                probability @ self.recovery_embedding.weight
            ).mean(0)
        route_features = torch.cat(
            [
                query,
                shared_actions.mean(0),
                shared_actions.amin(0),
                recovery_context,
            ],
            dim=-1,
        )
        recovery_relevance = (
            F.normalize(
                query + recovery_context, dim=-1, eps=1e-6
            )
            @ F.normalize(shared_actions, dim=-1, eps=1e-6).T
        )
        return {
            "router_logits": self.route_head(route_features),
            "recovery_relevance": recovery_relevance,
            "bridge_gate": torch.sigmoid(
                recovery_context.norm() - query.norm()
            ),
        }

    def trajectory_forward(
        self,
        action_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        path_hidden: torch.Tensor,
        path_mask: torch.Tensor,
        outcome_action_hidden: torch.Tensor | None = None,
        route_action_hidden: torch.Tensor | None = None,
        need_hidden: torch.Tensor | None = None,
        produce_hidden: torch.Tensor | None = None,
        need_mask: torch.Tensor | None = None,
        completed_mask: torch.Tensor | None = None,
        ancestor_matrix: torch.Tensor | None = None,
    ):
        plan = self.planner_forward(
            action_hidden,
            query_hidden,
            need_hidden=need_hidden,
            produce_hidden=produce_hidden,
            need_mask=need_mask,
            completed_mask=completed_mask,
        )
        outcome_action_hidden = (
            action_hidden
            if outcome_action_hidden is None
            else outcome_action_hidden
        )
        outcome = self.critic_forward(
            outcome_action_hidden,
            path_hidden,
            path_mask,
            query_hidden=query_hidden,
            ancestor_matrix=ancestor_matrix,
        )
        route = self.router_forward(
            query_hidden,
            (
                outcome_action_hidden
                if route_action_hidden is None
                else route_action_hidden
            ),
            outcome["recovery"],
        )
        return {"plan": plan, "outcome": outcome, "route": route}


def parameter_counts(model: RcgMoeOpdV1) -> dict[str, int]:
    return {
        "shared": sum(
            parameter.numel() for parameter in model.shared_parameters()
        ),
        "planner": sum(
            parameter.numel() for parameter in model.planner_parameters()
        ),
        "outcome": sum(
            parameter.numel() for parameter in model.outcome_parameters()
        ),
        "orchestration": sum(
            parameter.numel()
            for parameter in model.orchestration_parameters()
        ),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }
