"""Separated planner and execution-critic towers for long-horizon RCG."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def make_set_encoder(latent_dim: int, layers: int, heads: int):
    layer = nn.TransformerEncoderLayer(
        d_model=latent_dim,
        nhead=heads,
        dim_feedforward=latent_dim * 4,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,
        dtype=torch.float32,
    )
    return nn.TransformerEncoder(
        layer, num_layers=layers, enable_nested_tensor=False
    )


def make_projection(hidden_size: int, latent_dim: int):
    return nn.Sequential(
        nn.Linear(hidden_size, latent_dim, bias=False, dtype=torch.float32),
        nn.LayerNorm(latent_dim, dtype=torch.float32),
    )


def make_expert(input_dim: int, latent_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, latent_dim * 2, dtype=torch.float32),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(latent_dim * 2, latent_dim, dtype=torch.float32),
        nn.LayerNorm(latent_dim, dtype=torch.float32),
    )


class DualTowerRCG(nn.Module):
    """Shared trunk with planner, quality, failure, and control experts."""

    def __init__(
        self,
        hidden_size: int = 4096,
        latent_dim: int = 512,
        shared_layers: int = 2,
        planner_layers: int = 3,
        critic_layers: int = 3,
        set_heads: int = 8,
        max_path_tokens: int = 160,
        failure_classes: int = 10,
        recovery_classes: int = 6,
        max_branches: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.max_path_tokens = max_path_tokens
        self.failure_classes = failure_classes
        self.recovery_classes = recovery_classes
        self.max_branches = max_branches

        self.shared_projection = make_projection(hidden_size, latent_dim)
        self.shared_set_encoder = make_set_encoder(
            latent_dim, shared_layers, set_heads
        )
        self.shared_query_norm = nn.LayerNorm(
            latent_dim, dtype=torch.float32
        )

        self.planner_projection = make_projection(hidden_size, latent_dim)
        self.planner_set_encoder = make_set_encoder(
            latent_dim, planner_layers, set_heads
        )
        self.planner_query = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.planner_key = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.planner_logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )
        self.planner_base_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.planner_dependency_query = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.planner_dependency_key = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.planner_dependency_pair = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.planner_parameter_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )
        planner_width = latent_dim * 4
        self.planner_readiness_head = nn.Sequential(
            nn.Linear(planner_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.planner_novelty_head = nn.Sequential(
            nn.Linear(planner_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.planner_steering_head = nn.Sequential(
            nn.Linear(planner_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim, dtype=torch.float32),
        )
        self.planner_admission_head = nn.Sequential(
            nn.Linear(planner_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.planner_branch_count_head = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, max_branches + 1, dtype=torch.float32),
        )

        self.critic_action_projection = make_projection(
            hidden_size, latent_dim
        )
        self.critic_action_encoder = make_set_encoder(
            latent_dim, critic_layers, set_heads
        )
        self.critic_path_projection = make_projection(
            hidden_size, latent_dim
        )
        self.critic_path_summary = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim, dtype=torch.float32),
            nn.LayerNorm(latent_dim, dtype=torch.float32),
        )
        self.critic_assignment_query = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.critic_assignment_key = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.critic_assignment_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )
        self.critic_base_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        critic_width = latent_dim * 8
        self.quality_expert = make_expert(critic_width, latent_dim)
        self.quality_coherence_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )
        self.quality_utility_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )
        self.quality_main_score_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )

        self.failure_expert = make_expert(critic_width, latent_dim)
        self.failure_source_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )
        self.failure_type_head = nn.Linear(
            latent_dim, failure_classes, dtype=torch.float32
        )
        self.failure_recovery_head = nn.Linear(
            latent_dim, recovery_classes, dtype=torch.float32
        )
        self.failure_continuation_head = nn.Linear(
            latent_dim, 1, dtype=torch.float32
        )

        self.control_expert = make_expert(critic_width, latent_dim)
        control_width = latent_dim * 3
        self.control_task_head = nn.Sequential(
            nn.Linear(control_width, latent_dim * 2, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, 1, dtype=torch.float32),
        )
        self.control_takeover_head = nn.Sequential(
            nn.Linear(control_width, latent_dim * 2, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, 1, dtype=torch.float32),
        )
        self.router_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 3, dtype=torch.float32),
        )
        self.recovery_embedding = nn.Embedding(
            recovery_classes, latent_dim, dtype=torch.float32
        )
        self.bridge_query = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim, dtype=torch.float32),
        )
        self.bridge_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )

    @staticmethod
    def _critic_head(input_dim: int, latent_dim: int, output_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, output_dim, dtype=torch.float32),
        )

    def planner_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("planner_"):
                yield parameter

    def critic_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(
                ("critic_", "quality_", "failure_", "control_")
            ):
                yield parameter

    def critic_core_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("critic_"):
                yield parameter

    def quality_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("quality_"):
                yield parameter

    def failure_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("failure_"):
                yield parameter

    def control_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("control_"):
                yield parameter

    def shared_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("shared_"):
                yield parameter

    def router_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(
                ("router_", "recovery_embedding.", "bridge_")
            ):
                yield parameter

    def set_active_group(self, group: str):
        valid = {
            "shared",
            "planner",
            "critic",
            "critic_core",
            "quality",
            "failure",
            "control",
            "control_router",
            "router",
            "joint_planner",
            "joint_critic",
            "all",
        }
        if group not in valid:
            raise ValueError(f"unknown parameter group: {group}")
        prefixes = {
            "shared": ("shared_",),
            "planner": ("planner_",),
            "critic": (
                "critic_",
                "quality_",
                "failure_",
                "control_",
            ),
            "critic_core": ("critic_",),
            "quality": ("quality_",),
            "failure": ("failure_",),
            "control": ("control_",),
            "control_router": (
                "control_",
                "router_",
                "recovery_embedding.",
                "bridge_",
            ),
            "router": (
                "router_",
                "recovery_embedding.",
                "bridge_",
            ),
            "joint_planner": ("shared_", "planner_"),
            "joint_critic": (
                "shared_",
                "critic_",
                "quality_",
                "failure_",
                "control_",
            ),
            "all": (
                "shared_",
                "planner_",
                "critic_",
                "quality_",
                "failure_",
                "control_",
                "router_",
                "recovery_embedding.",
                "bridge_",
            ),
        }[group]
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        self.train()
        if group not in {"planner", "joint_planner", "all"}:
            self.planner_set_encoder.eval()
        if group not in {"critic", "critic_core", "joint_critic", "all"}:
            self.critic_action_encoder.eval()
        if group not in {"shared", "joint_planner", "joint_critic", "all"}:
            self.shared_set_encoder.eval()

    def set_active_tower(self, tower: str):
        self.set_active_group(tower)

    def shared_encode_actions(self, action_hidden: torch.Tensor):
        semantic = self.shared_projection(action_hidden.float())
        contextual = self.shared_set_encoder(
            semantic.unsqueeze(0)
        ).squeeze(0)
        return contextual, semantic

    def shared_encode_query(self, query_hidden: torch.Tensor):
        query = self.shared_projection(query_hidden.float())
        if query.ndim == 2:
            query = query.mean(dim=0)
        return self.shared_query_norm(query)

    def planner_dependency_logits(self, actions: torch.Tensor):
        count = actions.shape[0]
        query = self.planner_dependency_query(actions)
        key = self.planner_dependency_key(actions)
        query_pair = query[:, None, :].expand(count, count, -1)
        key_pair = key[None, :, :].expand(count, count, -1)
        pair = torch.cat(
            [
                query_pair,
                key_pair,
                query_pair - key_pair,
                query_pair * key_pair,
            ],
            dim=-1,
        )
        logits = self.planner_dependency_pair(pair).squeeze(-1)
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
        shared_actions, shared_semantic = self.shared_encode_actions(
            action_hidden
        )
        expert_semantic = self.planner_projection(action_hidden.float())
        semantic = shared_semantic + expert_semantic
        actions = shared_actions + self.planner_set_encoder(
            semantic.unsqueeze(0)
        ).squeeze(0)
        shared_query = self.shared_encode_query(query_hidden)
        query = shared_query + self.planner_projection(
            query_hidden.float()
        ).mean(dim=0)
        base_context = query.expand_as(actions)
        base_gate = torch.sigmoid(
            self.planner_base_gate(
                torch.cat([actions, base_context], dim=-1)
            )
        )
        actions = actions + base_gate * base_context
        relevance = (
            self.planner_query(query)
            @ self.planner_key(actions).T
            / math.sqrt(self.latent_dim)
        )
        semantic_relevance = (
            F.normalize(query, dim=-1, eps=1e-6)
            @ F.normalize(semantic, dim=-1, eps=1e-6).T
        )
        relevance = (
            relevance
            + self.planner_logit_scale.exp().clamp(max=100.0)
            * semantic_relevance
        )

        dependency_logits = self.planner_dependency_logits(actions)
        if need_hidden is not None and produce_hidden is not None:
            need = (
                self.shared_projection(need_hidden.float())
                + self.planner_projection(need_hidden.float())
            )
            produce = (
                self.shared_projection(produce_hidden.float())
                + self.planner_projection(produce_hidden.float())
            )
            parameter_flow = (
                F.normalize(need, dim=-1, eps=1e-6)
                @ F.normalize(produce, dim=-1, eps=1e-6).T
            )
            dependency_logits = (
                dependency_logits
                + self.planner_parameter_scale.exp().clamp(max=100.0)
                * parameter_flow
            )
        count = actions.shape[0]
        diagonal = torch.eye(
            count, dtype=torch.bool, device=actions.device
        )
        dependency_logits = dependency_logits.masked_fill(diagonal, -20.0)
        if need_mask is not None:
            dependency_logits = dependency_logits.masked_fill(
                ~need_mask.to(actions.device).bool()[:, None], -20.0
            )
        dependency_probability = dependency_logits.sigmoid().masked_fill(
            diagonal, 0.0
        )
        dependency_norm = dependency_probability / dependency_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        dependency_context = dependency_norm @ actions

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
        policy_features = torch.cat(
            [
                actions,
                dependency_context,
                completed_context,
                actions * (dependency_context + completed_context),
            ],
            dim=-1,
        )
        learned_readiness = self.planner_readiness_head(
            policy_features
        ).squeeze(-1)
        unresolved = (~completed_mask).float()
        structural_readiness = torch.logit(
            (
                1.0
                - dependency_probability * unresolved.unsqueeze(0)
            )
            .clamp(1e-6, 1.0)
            .prod(-1)
            .clamp(1e-5, 1.0 - 1e-5)
        )
        readiness = (learned_readiness + structural_readiness).masked_fill(
            completed_mask, -20.0
        )

        normalized = F.normalize(actions, dim=-1, eps=1e-6)
        similarity = normalized @ normalized.T
        sibling_similarity = similarity.masked_fill(diagonal, -1.0)
        sibling_index = sibling_similarity.argmax(-1)
        sibling = actions[sibling_index]
        if count == 1:
            sibling = torch.zeros_like(actions)
        novelty_features = torch.cat(
            [
                actions,
                sibling,
                (actions - sibling).abs(),
                actions * sibling,
            ],
            dim=-1,
        )
        novelty = self.planner_novelty_head(
            novelty_features
        ).squeeze(-1)
        steering = F.normalize(
            self.planner_steering_head(novelty_features),
            dim=-1,
            eps=1e-6,
        )
        admission = self.planner_admission_head(
            policy_features
        ).squeeze(-1)
        count_features = torch.cat(
            [
                query,
                actions.mean(dim=0),
                actions.amin(dim=0),
                actions.amax(dim=0),
            ],
            dim=-1,
        )
        branch_count_logits = self.planner_branch_count_head(
            count_features
        )
        return {
            "actions": actions,
            "action_semantic": semantic,
            "shared_actions": shared_actions,
            "shared_query": shared_query,
            "base_gate": base_gate.squeeze(-1),
            "relevance": relevance,
            "dependency_logits": dependency_logits,
            "dependency_probability": dependency_probability,
            "readiness": readiness,
            "novelty": novelty,
            "steering": steering,
            "admission": admission,
            "branch_count_logits": branch_count_logits,
            "action_similarity": similarity,
        }

    def encode_critic_paths(
        self, path_hidden: torch.Tensor, path_mask: torch.Tensor
    ):
        if path_hidden.ndim == 2:
            pooled_hidden = path_hidden
        elif path_hidden.ndim == 3:
            tokens = min(path_hidden.shape[1], self.max_path_tokens)
            path_hidden = path_hidden[:, -tokens:]
            path_mask = path_mask[:, -tokens:].bool()
            pooled_hidden = (
                (path_hidden * path_mask.unsqueeze(-1)).sum(1)
                / path_mask.sum(1, keepdim=True).clamp_min(1)
            )
        else:
            raise ValueError("path_hidden must have shape [N,H] or [N,T,H]")
        projected = (
            self.shared_projection(pooled_hidden.float())
            + self.critic_path_projection(pooled_hidden.float())
        )
        contextual = projected + self.critic_path_summary(projected)
        return contextual, projected

    def critic_forward(
        self,
        action_hidden: torch.Tensor,
        path_hidden: torch.Tensor,
        path_mask: torch.Tensor,
        query_hidden: torch.Tensor | None = None,
        ancestor_matrix: torch.Tensor | None = None,
    ):
        shared_actions, shared_semantic = self.shared_encode_actions(
            action_hidden
        )
        action_semantic = (
            shared_semantic
            + self.critic_action_projection(action_hidden.float())
        )
        actions = shared_actions + self.critic_action_encoder(
            action_semantic.unsqueeze(0)
        ).squeeze(0)
        paths, path_semantic = self.encode_critic_paths(
            path_hidden, path_mask
        )
        count = actions.shape[0]
        if query_hidden is not None:
            shared_query = self.shared_encode_query(query_hidden)
            critic_query = self.critic_action_projection(
                query_hidden.float()
            ).mean(dim=0)
            base_context = (shared_query + critic_query).expand_as(paths)
            base_gate = torch.sigmoid(
                self.critic_base_gate(
                    torch.cat([paths, base_context], dim=-1)
                )
            )
            paths = paths + base_gate * base_context
        else:
            base_gate = torch.zeros(
                count, 1, device=paths.device, dtype=paths.dtype
            )
        assignment = (
            self.critic_assignment_query(paths)
            @ self.critic_assignment_key(actions).T
            / math.sqrt(self.latent_dim)
        )
        semantic_assignment = (
            F.normalize(path_semantic, dim=-1, eps=1e-6)
            @ F.normalize(action_semantic, dim=-1, eps=1e-6).T
        )
        assignment = (
            assignment
            + self.critic_assignment_scale.exp().clamp(max=100.0)
            * semantic_assignment
        )
        assigned = F.softmax(assignment, dim=-1) @ actions

        if ancestor_matrix is None:
            ancestor_matrix = torch.zeros(
                count, count, device=paths.device
            )
        ancestor_matrix = ancestor_matrix.to(paths.device).float()
        ancestor_norm = ancestor_matrix / ancestor_matrix.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        ancestor_context = ancestor_norm @ paths
        if query_hidden is None:
            query_context = torch.zeros_like(paths)
        else:
            query_context = (shared_query + critic_query).expand_as(paths)
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
        quality_features = self.quality_expert(features)
        coherence = self.quality_coherence_head(
            quality_features
        ).squeeze(-1)
        utility = self.quality_utility_head(
            quality_features
        ).squeeze(-1)
        main_score = self.quality_main_score_head(
            quality_features
        ).squeeze(-1)

        failure_features = self.failure_expert(features)
        failure_source = self.failure_source_head(
            failure_features
        ).squeeze(-1)
        failure_type = self.failure_type_head(failure_features)
        recovery = self.failure_recovery_head(failure_features)
        continuation = self.failure_continuation_head(
            failure_features
        ).squeeze(-1)

        control_features = self.control_expert(features)
        task_features = torch.cat(
            [
                control_features.mean(0),
                control_features.amin(0),
                control_features.amax(0),
            ],
            dim=-1,
        )
        task_solvability = self.control_task_head(
            task_features
        ).squeeze(-1)
        takeover = self.control_takeover_head(
            task_features
        ).squeeze(-1)
        return {
            "actions": actions,
            "paths": paths,
            "assignment_logits": assignment,
            "coherence": coherence,
            "failure_source": failure_source,
            "failure_type": failure_type,
            "recovery": recovery,
            "utility": utility,
            "main_score": main_score,
            "continuation": continuation,
            "task_solvability": task_solvability,
            "takeover": takeover,
            "shared_actions": shared_actions,
            "critic_features": features,
            "quality_features": quality_features,
            "failure_features": failure_features,
            "control_features": control_features,
            "base_gate": base_gate.squeeze(-1),
        }

    def router_forward(
        self,
        query_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        recovery_logits: torch.Tensor | None = None,
    ):
        shared_actions, _ = self.shared_encode_actions(action_hidden)
        shared_query = self.shared_encode_query(query_hidden)
        action_mean = shared_actions.mean(dim=0)
        action_min = shared_actions.amin(dim=0)
        if recovery_logits is None:
            recovery_context = torch.zeros_like(shared_query)
        else:
            recovery_probability = F.softmax(
                recovery_logits.float(), dim=-1
            )
            recovery_context = (
                recovery_probability @ self.recovery_embedding.weight
            ).mean(dim=0)
        router_features = torch.cat(
            [
                shared_query,
                action_mean,
                action_min + recovery_context,
            ],
            dim=-1,
        )
        router_logits = self.router_head(router_features)
        bridge_features = torch.cat(
            [shared_query, action_mean, recovery_context], dim=-1
        )
        bridge_query = self.bridge_query(bridge_features)
        bridge_gate = torch.sigmoid(
            self.bridge_gate(
                torch.cat([shared_query, recovery_context], dim=-1)
            )
        ).reshape(())
        recovery_relevance = (
            F.normalize(bridge_query, dim=-1, eps=1e-6)
            @ F.normalize(shared_actions, dim=-1, eps=1e-6).T
        )
        return {
            "router_logits": router_logits,
            "bridge_query": bridge_query,
            "bridge_gate": bridge_gate,
            "recovery_relevance": recovery_relevance,
        }


def parameter_counts(model: DualTowerRCG):
    return {
        "shared": sum(
            parameter.numel() for parameter in model.shared_parameters()
        ),
        "planner": sum(
            parameter.numel() for parameter in model.planner_parameters()
        ),
        "critic_core": sum(
            parameter.numel()
            for parameter in model.critic_core_parameters()
        ),
        "quality": sum(
            parameter.numel() for parameter in model.quality_parameters()
        ),
        "failure": sum(
            parameter.numel() for parameter in model.failure_parameters()
        ),
        "control": sum(
            parameter.numel() for parameter in model.control_parameters()
        ),
        "router": sum(
            parameter.numel() for parameter in model.router_parameters()
        ),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }
