"""Permutation-equivariant RCG controller for unified action-DAG scheduling."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class DynamicRCG(nn.Module):
    """Model-independent policy heads over a dynamic set of action nodes.

    The encoder hidden width is adapter-specific, while every policy head is
    independent of the number and ordering of actions.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        latent_dim: int = 384,
        set_layers: int = 2,
        set_heads: int = 8,
        max_chain_tokens: int = 160,
        failure_classes: int = 10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.max_chain_tokens = max_chain_tokens
        self.failure_classes = failure_classes

        self.action_proj = nn.Sequential(
            nn.Linear(hidden_size, latent_dim, bias=False, dtype=torch.float32),
            nn.LayerNorm(latent_dim, dtype=torch.float32),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=set_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=torch.float32,
        )
        self.action_set_encoder = nn.TransformerEncoder(
            layer, num_layers=set_layers, enable_nested_tensor=False
        )

        self.position = nn.Embedding(
            max_chain_tokens, latent_dim, dtype=torch.float32
        )
        self.path_encoder = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            dtype=torch.float32,
        )
        self.path_pool = nn.Linear(latent_dim, 1, dtype=torch.float32)

        self.assignment_query = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.assignment_key = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.assignment_logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )
        self.dependency_logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0), dtype=torch.float32)
        )

        self.dependency_query = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.dependency_key = nn.Linear(
            latent_dim, latent_dim, bias=False, dtype=torch.float32
        )
        self.dependency_pair = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )

        policy_width = latent_dim * 4
        self.readiness_head = nn.Sequential(
            nn.Linear(policy_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.coherence_head = nn.Sequential(
            nn.Linear(policy_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.novelty_head = nn.Sequential(
            nn.Linear(policy_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(policy_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.failure_source_head = nn.Sequential(
            nn.Linear(policy_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1, dtype=torch.float32),
        )
        self.failure_type_head = nn.Sequential(
            nn.Linear(policy_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, failure_classes, dtype=torch.float32),
        )
        self.task_solvability_head = nn.Sequential(
            nn.Linear(policy_width * 3, latent_dim * 2, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, 1, dtype=torch.float32),
        )
        self.steering_head = nn.Sequential(
            nn.Linear(policy_width, latent_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim, dtype=torch.float32),
        )

    def encode_actions(
        self, action_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_hidden.ndim != 2:
            raise ValueError("action_hidden must have shape [N,H]")
        semantic = self.action_proj(action_hidden.float())
        contextual = self.action_set_encoder(semantic.unsqueeze(0)).squeeze(0)
        return contextual, semantic

    def encode_paths(
        self, path_hidden: torch.Tensor, path_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if path_hidden.ndim != 3:
            raise ValueError("path_hidden must have shape [N,T,H]")
        tokens = min(path_hidden.shape[1], self.max_chain_tokens)
        path_hidden = path_hidden[:, -tokens:]
        path_mask = path_mask[:, -tokens:].bool()
        positions = torch.arange(tokens, device=path_hidden.device)
        projected = self.action_proj(path_hidden.float())
        semantic_weights = path_mask.float()
        semantic = (
            (projected * semantic_weights.unsqueeze(-1)).sum(1)
            / semantic_weights.sum(1, keepdim=True).clamp_min(1.0)
        )
        encoded = projected + self.position(positions).unsqueeze(0)
        encoded, _ = self.path_encoder(encoded)
        pool_logits = self.path_pool(encoded).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~path_mask, -1e4)
        weights = F.softmax(pool_logits, dim=-1)
        contextual = torch.einsum("nt,ntd->nd", weights, encoded)
        return contextual, semantic

    def dependency_logits(self, actions: torch.Tensor) -> torch.Tensor:
        nodes = actions.shape[0]
        query = self.dependency_query(actions)
        key = self.dependency_key(actions)
        query_pair = query[:, None, :].expand(nodes, nodes, -1)
        key_pair = key[None, :, :].expand(nodes, nodes, -1)
        features = torch.cat(
            [
                query_pair,
                key_pair,
                query_pair - key_pair,
                query_pair * key_pair,
            ],
            dim=-1,
        )
        logits = self.dependency_pair(features).squeeze(-1)
        diagonal = torch.eye(nodes, dtype=torch.bool, device=actions.device)
        return logits.masked_fill(diagonal, -20.0)

    def forward(
        self,
        action_hidden: torch.Tensor,
        path_hidden: torch.Tensor,
        path_mask: torch.Tensor,
        completed_mask: torch.Tensor | None = None,
        need_hidden: torch.Tensor | None = None,
        produce_hidden: torch.Tensor | None = None,
        need_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        actions, action_semantic = self.encode_actions(action_hidden)
        paths, path_semantic = self.encode_paths(path_hidden, path_mask)
        nodes = actions.shape[0]
        if paths.shape[0] != nodes:
            raise ValueError("action and path node counts differ")
        if completed_mask is None:
            completed_mask = torch.zeros(
                nodes, dtype=torch.bool, device=actions.device
            )
        completed_mask = completed_mask.to(actions.device).bool()

        learned_assignment = (
            self.assignment_query(paths)
            @ self.assignment_key(actions).T
            / math.sqrt(self.latent_dim)
        )
        semantic_assignment = (
            F.normalize(path_semantic, dim=-1, eps=1e-6)
            @ F.normalize(action_semantic, dim=-1, eps=1e-6).T
        )
        assignment_logits = (
            learned_assignment
            + self.assignment_logit_scale.exp().clamp(max=100.0)
            * semantic_assignment
        )
        assigned_actions = F.softmax(assignment_logits, dim=-1) @ actions

        dependency_logits = self.dependency_logits(actions)
        if need_hidden is not None and produce_hidden is not None:
            need_semantic = self.action_proj(need_hidden.float())
            produce_semantic = self.action_proj(produce_hidden.float())
            parameter_flow = (
                F.normalize(need_semantic, dim=-1, eps=1e-6)
                @ F.normalize(produce_semantic, dim=-1, eps=1e-6).T
            )
            dependency_logits = (
                dependency_logits
                + self.dependency_logit_scale.exp().clamp(max=100.0)
                * parameter_flow
            )
        diagonal = torch.eye(nodes, dtype=torch.bool, device=actions.device)
        dependency_logits = dependency_logits.masked_fill(diagonal, -20.0)
        if need_mask is not None:
            dependency_logits = dependency_logits.masked_fill(
                ~need_mask.to(actions.device).bool()[:, None], -20.0
            )
        dependency_prob = dependency_logits.sigmoid().masked_fill(diagonal, 0.0)
        dependency_norm = dependency_prob / dependency_prob.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        dependency_context = dependency_norm @ actions

        completed_weights = completed_mask.float()
        completed_context = (
            (completed_weights[:, None] * actions).sum(0)
            / completed_weights.sum().clamp_min(1.0)
        ).expand_as(actions)
        readiness_features = torch.cat(
            [
                actions,
                dependency_context,
                completed_context,
                actions * (dependency_context + completed_context),
            ],
            dim=-1,
        )
        learned_readiness = self.readiness_head(readiness_features).squeeze(-1)
        unresolved = (~completed_mask).float()
        unblocked_probability = (
            1.0 - dependency_prob * unresolved.unsqueeze(0)
        ).clamp(1e-6, 1.0).prod(-1)
        structural_readiness = torch.logit(
            unblocked_probability.clamp(1e-5, 1.0 - 1e-5)
        )
        readiness = (learned_readiness + structural_readiness).masked_fill(
            completed_mask, -20.0
        )

        result_features = torch.cat(
            [
                paths,
                assigned_actions,
                (paths - assigned_actions).abs(),
                paths * assigned_actions,
            ],
            dim=-1,
        )
        coherence = self.coherence_head(result_features).squeeze(-1)
        utility = self.utility_head(result_features).squeeze(-1)
        failure_source = self.failure_source_head(result_features).squeeze(-1)
        failure_type = self.failure_type_head(result_features)
        task_features = torch.cat(
            [
                result_features.mean(dim=0),
                result_features.amin(dim=0),
                result_features.amax(dim=0),
            ],
            dim=-1,
        )
        task_solvability = self.task_solvability_head(
            task_features
        ).squeeze(-1)

        normalized_paths = F.normalize(paths, dim=-1, eps=1e-6)
        similarity = normalized_paths @ normalized_paths.T
        sibling_similarity = similarity.masked_fill(diagonal, -1.0)
        sibling_index = sibling_similarity.argmax(dim=-1)
        sibling_context = paths[sibling_index]
        novelty_features = torch.cat(
            [
                paths,
                sibling_context,
                (paths - sibling_context).abs(),
                paths * sibling_context,
            ],
            dim=-1,
        )
        novelty = self.novelty_head(novelty_features).squeeze(-1)
        steering = F.normalize(
            self.steering_head(novelty_features), dim=-1, eps=1e-6
        )

        sibling_weights = F.softmax(
            dependency_logits.masked_fill(diagonal, -1e4), dim=-1
        )
        if nodes == 1:
            sibling_weights = torch.zeros_like(sibling_weights)

        return {
            "actions": actions,
            "action_semantic": action_semantic,
            "paths": paths,
            "path_semantic": path_semantic,
            "assignment_logits": assignment_logits,
            "dependency_logits": dependency_logits,
            "dependency_prob": dependency_prob,
            "readiness": readiness,
            "coherence": coherence,
            "novelty": novelty,
            "utility": utility,
            "failure_source": failure_source,
            "failure_type": failure_type,
            "task_solvability": task_solvability,
            "steering": steering,
            "sibling_weights": sibling_weights,
            "path_similarity": similarity,
        }


def dynamic_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
