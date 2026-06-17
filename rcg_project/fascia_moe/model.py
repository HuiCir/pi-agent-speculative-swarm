from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import FasciaConfig
from .schema import EXPERT_NAMES


def _encoder(width: int, layers: int, heads: int, dropout: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=layers,
        enable_nested_tensor=False,
    )


class ResidualMLP(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class RoleExpert(nn.Module):
    def __init__(self, config: FasciaConfig):
        super().__init__()
        self.node_encoder = _encoder(
            config.latent_dim,
            config.expert_layers,
            config.num_heads,
            config.dropout,
        )
        self.node_adapter = ResidualMLP(config.latent_dim, config.dropout)
        self.global_adapter = ResidualMLP(config.latent_dim, config.dropout)
        self.node_gate = nn.Linear(config.latent_dim * 2, config.latent_dim)
        self.global_gate = nn.Linear(config.latent_dim * 2, config.latent_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contextual = self.node_encoder(nodes.unsqueeze(0)).squeeze(0)
        global_expanded = global_state.unsqueeze(0).expand(nodes.shape[0], -1)
        node_gate = torch.sigmoid(
            self.node_gate(torch.cat([contextual, global_expanded], dim=-1))
        )
        node_delta = self.node_adapter(contextual) * node_gate
        pooled = contextual.mean(dim=0)
        global_gate = torch.sigmoid(
            self.global_gate(torch.cat([global_state, pooled], dim=-1))
        )
        global_delta = self.global_adapter(global_state + pooled) * global_gate
        return node_delta, global_delta


class FasciaMoE(nn.Module):
    """Sparse recurrent event controller.

    The router executes only top-k experts. The returned recurrent state can
    be fed into the next event after tool results or subagent messages arrive.
    """

    def __init__(self, config: FasciaConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.latent_dim

        self.query_projection = nn.Sequential(
            nn.Linear(config.input_dim, width, bias=False),
            nn.LayerNorm(width),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(config.input_dim, width, bias=False),
            nn.LayerNorm(width),
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(config.input_dim, width, bias=False),
            nn.LayerNorm(width),
        )
        self.node_feature_projection = nn.Sequential(
            nn.Linear(config.feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.global_feature_projection = nn.Sequential(
            nn.Linear(config.global_feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.shared_encoder = _encoder(
            width,
            config.shared_layers,
            config.num_heads,
            config.dropout,
        )
        self.recurrent = nn.GRUCell(width * 3, width)
        self.router = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(width * 2, config.num_experts),
        )
        self.experts = nn.ModuleList(
            RoleExpert(config) for _ in range(config.num_experts)
        )
        self.post_node = ResidualMLP(width, config.dropout)
        self.post_global = ResidualMLP(width, config.dropout)

        node_width = width * 3
        self.admission_head = self._node_head(node_width, 1)
        self.priority_head = self._node_head(node_width, 1)
        self.stage_head = self._node_head(node_width, config.max_waves + 1)
        self.coherence_head = self._node_head(node_width, 1)
        self.contribution_head = self._node_head(node_width, 1)
        self.novelty_head = self._node_head(node_width, 1)
        self.memory_keep_head = self._node_head(node_width, 1)
        self.recovery_head = self._node_head(
            node_width, config.recovery_classes
        )

        pair_width = width * 5
        self.pair_adapter = nn.Sequential(
            nn.LayerNorm(pair_width),
            nn.Linear(pair_width, width * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(width * 2, width),
            nn.SiLU(),
        )
        self.dependency_head = nn.Linear(width, 1)
        self.residual_route_head = nn.Linear(width, 1)

        global_width = width * 3
        self.branch_count_head = self._global_head(
            global_width, config.max_branches + 1
        )
        self.initial_wave_head = self._global_head(
            global_width, config.max_concurrency + 1
        )
        self.later_wave_head = self._global_head(
            global_width, config.max_concurrency + 1
        )
        self.wave_count_head = self._global_head(
            global_width, config.max_waves + 1
        )
        self.task_complete_head = self._global_head(global_width, 1)
        self.task_solvable_head = self._global_head(global_width, 1)
        self.must_report_failure_head = self._global_head(global_width, 1)
        self.takeover_head = self._global_head(global_width, 1)
        self.halt_head = self._global_head(global_width, 1)
        self.remaining_turns_head = self._global_head(global_width, 1)

    @staticmethod
    def _node_head(input_width: int, output_width: int) -> nn.Module:
        hidden = input_width // 3
        return nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_width),
        )

    @staticmethod
    def _global_head(input_width: int, output_width: int) -> nn.Module:
        hidden = input_width // 3
        return nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_width),
        )

    @staticmethod
    def _stable_state(value: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
        value = F.layer_norm(value, value.shape[-1:])
        return value.clamp(-8.0, 8.0)

    def _pair_features(
        self,
        nodes: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        count = nodes.shape[0]
        left = nodes[:, None, :].expand(count, count, -1)
        right = nodes[None, :, :].expand(count, count, -1)
        global_values = global_state.view(1, 1, -1).expand(count, count, -1)
        return torch.cat(
            [left, right, left - right, left * right, global_values],
            dim=-1,
        )

    def forward(
        self,
        query_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        evidence_hidden: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
        previous_state: torch.Tensor | None = None,
        expert_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if query_hidden.ndim == 2:
            query_hidden = query_hidden.mean(dim=0)
        count = candidate_hidden.shape[0]
        device = candidate_hidden.device
        dtype = candidate_hidden.dtype
        if evidence_hidden is None:
            evidence_hidden = torch.zeros_like(candidate_hidden)
        if node_features is None:
            node_features = torch.zeros(
                count, self.config.feature_dim, device=device, dtype=dtype
            )
        if global_features is None:
            global_features = torch.zeros(
                self.config.global_feature_dim, device=device, dtype=dtype
            )

        query = self.query_projection(query_hidden)
        candidates = self.candidate_projection(candidate_hidden)
        evidence = self.evidence_projection(evidence_hidden)
        features = self.node_feature_projection(node_features)
        nodes = candidates + evidence + features + query.unsqueeze(0)
        nodes = nodes + self.shared_encoder(nodes.unsqueeze(0)).squeeze(0)
        pooled = nodes.mean(dim=0)
        runtime = self.global_feature_projection(global_features)
        if previous_state is None:
            previous_state = torch.zeros_like(query)
        else:
            previous_state = self._stable_state(previous_state)
        recurrent_input = torch.cat([query, pooled, runtime], dim=-1)
        recurrent_state = self.recurrent(recurrent_input, previous_state)
        recurrent_state = self._stable_state(recurrent_state)

        router_input = torch.cat([query, pooled, recurrent_state], dim=-1)
        router_logits = self.router(router_input)
        router_probabilities = F.softmax(router_logits, dim=-1)
        if expert_override is None:
            top_values, top_indices = torch.topk(
                router_logits, k=self.config.top_k, dim=-1
            )
        else:
            top_indices = expert_override.to(
                device=router_logits.device, dtype=torch.long
            )[: self.config.top_k]
            top_values = router_logits.index_select(0, top_indices)
        top_gates = F.softmax(top_values, dim=-1)
        mixed_nodes = torch.zeros_like(nodes)
        mixed_global = torch.zeros_like(recurrent_state)
        for gate, expert_index in zip(top_gates, top_indices):
            expert_nodes, expert_global = self.experts[int(expert_index)](
                nodes, recurrent_state
            )
            mixed_nodes = mixed_nodes + gate * expert_nodes
            mixed_global = mixed_global + gate * expert_global
        mixed_nodes = self.post_node(nodes + mixed_nodes)
        mixed_global = self.post_global(recurrent_state + mixed_global)
        mixed_global = self._stable_state(mixed_global)

        global_expanded = mixed_global.unsqueeze(0).expand(count, -1)
        query_expanded = query.unsqueeze(0).expand(count, -1)
        node_output = torch.cat(
            [mixed_nodes, global_expanded, query_expanded], dim=-1
        )
        global_output = torch.cat([mixed_global, query, pooled], dim=-1)
        pair_hidden = self.pair_adapter(
            self._pair_features(mixed_nodes, mixed_global)
        )
        diagonal = torch.eye(count, dtype=torch.bool, device=device)
        dependency = self.dependency_head(pair_hidden).squeeze(-1)
        dependency = dependency.masked_fill(diagonal, -20.0)
        residual_route = self.residual_route_head(pair_hidden).squeeze(-1)
        residual_route = residual_route.masked_fill(diagonal, -20.0)

        return {
            "admission": self.admission_head(node_output).squeeze(-1),
            "priority": self.priority_head(node_output).squeeze(-1),
            "stage_logits": self.stage_head(node_output),
            "coherence": self.coherence_head(node_output).squeeze(-1),
            "contribution": self.contribution_head(node_output).squeeze(-1),
            "novelty": self.novelty_head(node_output).squeeze(-1),
            "memory_keep": self.memory_keep_head(node_output).squeeze(-1),
            "recovery_logits": self.recovery_head(node_output),
            "dependency": dependency,
            "residual_route": residual_route,
            "branch_count_logits": self.branch_count_head(global_output),
            "initial_wave_logits": self.initial_wave_head(global_output),
            "later_wave_logits": self.later_wave_head(global_output),
            "wave_count_logits": self.wave_count_head(global_output),
            "task_complete": self.task_complete_head(global_output).squeeze(-1),
            "task_solvable": self.task_solvable_head(global_output).squeeze(-1),
            "must_report_failure": self.must_report_failure_head(
                global_output
            ).squeeze(-1),
            "takeover": self.takeover_head(global_output).squeeze(-1),
            "halt": self.halt_head(global_output).squeeze(-1),
            "remaining_turns": F.softplus(
                self.remaining_turns_head(global_output).squeeze(-1)
            ),
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "active_experts": top_indices,
            "active_gates": top_gates,
            "recurrent_state": mixed_global,
        }

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        shared = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("experts.")
        )
        expert_each = [
            sum(parameter.numel() for parameter in expert.parameters())
            for expert in self.experts
        ]
        active = shared + sum(sorted(expert_each, reverse=True)[: self.config.top_k])
        return {
            "total": total,
            "shared": shared,
            "expert_mean": round(sum(expert_each) / len(expert_each)),
            "active_upper_bound": active,
        }

    @property
    def expert_names(self) -> tuple[str, ...]:
        return EXPERT_NAMES
