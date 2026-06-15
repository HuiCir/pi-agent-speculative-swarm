"""HTTP sidecar that exposes a trained DynamicRCG controller to the swarm."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from build_failure_aware_training_data import recovery_action
from dynamic_swarm_train import (
    FAILURE_TYPES,
    encode_texts,
    load_base,
    resolve_device,
)
from rcg.dual_controller import DualTowerRCG
from rcg.dynamic_controller import DynamicRCG
from rcg.moe_opd_controller import RcgMoeOpdV1
from rcg.opd_serialization import (
    action_text as canonical_action_text,
    execution_text as canonical_execution_text,
    query_text as canonical_query_text,
)
from train_rcg_moe_v10 import (
    RECOVERY_TYPES,
    deterministic_parameter_vectors,
)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class RcgPolicyEngine:
    def __init__(self, args):
        self.args = args
        args.device = resolve_device(args.device)
        self.device = torch.device(args.device)
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        config = checkpoint.get("args", {})
        self.version = int(checkpoint.get("version", 0))
        self.is_v10 = self.version >= 10
        self.is_v11 = self.version >= 11
        self.is_v12 = self.version >= 12
        self.is_selection_coupled = bool(
            config.get("coupled_selection", False)
        )
        torch.manual_seed(int(args.random_seed))
        load_args = SimpleNamespace(
            target_path=args.target_path,
            attn_implementation=args.attn_implementation,
            device=args.device,
        )
        self.base, self.tokenizer = load_base(load_args)
        self.external_encoder = None
        encoder_layers = int(config.get("encoder_layers", 0))
        if encoder_layers:
            original_layers = len(self.base.model.layers)
            self.base.model.layers = torch.nn.ModuleList(
                list(self.base.model.layers[:encoder_layers])
            )
            self.base.config.num_hidden_layers = encoder_layers
            print(
                f"Truncated frozen encoder {original_layers}"
                f"->{encoder_layers} layers.",
                flush=True,
            )
        self._initialize_controller(checkpoint, config)

    @classmethod
    def from_external_encoder(
        cls,
        args,
        encoder,
        hidden_size,
    ):
        """Load only the RCG controller and reuse an externally owned encoder."""
        self = cls.__new__(cls)
        self.args = args
        args.device = resolve_device(args.device)
        self.device = torch.device(args.device)
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        config = checkpoint.get("args", {})
        self.base = None
        self.tokenizer = None
        self.external_encoder = encoder
        self.encoder_hidden_size = int(hidden_size)
        self._initialize_controller(checkpoint, config)
        return self

    def _initialize_controller(self, checkpoint, config):
        self.version = int(checkpoint.get("version", 0))
        self.is_v10 = self.version >= 10
        self.is_v11 = self.version >= 11
        self.is_v12 = self.version >= 12
        self.is_selection_coupled = bool(
            config.get("coupled_selection", False)
        )
        torch.manual_seed(int(self.args.random_seed))
        hidden_size = (
            self.encoder_hidden_size
            if self.external_encoder is not None
            else self.base.config.hidden_size
        )
        self.step = 0 if self.args.random_init else int(
            checkpoint.get("step", 0)
        )
        self.failure_aware = self.version >= 9
        if self.is_v12:
            self.controller = RcgMoeOpdV1(
                hidden_size=hidden_size,
                latent_dim=int(config.get("latent_dim", 768)),
                shared_layers=int(config.get("shared_layers", 6)),
                planner_layers=int(config.get("planner_layers", 2)),
                outcome_layers=int(config.get("outcome_layers", 2)),
                orchestration_layers=int(
                    config.get("orchestration_layers", 2)
                ),
                set_heads=int(config.get("set_heads", 8)),
                failure_classes=len(FAILURE_TYPES),
                recovery_classes=len(RECOVERY_TYPES),
                max_branches=int(config.get("max_branches") or 24),
                max_waves=int(config.get("max_waves") or 12),
                max_concurrency=int(config.get("max_concurrency") or 8),
                coupled_selection=self.is_selection_coupled,
            ).to(self.device)
            if not self.args.random_init:
                self.controller.load_state_dict(
                    checkpoint["model"], strict=True
                )
        elif self.is_v10:
            self.controller = DualTowerRCG(
                hidden_size=hidden_size,
                latent_dim=int(config.get("latent_dim", 512)),
                shared_layers=int(config.get("shared_layers", 2)),
                planner_layers=int(config.get("planner_layers", 2)),
                critic_layers=int(config.get("critic_layers", 2)),
                set_heads=int(config.get("set_heads", 8)),
                max_path_tokens=int(config.get("path_tokens", 128)),
                failure_classes=len(FAILURE_TYPES),
                recovery_classes=len(RECOVERY_TYPES),
                max_branches=int(config.get("max_branches") or 16),
            ).to(self.device)
            if not self.args.random_init:
                load_result = self.controller.load_state_dict(
                    checkpoint["model"], strict=self.is_v11
                )
                if not self.is_v11:
                    legacy_critic_prefixes = (
                        "critic_coherence_head.",
                        "critic_failure_source_head.",
                        "critic_failure_type_head.",
                        "critic_recovery_head.",
                        "critic_utility_head.",
                        "critic_main_score_head.",
                        "critic_continuation_head.",
                        "critic_task_head.",
                        "critic_takeover_head.",
                    )
                    unexpected = [
                        key for key in load_result.unexpected_keys
                        if not key.startswith(legacy_critic_prefixes)
                    ]
                    incompatible_missing = [
                        key for key in load_result.missing_keys
                        if not key.startswith(
                            (
                                "planner_admission_head.",
                                "planner_branch_count_head.",
                                "planner_base_gate.",
                                "critic_main_score_head.",
                                "critic_continuation_head.",
                                "critic_takeover_head.",
                                "critic_base_gate.",
                                "quality_",
                                "failure_",
                                "control_",
                            )
                        )
                    ]
                    if incompatible_missing or unexpected:
                        raise RuntimeError(
                            "V10 checkpoint is incompatible with the V11 "
                            f"controller: missing={incompatible_missing}, "
                            f"unexpected={unexpected}"
                        )
        else:
            self.controller = DynamicRCG(
                hidden_size=hidden_size,
                latent_dim=int(config.get("latent_dim", 384)),
                set_layers=int(config.get("set_layers", 2)),
                set_heads=int(config.get("set_heads", 8)),
                max_chain_tokens=int(config.get("path_tokens", 160)),
                failure_classes=len(FAILURE_TYPES),
            ).to(self.device)
            if not self.args.random_init:
                load_result = self.controller.load_state_dict(
                    checkpoint["model"], strict=False
                )
                if self.failure_aware and (
                    load_result.missing_keys
                    or load_result.unexpected_keys
                ):
                    raise RuntimeError(
                        "Failure-aware checkpoint is incompatible: "
                        f"missing={load_result.missing_keys}, "
                        f"unexpected={load_result.unexpected_keys}"
                    )
        self.controller.eval()
        self.action_tokens = int(config.get("action_tokens", 128))
        self.query_tokens = int(config.get("query_tokens", 128))
        self.path_tokens = int(config.get("path_tokens", 160))
        self.parameter_tokens = int(config.get("parameter_tokens", 32))
        self.encode_batch_size = int(config.get("encode_batch_size", 4))
        self.path_encode_batch_size = int(
            config.get(
                "path_encode_batch_size",
                min(2, self.encode_batch_size),
            )
        )
        self.variant = (
            "moe_opd_v1_random_init"
            if self.args.random_init and self.is_v12
            else "moe_opd_v1_trained"
            if self.is_v12
            else "v11_random_init"
            if self.args.random_init and self.is_v11
            else "v11_trained"
            if self.is_v11
            else "v10_random_init"
            if self.args.random_init and self.is_v10
            else "v10_trained"
            if self.is_v10
            else f"v{self.version}"
        )

    def encode_texts(self, texts, max_length, batch_size=None):
        if self.external_encoder is not None:
            return self.external_encoder(
                texts,
                max_length,
                batch_size or self.encode_batch_size,
            )
        return encode_texts(
            self.base,
            self.tokenizer,
            texts,
            max_length,
            batch_size or self.encode_batch_size,
        )

    def encode_pooled(self, texts, max_length, combine_last=False):
        encoded = self.encode_texts(
            texts,
            max_length,
            self.encode_batch_size,
        )
        pooled = encoded["mean"].float()
        if combine_last:
            pooled = 0.75 * pooled + 0.25 * encoded["last"].float()
        return pooled

    def pooled_encoded(self, encoded):
        pooled = encoded["mean"].float()
        if self.is_v12:
            pooled = 0.75 * pooled + 0.25 * encoded["last"].float()
        return pooled

    def parameter_vectors(self, texts):
        if self.is_v12:
            return self.encode_pooled(
                texts,
                self.parameter_tokens,
                combine_last=True,
            )
        if self.is_v10:
            return deterministic_parameter_vectors(
                texts,
                (
                    self.encoder_hidden_size
                    if self.external_encoder is not None
                    else self.base.config.hidden_size
                ),
            ).to(self.device)
        return self.encode_texts(
            texts,
            self.parameter_tokens,
            self.encode_batch_size,
        )["mean"]

    def calibrated_coherence(self, logits):
        temperature = max(float(self.args.coherence_temperature), 1e-3)
        return torch.sigmoid(
            logits.float() / temperature + float(self.args.coherence_bias)
        )

    @staticmethod
    def draft_has_error(draft):
        calls = draft.get("toolCalls") or []
        return str(draft.get("status")) != "ok" or any(
            bool(call.get("isError")) for call in calls
        )

    @staticmethod
    def draft_has_success(draft):
        calls = draft.get("toolCalls") or []
        return (
            str(draft.get("status")) == "ok"
            and any(
                not bool(call.get("isError"))
                and bool(str(call.get("resultText") or call.get("resultPreview") or "").strip())
                for call in calls
            )
        )

    @staticmethod
    def execution_flags(action, draft):
        expected_tool = str(
            action.get("toolName") or action.get("title") or ""
        )
        calls = draft.get("toolCalls") or []
        expected_calls = [
            call for call in calls
            if str(call.get("toolName") or "") == expected_tool
        ]
        successful_expected = [
            call for call in expected_calls
            if not bool(call.get("isError"))
            and bool(
                str(
                    call.get("resultText")
                    or call.get("resultPreview")
                    or ""
                ).strip()
            )
        ]
        successful_other = [
            call for call in calls
            if str(call.get("toolName") or "") != expected_tool
            and not bool(call.get("isError"))
        ]
        recovered = bool(successful_expected)
        terminal_error = (
            str(draft.get("status")) != "ok"
            or not recovered
            or (
                bool(expected_calls)
                and bool(expected_calls[-1].get("isError"))
            )
        )
        return {
            "success": recovered and not terminal_error,
            "terminal_error": terminal_error,
            "tool_mismatch": bool(successful_other) and not recovered,
        }

    @staticmethod
    def structural_failure_type(action, draft, flags):
        if flags["tool_mismatch"]:
            return "provenance_mismatch"
        calls = draft.get("toolCalls") or []
        text = " ".join(
            str(
                call.get("resultText")
                or call.get("resultPreview")
                or call.get("error")
                or ""
            )
            for call in calls
        ).lower()
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if any(marker in text for marker in ("401", "403", "unauthorized", "forbidden", "quota", "api key")):
            return "auth_or_quota"
        if any(marker in text for marker in ("500", "502", "503", "504", "bad gateway", "service unavailable")):
            return "server_or_network"
        if any(marker in text for marker in ("not found", "doesn't exist", "does not exist", "suspended")):
            return "api_unavailable"
        if any(
            marker in text
            for marker in (
                "400",
                "invalid parameter",
                "invalid customer",
                "missing required",
                "validation error",
                "do not match",
                "does not match",
            )
        ):
            return "invalid_call"
        if not text.strip():
            return "empty_result"
        return "other_failure"

    @staticmethod
    def acyclic_dependencies(probabilities, threshold):
        nodes = probabilities.shape[0]
        edges = []
        adjacency = [[] for _ in range(nodes)]

        def reaches(start, goal):
            stack = [start]
            seen = set()
            while stack:
                node = stack.pop()
                if node == goal:
                    return True
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(adjacency[node])
            return False

        candidates = sorted(
            (
                (float(probabilities[target, source]), target, source)
                for target in range(nodes)
                for source in range(nodes)
                if target != source
                and float(probabilities[target, source]) >= threshold
            ),
            reverse=True,
        )
        for probability, target, source in candidates:
            if reaches(target, source):
                continue
            adjacency[source].append(target)
            edges.append((target, source, probability))
        return edges

    @staticmethod
    def parameter_names(schema):
        names = set()
        if isinstance(schema, dict):
            properties = schema.get("properties")
            if isinstance(properties, dict):
                names.update(str(name) for name in properties)
            for key, value in schema.items():
                if key not in {"description", "title", "default", "examples"}:
                    names.update(RcgPolicyEngine.parameter_names(value))
        elif isinstance(schema, list):
            for item in schema:
                names.update(RcgPolicyEngine.parameter_names(item))
        return sorted(name for name in names if name)

    @classmethod
    def parameter_flow_texts(cls, actions):
        needs = [
            cls.parameter_names(action.get("parameters"))
            for action in actions
        ]
        all_names = sorted({name for items in needs for name in items})
        produces = []
        for action, own_needs in zip(actions, needs):
            description = str(
                action.get("description")
                or action.get("objective")
                or ""
            ).lower()
            produced = []
            for name in all_names:
                escaped = re.escape(name.lower())
                if re.search(
                    rf"(?:produce|return|output|provide|yield)[a-z\s_-]{{0,32}}{escaped}",
                    description,
                ):
                    produced.append(name)
            produces.append(produced)
        return (
            [" ; ".join(items) if items else "<NO_UNRESOLVED_INPUT>" for items in needs],
            [" ; ".join(items) if items else "<NO_DECLARED_OUTPUT>" for items in produces],
            [bool(items) for items in needs],
        )

    @torch.inference_mode()
    def dependency_probabilities(self, action_context, actions):
        if self.is_v10:
            return self.controller.planner_dependency_logits(
                action_context
            ).sigmoid()
        logits = self.controller.dependency_logits(action_context)
        need_texts, produce_texts, need_mask = self.parameter_flow_texts(actions)
        encoded_need = self.encode_texts(
            need_texts,
            32,
            self.encode_batch_size,
        )
        encoded_produce = self.encode_texts(
            produce_texts,
            32,
            self.encode_batch_size,
        )
        need_semantic = self.controller.action_proj(
            encoded_need["mean"].float()
        )
        produce_semantic = self.controller.action_proj(
            encoded_produce["mean"].float()
        )
        flow = (
            F.normalize(need_semantic, dim=-1, eps=1e-6)
            @ F.normalize(produce_semantic, dim=-1, eps=1e-6).T
        )
        logits = (
            logits
            + self.controller.dependency_logit_scale.exp().clamp(max=100.0)
            * flow
        )
        mask = torch.tensor(
            need_mask, dtype=torch.bool, device=self.device
        )
        logits = logits.masked_fill(~mask[:, None], -20.0)
        diagonal = torch.eye(
            len(actions), dtype=torch.bool, device=self.device
        )
        return logits.sigmoid().masked_fill(diagonal, 0.0)

    @torch.inference_mode()
    def plan(self, payload):
        started = time.perf_counter()
        tools = payload.get("tools") or []
        maximum = max(1, int(payload.get("maxBranches") or 4))
        task = str(payload.get("task") or "")
        latest_observation = str(payload.get("latestObservation") or "")
        global_state = payload.get("globalState") or {}
        runtime_state = {}
        if latest_observation and latest_observation.strip() != task.strip():
            runtime_state["latestObservation"] = latest_observation[-4000:]
        for source, target in (
            ("completedActions", "completedActions"),
            ("toolResults", "toolResults"),
            ("activeBranches", "activeBranches"),
        ):
            value = global_state.get(source) or []
            if value:
                runtime_state[target] = value
        if not tools:
            return {"actions": [], "modelStep": self.step, "latencyMs": 0.0}

        action_texts = [
            canonical_action_text(
                {
                    "tool_name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters"),
                }
            )
            for tool in tools
        ]
        encoded_actions = self.encode_pooled(
            action_texts,
            self.action_tokens,
            combine_last=self.is_v12,
        )
        encoded_query = self.encode_pooled(
            [canonical_query_text(task, runtime_state or None)],
            self.query_tokens,
            combine_last=self.is_v12,
        )
        need_texts, produce_texts, need_mask = self.parameter_flow_texts(
            tools
        )
        if self.is_v10:
            planned = self.controller.planner_forward(
                encoded_actions,
                encoded_query,
                need_hidden=self.parameter_vectors(need_texts),
                produce_hidden=self.parameter_vectors(produce_texts),
                need_mask=torch.tensor(
                    need_mask, dtype=torch.bool, device=self.device
                ),
            )
            action_context = planned["actions"]
            action_semantic = planned["action_semantic"]
            relevance = planned["relevance"]
            all_dependency = planned["dependency_probability"]
            admission_probability = (
                planned["admission"].sigmoid()
                if self.is_v11
                else None
            )
            selection_logit = (
                planned["admission"]
                if self.is_selection_coupled
                else relevance
            )
            stage_predictions = (
                planned["stage_logits"].argmax(-1)
                if self.is_v12
                else None
            )
            initial_wave_size = (
                int(planned["initial_wave_size_logits"].argmax().item())
                if self.is_v12
                else maximum
            )
            later_wave_size = (
                int(planned["later_wave_size_logits"].argmax().item())
                if self.is_v12
                else maximum
            )
            predicted_wave_count = (
                int(planned["wave_count_logits"].argmax().item())
                if self.is_v12
                else 1
            )
            plan_confidence = (
                float(planned["plan_confidence"].sigmoid().item())
                if self.is_v12
                else 1.0
            )
            if self.is_v11:
                desired_count = int(
                    planned["branch_count_logits"].argmax().item()
                )
                desired_count = max(
                    1, min(maximum, desired_count, len(tools))
                )
            else:
                relevance_float = relevance.float()
                threshold = (
                    relevance_float.mean()
                    + self.args.plan_relative_margin
                    * relevance_float.std(unbiased=False)
                )
                desired_count = int(
                    (relevance_float >= threshold).sum().item()
                )
                desired_count = max(
                    1, min(maximum, desired_count, len(tools))
                )
        else:
            stage_predictions = None
            initial_wave_size = maximum
            later_wave_size = maximum
            predicted_wave_count = 1
            plan_confidence = 1.0
            encoded_action_tokens = self.encode_texts(
                action_texts,
                self.action_tokens,
                self.encode_batch_size,
            )
            encoded_query_tokens = self.encode_texts(
                [f"<TASK_QUERY>\n{task}\n</TASK_QUERY>"],
                self.query_tokens,
                1,
            )
            action_context, action_semantic = (
                self.controller.encode_actions(
                    encoded_action_tokens["mean"]
                )
            )
            query_path, query_semantic = self.controller.encode_paths(
                encoded_query_tokens["hidden"],
                encoded_query_tokens["mask"],
            )
            learned_relevance = (
                self.controller.assignment_query(query_path)
                @ self.controller.assignment_key(action_context).T
                / math.sqrt(self.controller.latent_dim)
            )[0]
            semantic_relevance = (
                F.normalize(query_semantic, dim=-1)
                @ F.normalize(action_semantic, dim=-1).T
            )[0]
            relevance = (
                learned_relevance
                + self.controller.assignment_logit_scale.exp().clamp(
                    max=100.0
                )
                * semantic_relevance
            )
            all_dependency = None
            admission_probability = None
            selection_logit = relevance
            desired_count = min(maximum, len(tools))
        action_vectors = F.normalize(action_semantic, dim=-1)
        similarity = action_vectors @ action_vectors.T

        selected = []
        if (
            self.is_v12
            and admission_probability is not None
            and not self.is_selection_coupled
        ):
            threshold = float(
                getattr(self.args, "plan_admission_threshold", 0.5)
            )
            remaining = {
                index
                for index in range(len(tools))
                if float(admission_probability[index]) >= threshold
            }
            if not remaining:
                remaining = {
                    int(
                        (
                            relevance.float()
                            + admission_probability.float()
                        ).argmax()
                    )
                }
            desired_count = min(desired_count, len(remaining))
        else:
            remaining = set(range(len(tools)))
        while remaining and len(selected) < desired_count:
            best_index = None
            best_score = -math.inf
            for index in remaining:
                redundancy = (
                    max(float(similarity[index, other]) for other in selected)
                    if selected
                    else 0.0
                )
                score = (
                    self.args.relevance_weight
                    * float(selection_logit[index])
                    - (1.0 - self.args.relevance_weight) * redundancy
                )
                if (
                    admission_probability is not None
                    and not self.is_selection_coupled
                ):
                    score += self.args.admission_weight * float(
                        admission_probability[index]
                    )
                if score > best_score:
                    best_index = index
                    best_score = score
            selected.append(best_index)
            remaining.remove(best_index)

        selected_context = action_context[selected]
        selected_tools = [tools[index] for index in selected]
        selected_dependency = (
            all_dependency[selected][:, selected]
            if all_dependency is not None
            else self.dependency_probabilities(
                selected_context, selected_tools
            )
        )
        dependency_edges = self.acyclic_dependencies(
            selected_dependency, self.args.plan_dependency_threshold
        )
        depends_on = {index: [] for index in range(len(selected))}
        for target, source, probability in dependency_edges:
            depends_on[target].append(
                {
                    "id": f"rcg-action-{source + 1}",
                    "probability": probability,
                }
            )

        planned_actions = []
        for rank, index in enumerate(selected):
            tool = tools[index]
            planned_actions.append(
                {
                    "id": f"rcg-action-{rank + 1}",
                    "title": str(tool.get("name") or f"Action {rank + 1}"),
                    "objective": (
                        f"Use {tool.get('name')} only for the portion of the task "
                        f"that matches this capability: {tool.get('description', '')}"
                    ),
                    "whyDistinct": (
                        f"Selected by {self.variant} semantic relevance with "
                        "sibling redundancy suppression."
                    ),
                    "suggestedChecks": [
                        "Resolve the exact arguments from the user query",
                        "Return the real tool result and local completion status",
                    ],
                    "risk": (
                        "The tool candidate is semantically ranked; verify its "
                        "arguments before execution."
                    ),
                    "toolName": tool.get("name"),
                    "toolDescription": tool.get("description", ""),
                    "parameters": tool.get("parameters"),
                    "dependsOn": [
                        item["id"] for item in depends_on.get(rank, [])
                    ],
                    "dependencyScores": depends_on.get(rank, []),
                    "rcgRelevance": float(relevance[index]),
                    "admission": (
                        float(admission_probability[index])
                        if admission_probability is not None
                        else 1.0
                    ),
                    "priority": float(selection_logit[index]),
                    "ready": not bool(depends_on.get(rank)),
                    "executionWave": (
                        max(1, int(stage_predictions[index]))
                        if stage_predictions is not None
                        else 1
                    ),
                    "recommendedWaveSize": max(
                        1, min(initial_wave_size, len(selected))
                    ),
                    "laterWaveSize": max(
                        1, min(later_wave_size, len(selected))
                    ),
                    "predictedWaveCount": max(1, predicted_wave_count),
                    "planConfidence": plan_confidence,
                    "stableKey": (
                        f"{tool.get('name', '')}:"
                        f"{str(tool.get('description') or '')[:160]}"
                    ),
                }
            )
        return {
            "actions": planned_actions,
            "modelStep": self.step,
            "modelVariant": self.variant,
            "candidateCount": len(tools),
            "admittedCount": len(planned_actions),
            "recommendedWaveSize": max(
                1, min(initial_wave_size, len(planned_actions))
            ),
            "laterWaveSize": max(
                1, min(later_wave_size, len(planned_actions))
            ),
            "predictedWaveCount": max(1, predicted_wave_count),
            "planConfidence": plan_confidence,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }

    @torch.inference_mode()
    def select(self, payload):
        started = time.perf_counter()
        actions = payload.get("actions") or []
        drafts = payload.get("drafts") or []
        if not drafts:
            return {
                "scores": [],
                "modelStep": self.step,
                "latencyMs": 0.0,
            }
        action_by_id = {str(item.get("id")): item for item in actions}
        ordered_actions = [
            action_by_id.get(str(draft.get("briefId")), {}) for draft in drafts
        ]
        action_texts = [
            canonical_action_text(
                {
                    "tool_name": action.get(
                        "toolName", action.get("title", "")
                    ),
                    "description": action.get(
                        "toolDescription", action.get("objective", "")
                    ),
                    "parameters": action.get("parameters"),
                },
            )
            for action in ordered_actions
        ]
        path_texts = []
        error_mask = []
        structural_flags = []
        for draft in drafts:
            calls = draft.get("toolCalls") or []
            flags = self.execution_flags(
                ordered_actions[len(path_texts)], draft
            )
            error = flags["terminal_error"] or flags["tool_mismatch"]
            error_mask.append(error)
            structural_flags.append(flags)
            action = ordered_actions[len(path_texts)]
            path_texts.append(
                canonical_execution_text(
                    str(payload.get("task", "")),
                    {
                        "tool_name": action.get(
                            "toolName", action.get("title", "")
                        ),
                        "description": action.get(
                            "toolDescription", action.get("objective", "")
                        ),
                        "parameters": action.get("parameters"),
                        "execution": {
                            "observed": True,
                            "calls": [
                                {
                                    "toolName": call.get("toolName"),
                                    "args": call.get("args"),
                                    "status": (
                                        "error"
                                        if call.get("isError")
                                        else "success"
                                    ),
                                    "resultText": (
                                        call.get("resultText")
                                        or call.get("resultPreview")
                                        or draft.get("finalText", "")
                                    ),
                                }
                                for call in calls
                            ],
                        },
                    },
                )
            )
        encoded_action = self.encode_texts(
            action_texts,
            self.action_tokens,
            self.encode_batch_size,
        )
        encoded_path = self.encode_texts(
            path_texts,
            self.path_tokens,
            self.path_encode_batch_size,
        )
        pooled_action = self.pooled_encoded(encoded_action)
        count = len(drafts)
        need_texts, produce_texts, need_mask = self.parameter_flow_texts(
            actions
        )
        if self.is_v10:
            encoded_query = self.encode_pooled(
                [
                    f"<TASK_QUERY>\n"
                    f"{payload.get('task', '')}\n"
                    "</TASK_QUERY>"
                ],
                self.query_tokens,
                combine_last=self.is_v12,
            )
            output = self.controller.critic_forward(
                pooled_action,
                (
                    0.75 * encoded_path["mean"].float()
                    + 0.25 * encoded_path["last"].float()
                ),
                torch.ones(
                    count, 1, dtype=torch.bool, device=self.device
                ),
                query_hidden=encoded_query,
            )
            planner = self.controller.planner_forward(
                pooled_action,
                encoded_query,
                need_hidden=self.parameter_vectors(need_texts),
                produce_hidden=self.parameter_vectors(produce_texts),
                need_mask=torch.tensor(
                    need_mask, dtype=torch.bool, device=self.device
                ),
            )
        else:
            encoded_need = self.parameter_vectors(need_texts)
            encoded_produce = self.parameter_vectors(produce_texts)
            output = self.controller(
                encoded_action["mean"],
                encoded_path["hidden"],
                encoded_path["mask"],
                completed_mask=torch.zeros(
                    count, dtype=torch.bool, device=self.device
                ),
                need_hidden=encoded_need,
                produce_hidden=encoded_produce,
                need_mask=torch.tensor(
                    need_mask, dtype=torch.bool, device=self.device
                ),
            )
            planner = None
        raw_coherence = output["coherence"].sigmoid().tolist()
        coherence = self.calibrated_coherence(output["coherence"]).tolist()
        utility = output["utility"].sigmoid().tolist()
        main_score = (
            output["main_score"].sigmoid().tolist()
            if self.is_v11
            else [0.0] * count
        )
        takeover_probability = (
            float(output["takeover"].sigmoid().item())
            if self.is_v11
            else 0.0
        )
        novelty = (
            planner["novelty"].clamp(0.0, 1.0).tolist()
            if self.is_v12 and planner is not None
            else planner["novelty"].sigmoid().tolist()
            if planner is not None
            else output["novelty"].sigmoid().tolist()
        )
        failure_source = output["failure_source"].sigmoid().tolist()
        predicted_failure_type = output["failure_type"].argmax(-1).tolist()
        learned_solvability = float(
            output["task_solvability"].sigmoid().item()
        )
        structural_solvable = not any(error_mask)
        model_solvability = (
            learned_solvability
            if self.failure_aware
            else float(structural_solvable)
        )
        if self.is_v12:
            task_complete_probability = float(
                output["task_complete"].sigmoid().item()
            )
            task_complete = (
                task_complete_probability
                >= float(
                    getattr(self.args, "task_complete_threshold", 0.5)
                )
            )
            task_solvable = (
                model_solvability
                >= self.args.task_solvability_threshold
            )
            model_must_report_failure = float(
                output["must_report_failure"].sigmoid().item()
            )
        else:
            task_complete_probability = float(structural_solvable)
            task_complete = structural_solvable
            task_solvable = (
                structural_solvable
                and (
                    model_solvability
                    >= self.args.task_solvability_threshold
                    if self.failure_aware
                    else True
                )
            )
            model_must_report_failure = float(not task_solvable)
        failure_reports = []
        for index, error in enumerate(error_mask):
            if not error:
                continue
            action = ordered_actions[index]
            failure_kind = self.structural_failure_type(
                action, drafts[index], structural_flags[index]
            )
            branch_id = str(drafts[index].get("briefId"))
            blocked = [
                str(candidate.get("id"))
                for candidate in actions
                if branch_id in (candidate.get("dependsOn") or [])
            ]
            failure_reports.append(
                {
                    "briefId": branch_id,
                    "toolName": action.get("toolName", action.get("title", "")),
                    "failureType": failure_kind,
                    "recoveryAction": recovery_action(failure_kind),
                    "status": drafts[index].get("status"),
                    "blockedBriefIds": blocked,
                    "message": (
                        f"Required branch {branch_id} failed at "
                        f"{action.get('toolName', action.get('title', 'unknown tool'))}: "
                        f"{failure_kind}."
                    ),
                }
            )
        retryable = bool(failure_reports) and all(
            report["failureType"] in {"timeout", "server_or_network"}
            for report in failure_reports
        )
        task_status = (
            "complete"
            if task_complete
            else "incomplete_retryable"
            if retryable
            else "unsolvable_current_plan"
        )
        combined = [
            0.5 * coherence[index]
            + 0.25 * utility[index]
            + 0.25 * novelty[index]
            for index in range(count)
        ]
        retained = [
            (
                coherence[index] >= self.args.coherence_threshold
                and combined[index] >= self.args.combined_threshold
                and (self.is_v12 or not error_mask[index])
            )
            or (
                self.is_v12
                and error_mask[index]
                and task_complete
                and model_must_report_failure >= 0.5
                and utility[index] >= self.args.combined_threshold
            )
            for index in range(count)
        ]
        if not any(retained):
            valid = [
                index
                for index in range(count)
                if self.is_v12 or not error_mask[index]
            ]
            if valid:
                retained[max(valid, key=lambda index: combined[index])] = True
        retained_indices = [
            index for index in range(count) if retained[index]
        ]
        main_candidates = (
            [
                index
                for index in retained_indices
                if main_score[index] >= self.args.main_score_threshold
            ]
            if self.is_v11
            else retained_indices
        )
        if not main_candidates and retained_indices:
            main_candidates = [
                max(
                    retained_indices,
                    key=lambda index: (
                        main_score[index],
                        combined[index],
                    ),
                )
            ]
        main_confidence = (
            min(main_score[index] for index in main_candidates)
            if main_candidates
            else 0.0
        )
        main_decision = (
            "takeover"
            if (
                self.is_v11
                and task_complete
                and takeover_probability
                >= self.args.takeover_threshold
                and main_confidence >= self.args.main_score_threshold
            )
            else "delegate"
        )
        scores = [
            {
                "briefId": str(drafts[index].get("briefId")),
                "coherence": coherence[index],
                "rawCoherence": raw_coherence[index],
                "utility": utility[index],
                "novelty": novelty[index],
                "retain": retained[index],
                "reportEvidence": bool(
                    self.is_v12
                    and retained[index]
                    and error_mask[index]
                    and coherence[index] < self.args.coherence_threshold
                ),
                "combined": combined[index],
                "mainScore": main_score[index],
                "structuralValid": not error_mask[index],
                "toolMismatch": structural_flags[index]["tool_mismatch"],
                "terminalError": structural_flags[index]["terminal_error"],
                "failureSource": (
                    failure_source[index]
                    if self.failure_aware
                    else float(error_mask[index])
                ),
                "failureType": (
                    self.structural_failure_type(
                        ordered_actions[index],
                        drafts[index],
                        structural_flags[index],
                    )
                    if error_mask[index]
                    else (
                        FAILURE_TYPES[predicted_failure_type[index]]
                        if self.failure_aware
                        else "success"
                    )
                ),
            }
            for index in range(count)
        ]
        return {
            "scores": scores,
            "taskSolvable": task_solvable,
            "taskComplete": task_complete,
            "taskStatus": task_status,
            "taskSolvability": model_solvability,
            "taskCompletion": task_complete_probability,
            "mainDecision": main_decision,
            "mainBriefIds": (
                [
                    str(drafts[index].get("briefId"))
                    for index in main_candidates
                ]
            ),
            "mainConfidence": min(
                main_confidence, takeover_probability
            ),
            "takeoverProbability": takeover_probability,
            "mustReportFailure": (
                model_must_report_failure >= 0.5
                or not task_solvable
            ),
            "failureReports": failure_reports,
            "modelStep": self.step,
            "modelVariant": self.variant,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }

    @torch.inference_mode()
    def route(self, payload):
        started = time.perf_counter()
        actions = payload.get("actions") or []
        drafts = payload.get("drafts") or []
        if not actions or not drafts:
            return {
                "routes": [],
                "modelStep": self.step,
                "latencyMs": 0.0,
            }
        draft_by_id = {
            str(draft.get("briefId")): draft for draft in drafts
        }
        ordered_drafts = [
            draft_by_id.get(str(action.get("id")), {
                "briefId": action.get("id"),
                "status": "error",
                "finalText": "",
                "toolCalls": [],
            })
            for action in actions
        ]
        action_texts = [
            canonical_action_text(
                {
                    "tool_name": action.get(
                        "toolName", action.get("title", "")
                    ),
                    "description": action.get(
                        "toolDescription", action.get("objective", "")
                    ),
                    "parameters": action.get("parameters"),
                },
            )
            for action in actions
        ]
        path_texts = []
        completed = []
        for action, draft in zip(actions, ordered_drafts):
            calls = draft.get("toolCalls") or []
            path_texts.append(
                canonical_execution_text(
                    str(payload.get("task", "")),
                    {
                        "tool_name": action.get(
                            "toolName", action.get("title", "")
                        ),
                        "description": action.get(
                            "toolDescription", action.get("objective", "")
                        ),
                        "parameters": action.get("parameters"),
                        "execution": {
                            "observed": True,
                            "calls": [
                                {
                                    "toolName": call.get("toolName"),
                                    "args": call.get("args"),
                                    "status": (
                                        "error"
                                        if call.get("isError")
                                        else "success"
                                    ),
                                    "resultText": (
                                        call.get("resultText")
                                        or call.get("resultPreview")
                                        or draft.get("finalText", "")
                                    ),
                                }
                                for call in calls
                            ],
                        },
                    },
                )
            )
            completed.append(
                self.execution_flags(action, draft)["success"]
            )

        encoded_action = self.encode_texts(
            action_texts,
            self.action_tokens,
            self.encode_batch_size,
        )
        encoded_path = self.encode_texts(
            path_texts,
            self.path_tokens,
            self.path_encode_batch_size,
        )
        pooled_action = self.pooled_encoded(encoded_action)
        count = len(actions)
        need_texts, produce_texts, need_mask = self.parameter_flow_texts(
            actions
        )
        if self.is_v10:
            encoded_query = self.encode_pooled(
                [
                    canonical_query_text(
                        str(payload.get("task", ""))
                    )
                ],
                self.query_tokens,
                combine_last=self.is_v12,
            )
            planner = self.controller.planner_forward(
                pooled_action,
                encoded_query,
                need_hidden=self.parameter_vectors(need_texts),
                produce_hidden=self.parameter_vectors(produce_texts),
                need_mask=torch.tensor(
                    need_mask,
                    dtype=torch.bool,
                    device=self.device,
                ),
                completed_mask=torch.tensor(
                    completed, dtype=torch.bool, device=self.device
                ),
            )
            critic = self.controller.critic_forward(
                pooled_action,
                (
                    0.75 * encoded_path["mean"].float()
                    + 0.25 * encoded_path["last"].float()
                ),
                torch.ones(
                    count, 1, dtype=torch.bool, device=self.device
                ),
                query_hidden=encoded_query,
            )
            router = self.controller.router_forward(
                encoded_query,
                pooled_action,
                critic["recovery"],
            )
            dependency = planner["dependency_probability"]
            readiness = planner["readiness"].sigmoid().tolist()
            coherence = self.calibrated_coherence(
                critic["coherence"]
            ).tolist()
            continuation = (
                critic["continuation"].sigmoid().tolist()
                if self.is_v11
                else [1.0] * count
            )
            route_class = int(router["router_logits"].argmax())
        else:
            output = self.controller(
                encoded_action["mean"],
                encoded_path["hidden"],
                encoded_path["mask"],
                completed_mask=torch.tensor(
                    completed, dtype=torch.bool, device=self.device
                ),
                need_hidden=self.parameter_vectors(need_texts),
                produce_hidden=self.parameter_vectors(produce_texts),
                need_mask=torch.tensor(
                    need_mask,
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
            dependency = output["dependency_prob"]
            readiness = output["readiness"].sigmoid().tolist()
            coherence = self.calibrated_coherence(
                output["coherence"]
            ).tolist()
            continuation = [1.0] * count
            route_class = None
        reliable = [
            completed[index]
            and coherence[index] >= self.args.route_source_threshold
            for index in range(count)
        ]
        action_index = {
            str(action.get("id")): index
            for index, action in enumerate(actions)
        }
        routes = []
        for target, action in enumerate(actions):
            explicit_sources = {
                action_index[source_id]
                for source_id in action.get("dependsOn") or []
                if source_id in action_index
            }
            learned_sources = {
                source
                for source in range(count)
                if source != target
                and float(dependency[target, source])
                >= self.args.route_dependency_threshold
            }
            candidate_sources = explicit_sources | learned_sources
            sources = [
                source for source in candidate_sources if reliable[source]
            ]
            sources.sort(
                key=lambda source: float(dependency[target, source]),
                reverse=True,
            )
            flags = self.execution_flags(
                action, ordered_drafts[target]
            )
            failure_kind = self.structural_failure_type(
                action, ordered_drafts[target], flags
            )
            own_error = failure_kind == "invalid_call"
            own_error = (
                own_error
                and continuation[target]
                >= self.args.continuation_threshold
            )
            continue_own = False
            if not sources and not own_error and not continue_own:
                continue
            routes.append(
                {
                    "targetBriefId": str(action.get("id")),
                    "sourceBriefIds": [
                        str(actions[source].get("id")) for source in sources
                    ],
                    "retryOwnErrors": own_error,
                    "continueOwn": continue_own,
                    "ready": bool(readiness[target] >= 0.5),
                    "reason": (
                        f"{self.variant} dependency/readiness routing with "
                        "successful source filtering; only relevant invalid "
                        "calls may receive one bounded parameter repair."
                    ),
                    "dependencyScores": {
                        str(actions[source].get("id")): float(
                            dependency[target, source]
                        )
                        for source in sources
                    },
                    "coherence": coherence[target],
                    "continuation": continuation[target],
                    "routeClass": route_class,
                }
            )
        return {
            "routes": routes,
            "modelStep": self.step,
            "modelVariant": self.variant,
            "routeClass": route_class,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }


def build_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/health":
                    result = {
                        "ok": True,
                        "checkpoint": engine.args.checkpoint,
                        "modelStep": engine.step,
                        "modelVersion": engine.version,
                        "modelVariant": engine.variant,
                    }
                elif self.path == "/plan":
                    result = engine.plan(payload)
                elif self.path == "/select":
                    result = engine.select(payload)
                elif self.path == "/route":
                    result = engine.route(payload)
                else:
                    self.send_error(404)
                    return
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as error:
                body = json.dumps(
                    {"error": type(error).__name__, "message": str(error)}
                ).encode("utf-8")
                self.send_response(500)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, pattern, *values):
            print(
                f"[rcg-policy] {self.address_string()} "
                f"{pattern % values}",
                flush=True,
            )

    return Handler


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "rcg_moe_opd_v1_harness_adapter_best.pt"
        ),
    )
    parser.add_argument(
        "--target-path",
        default=os.environ.get(
            "PI_LOCAL_QWEN_MODEL", "models/qwen3-8b"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--relevance-weight", type=float, default=0.95)
    parser.add_argument("--admission-weight", type=float, default=1.0)
    parser.add_argument(
        "--plan-admission-threshold", type=float, default=0.5
    )
    parser.add_argument("--plan-relative-margin", type=float, default=0.25)
    parser.add_argument("--plan-dependency-threshold", type=float, default=0.65)
    parser.add_argument("--route-dependency-threshold", type=float, default=0.55)
    parser.add_argument("--route-source-threshold", type=float, default=0.35)
    parser.add_argument("--coherence-temperature", type=float, default=1.0)
    parser.add_argument("--coherence-bias", type=float, default=0.0)
    parser.add_argument("--coherence-threshold", type=float, default=0.7)
    parser.add_argument("--combined-threshold", type=float, default=0.45)
    parser.add_argument("--takeover-threshold", type=float, default=0.75)
    parser.add_argument(
        "--continuation-threshold", type=float, default=0.55
    )
    parser.add_argument("--main-score-threshold", type=float, default=0.75)
    parser.add_argument(
        "--task-solvability-threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--task-complete-threshold", type=float, default=0.5
    )
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--random-seed", type=int, default=67)
    parser.add_argument("--device", default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    engine = RcgPolicyEngine(args)
    server = HTTPServer((args.host, args.port), build_handler(engine))
    print(
        f"RCG policy ready at http://{args.host}:{args.port} "
        f"checkpoint_step={engine.step}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
