"""Persistent local Qwen3-8B runtime shared by pi-agent and the RCG controller.

The process speaks newline-delimited JSON over stdin/stdout. Model logs go to
stderr so stdout remains a strict transport channel.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback

import mlx.core as mx
import torch
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import LRUPromptCache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from api_harness_policy import PromptHarnessPolicy
from fascia_moe.policy_engine import FasciaPolicyEngine


def log(message):
    print(f"[local-qwen-rcg] {message}", file=sys.stderr, flush=True)


def text_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
        elif item.get("type") == "image":
            parts.append("[Image omitted from local text-only runtime]")
    return "\n".join(part for part in parts if part)


def context_messages(context):
    messages = []
    system_prompt = str(context.get("systemPrompt") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in context.get("messages") or []:
        role = message.get("role")
        if role == "user":
            messages.append({
                "role": "user",
                "content": text_content(message.get("content")),
            })
        elif role == "assistant":
            content_parts = []
            tool_calls = []
            for item in message.get("content") or []:
                if item.get("type") == "text":
                    content_parts.append(str(item.get("text") or ""))
                elif item.get("type") == "thinking":
                    thinking = str(item.get("thinking") or "")
                    if thinking:
                        content_parts.append(f"<think>{thinking}</think>")
                elif item.get("type") == "toolCall":
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or ""),
                            "arguments": json.dumps(
                                item.get("arguments") or {},
                                ensure_ascii=False,
                            ),
                        },
                    })
            converted = {
                "role": "assistant",
                "content": "\n".join(
                    part for part in content_parts if part
                ),
            }
            if tool_calls:
                converted["tool_calls"] = tool_calls
            messages.append(converted)
        elif role == "toolResult":
            messages.append({
                "role": "tool",
                "name": str(message.get("toolName") or "tool"),
                "content": text_content(message.get("content")),
            })
    return messages


def context_tools(context):
    tools = []
    for tool in context.get("tools") or []:
        tools.append({
            "type": "function",
            "function": {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "parameters": tool.get("parameters") or {
                    "type": "object",
                    "properties": {},
                },
            },
        })
    return tools


class SharedQwenRuntime:
    def __init__(self, args):
        self.args = args
        log(f"loading one native model from {args.model_path}")
        self.model, self.tokenizer = load(args.model_path, lazy=True)
        self.model_key = os.path.realpath(args.model_path)
        self.prompt_cache = LRUPromptCache(
            max_size=args.cache_sequences,
            max_bytes=args.cache_bytes,
        )
        self.encoder_layers = args.encoder_layers
        self.policy = None
        self.prompt_policy = None
        if args.policy_mode == "trained":
            if not args.rcg_checkpoint:
                raise ValueError("trained policy mode requires --rcg-checkpoint")
            self.policy = FasciaPolicyEngine(
                args.rcg_checkpoint,
                self.encode_texts,
                self.model.args.hidden_size,
                args.rcg_device,
            )
            log(
                "Fascia controller loaded without a second base model: "
                f"step={self.policy.step} version={self.policy.version}"
            )
        mx.eval(self.model.parameters())
        if args.policy_mode == "prompt":
            self.prompt_policy = PromptHarnessPolicy(
                self.generate_batch,
                log=log,
            )
            log("prompt-only API harness policy enabled; no Fascia weights loaded")

    def encode_texts(self, texts, max_length, batch_size):
        if not texts:
            empty = torch.empty(
                0,
                0,
                self.model.args.hidden_size,
                dtype=torch.float32,
            )
            return {
                "hidden": empty,
                "mask": torch.empty(0, 0, dtype=torch.bool),
                "last": empty[:, 0],
                "mean": empty[:, 0],
            }
        encoded = [
            self.tokenizer.encode(str(text), add_special_tokens=False)[
                :max_length
            ]
            for text in texts
        ]
        encoded = [tokens or [self.tokenizer.eos_token_id] for tokens in encoded]
        width = max(len(tokens) for tokens in encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = next(iter(self.tokenizer.eos_token_ids))
        hidden_parts = []
        mask_parts = []
        last_parts = []
        mean_parts = []
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start:start + batch_size]
            mask_values = [
                [True] * len(tokens) + [False] * (width - len(tokens))
                for tokens in chunk
            ]
            ids = mx.array([
                tokens + [pad_id] * (width - len(tokens))
                for tokens in chunk
            ])
            hidden = self.model.model.embed_tokens(ids)
            attention_mask = create_attention_mask(hidden, None)
            for layer in self.model.model.layers[:self.encoder_layers]:
                hidden = layer(hidden, attention_mask, None)
            hidden = self.model.model.norm(hidden)
            mx.eval(hidden)
            hidden_torch = torch.from_numpy(
                __import__("numpy").array(hidden.astype(mx.float32))
            )
            mask_torch = torch.tensor(mask_values, dtype=torch.bool)
            lengths = mask_torch.long().sum(-1).clamp_min(1)
            rows = torch.arange(hidden_torch.shape[0])
            last = hidden_torch[rows, lengths - 1]
            mean = (
                (hidden_torch * mask_torch.unsqueeze(-1)).sum(1)
                / lengths.unsqueeze(-1)
            )
            hidden_parts.append(hidden_torch)
            mask_parts.append(mask_torch)
            last_parts.append(last)
            mean_parts.append(mean)
        device = self.policy.device if self.policy is not None else torch.device("cpu")
        return {
            "hidden": torch.cat(hidden_parts).to(device),
            "mask": torch.cat(mask_parts).to(device),
            "last": torch.cat(last_parts).to(device),
            "mean": torch.cat(mean_parts).to(device),
        }

    def tokenize_request(self, request):
        context = request.get("context") or {}
        options = request.get("options") or {}
        messages = context_messages(context)
        tools = context_tools(context)
        reasoning = str(options.get("reasoning") or "off")
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=reasoning != "off",
        )
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return list(tokens)

    def generate_batch(self, requests):
        started = time.perf_counter()
        prompts = [self.tokenize_request(request) for request in requests]
        caches = []
        rests = []
        prefixes = []
        cache_reads = []
        for prompt in prompts:
            cache, rest = self.prompt_cache.fetch_nearest_cache(
                self.model_key, prompt
            )
            prefix_length = len(prompt) - len(rest)
            if cache is not None and not rest:
                trim_prompt_cache(cache, 1)
                rest = prompt[-1:]
                prefix_length = len(prompt) - 1
            caches.append(cache)
            rests.append(rest)
            prefixes.append(prompt[:prefix_length])
            cache_reads.append(prefix_length)

        max_tokens = [
            max(
                1,
                min(
                    int((request.get("options") or {}).get("maxTokens") or 1024),
                    self.args.max_output_tokens,
                ),
            )
            for request in requests
        ]
        samplers = [
            make_sampler(
                temp=float(
                    (request.get("options") or {}).get("temperature")
                    if (request.get("options") or {}).get("temperature")
                    is not None
                    else self.args.temperature
                ),
                top_p=self.args.top_p,
                top_k=self.args.top_k,
            )
            for request in requests
        ]
        log(
            "generate batch "
            + json.dumps(
                {
                    "size": len(requests),
                    "sessions": [
                        str((request.get("options") or {}).get("sessionId") or "")
                        for request in requests
                    ],
                    "maxTokens": max_tokens,
                    "promptTokens": [len(prompt) for prompt in prompts],
                    "cacheReadTokens": cache_reads,
                },
                ensure_ascii=False,
            )
        )
        generator = BatchGenerator(
            self.model,
            stop_tokens=[[token] for token in self.tokenizer.eos_token_ids],
            completion_batch_size=self.args.decode_concurrency,
            prefill_batch_size=self.args.prefill_concurrency,
            prefill_step_size=self.args.prefill_step_size,
            max_kv_size=self.args.max_kv_size,
        )
        uids = generator.insert(
            rests,
            max_tokens=max_tokens,
            caches=caches,
            all_tokens=prefixes,
            samplers=samplers,
        )
        generated = {uid: [] for uid in uids}
        finished = {}
        with generator.stats() as stats:
            while responses := generator.next_generated():
                for response in responses:
                    if response.finish_reason != "stop":
                        generated[response.uid].append(response.token)
                    if response.finish_reason is not None:
                        finished[response.uid] = response
        generator.close()

        results = []
        for index, uid in enumerate(uids):
            response = finished[uid]
            self.prompt_cache.insert_cache(
                self.model_key,
                response.all_tokens[:],
                response.prompt_cache,
                cache_type="assistant",
            )
            output_tokens = generated[uid]
            results.append({
                "text": self.tokenizer.decode(output_tokens),
                "finishReason": response.finish_reason,
                "promptTokens": len(prompts[index]),
                "outputTokens": len(output_tokens),
                "cacheReadTokens": cache_reads[index],
                "cacheWriteTokens": len(prompts[index]) - cache_reads[index],
                "promptTps": stats.prompt_tps,
                "generationTps": stats.generation_tps,
                "peakMemoryGb": stats.peak_memory,
                "cacheEntries": len(self.prompt_cache),
                "cacheBytes": self.prompt_cache.nbytes,
            })
        log(
            "generate complete "
            + json.dumps(
                {
                    "seconds": round(time.perf_counter() - started, 3),
                    "outputTokens": [
                        len(generated[uid]) for uid in uids
                    ],
                    "finishReasons": [
                        finished[uid].finish_reason for uid in uids
                    ],
                    "cacheEntries": len(self.prompt_cache),
                }
            )
        )
        return results

    def policy_call(self, command, payload):
        policy = self.policy or self.prompt_policy
        if policy is None:
            raise RuntimeError("No dynamic swarm policy was configured")
        if command == "plan":
            return policy.plan(payload)
        if command == "select":
            return policy.select(payload)
        if command == "route":
            return policy.route(payload)
        raise ValueError(f"Unknown policy command: {command}")


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def reader(requests):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            requests.put(json.loads(line))
        except Exception as error:
            emit({
                "id": None,
                "ok": False,
                "error": f"Invalid JSON request: {error}",
            })
    requests.put(None)


def serve(runtime):
    requests = queue.Queue()
    threading.Thread(
        target=reader,
        args=(requests,),
        name="local-qwen-stdin",
        daemon=True,
    ).start()
    emit({
        "id": "__ready__",
        "ok": True,
        "result": {
            "modelPath": runtime.args.model_path,
            "modelType": runtime.model.model_type,
            "hiddenSize": runtime.model.args.hidden_size,
            "layers": runtime.model.args.num_hidden_layers,
            "rcgStep": runtime.policy.step if runtime.policy else None,
            "policyMode": runtime.args.policy_mode,
        },
    })
    deferred = None
    while True:
        request = deferred if deferred is not None else requests.get()
        deferred = None
        if request is None:
            return
        command = request.get("command")
        request_id = request.get("id")
        try:
            if command == "generate":
                batch = [request]
                stop_after_batch = False
                deadline = time.monotonic() + runtime.args.batch_window_ms / 1000
                while len(batch) < runtime.args.decode_concurrency:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        candidate = requests.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if candidate is None:
                        stop_after_batch = True
                        break
                    if candidate.get("command") != "generate":
                        deferred = candidate
                        break
                    batch.append(candidate)
                results = runtime.generate_batch(batch)
                for item, result in zip(batch, results):
                    emit({"id": item.get("id"), "ok": True, "result": result})
                if stop_after_batch:
                    return
            elif command in {"plan", "select", "route"}:
                result = runtime.policy_call(
                    command, request.get("payload") or {}
                )
                emit({"id": request_id, "ok": True, "result": result})
            elif command == "stats":
                emit({
                    "id": request_id,
                    "ok": True,
                    "result": {
                        "cacheEntries": len(runtime.prompt_cache),
                        "cacheBytes": runtime.prompt_cache.nbytes,
                        "modelPath": runtime.args.model_path,
                        "rcgStep": runtime.policy.step if runtime.policy else None,
                        "policyMode": runtime.args.policy_mode,
                        "promptPolicy": (
                            runtime.prompt_policy.stats
                            if runtime.prompt_policy
                            else None
                        ),
                    },
                })
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as error:
            log(traceback.format_exc())
            emit({
                "id": request_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            })


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rcg-checkpoint")
    parser.add_argument(
        "--policy-mode",
        choices=("none", "trained", "prompt"),
        default="none",
    )
    parser.add_argument("--rcg-device", default="cpu")
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--cache-sequences", type=int, default=64)
    parser.add_argument("--cache-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--max-kv-size", type=int)
    parser.add_argument("--decode-concurrency", type=int, default=8)
    parser.add_argument("--prefill-concurrency", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--batch-window-ms", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    return parser


def main():
    args = build_parser().parse_args()
    runtime = SharedQwenRuntime(args)
    serve(runtime)


if __name__ == "__main__":
    main()
