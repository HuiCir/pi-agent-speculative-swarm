from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


class PartialCacheLimitReached(RuntimeError):
    pass


def stable_int(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8], "big"
    )


def query_text(row: dict) -> str:
    return (
        f"source={row['source_type']}; domain={row['domain']}\n"
        f"task={row['query']}"
    )


def candidate_text(candidate: dict) -> str:
    return (
        f"action={candidate['name']}\n"
        f"description={candidate.get('description', '')}\n"
        f"parameters={json.dumps(candidate.get('parameters') or {}, ensure_ascii=False)}"
    )


def evidence_text(candidate: dict) -> str:
    evidence = str(candidate.get("evidence") or "")
    runtime = candidate.get("runtime") or {}
    if not evidence:
        return "<NO_OBSERVED_EVIDENCE>"
    return (
        f"action={candidate['name']}; observed={runtime.get('observed')}; "
        f"success={runtime.get('success')}; failure={runtime.get('failure_type')}\n"
        f"{evidence}"
    )


def all_texts(rows: list[dict]) -> dict[str, list[str]]:
    values = {"query": [], "candidate": [], "evidence": []}
    for row in rows:
        values["query"].append(query_text(row))
        for candidate in row["candidates"]:
            values["candidate"].append(candidate_text(candidate))
            values["evidence"].append(evidence_text(candidate))
    return {
        key: list(dict.fromkeys(items)) for key, items in values.items()
    }


def fingerprint(rows: list[dict], settings: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    )
    for row in rows:
        digest.update(row["task_id"].encode("utf-8"))
        digest.update(query_text(row).encode("utf-8"))
        for candidate in row["candidates"]:
            digest.update(candidate_text(candidate).encode("utf-8"))
            digest.update(evidence_text(candidate).encode("utf-8"))
    return digest.hexdigest()


def hash_encode(texts: list[str], width: int) -> torch.Tensor:
    output = []
    for text in texts:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_int(text) % (2**63 - 1))
        value = torch.randn(width, generator=generator)
        output.append(F.normalize(value, dim=0).to(torch.float16))
    return torch.stack(output)


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cache_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def partial_cache_dir(path: Path, key: str) -> Path:
    return path.with_suffix(path.suffix + f".{key[:16]}.parts")


def partial_chunk_path(
    directory: Path,
    name: str,
    start: int,
    end: int,
) -> Path:
    return directory / f"{cache_slug(name)}_{start:07d}_{end:07d}.pt"


def texts_digest(texts: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(texts, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@torch.inference_mode()
def qwen_encode(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    *,
    progress_label: str | None = None,
) -> torch.Tensor:
    output: list[torch.Tensor | None] = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    total_batches = (len(order) + batch_size - 1) // batch_size
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = [texts[index] for index in indices]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        backbone = model.model if hasattr(model, "model") else model
        hidden = backbone(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        mean = (
            (hidden * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )
        last_index = mask.long().sum(-1).clamp_min(1) - 1
        last = hidden[
            torch.arange(hidden.shape[0], device=device), last_index
        ]
        pooled = (0.75 * mean + 0.25 * last).cpu().to(torch.float16)
        for local, original in enumerate(indices):
            output[original] = pooled[local]
        del ids, mask, hidden, mean, last
        batch_index = start // batch_size + 1
        if progress_label and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % 20 == 0
        ):
            print(
                f"{progress_label}: batch {batch_index}/{total_batches}",
                flush=True,
            )
    return torch.stack([value for value in output if value is not None])


def encode_qwen_with_partial_cache(
    *,
    model,
    tokenizer,
    missing: list[str],
    positions: list[int],
    values: list[torch.Tensor | None],
    partial_dir: Path,
    name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    chunk_budget: list[int] | None = None,
) -> None:
    if not missing:
        return
    chunk_size = max(batch_size * (8 if name == "evidence" else 16), 64)
    total_chunks = (len(missing) + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, len(missing), chunk_size), 1):
        end = min(start + chunk_size, len(missing))
        chunk_texts = missing[start:end]
        expected_digest = texts_digest(chunk_texts)
        chunk_file = partial_chunk_path(partial_dir, name, start, end)
        chunk_encoded = None
        encoded_new_chunk = False
        if chunk_file.exists():
            chunk_payload = torch.load(
                chunk_file, map_location="cpu", weights_only=False
            )
            if chunk_payload.get("texts_sha256") == expected_digest:
                chunk_encoded = chunk_payload["encoded"]
                print(
                    f"loaded partial {name}: chunk {chunk_index}/{total_chunks} "
                    f"items={end - start}",
                    flush=True,
                )
            else:
                print(
                    f"discarding stale partial {name}: "
                    f"chunk {chunk_index}/{total_chunks}",
                    flush=True,
                )
        if chunk_encoded is None:
            if chunk_budget is not None and chunk_budget[0] <= 0:
                raise PartialCacheLimitReached(
                    f"partial cache chunk limit reached before {name} "
                    f"chunk {chunk_index}/{total_chunks}"
                )
            print(
                f"encoding {name}: chunk {chunk_index}/{total_chunks} "
                f"items={len(chunk_texts)}",
                flush=True,
            )
            chunk_encoded = qwen_encode(
                model,
                tokenizer,
                chunk_texts,
                max_length,
                batch_size,
                device,
                progress_label=f"{name} chunk {chunk_index}/{total_chunks}",
            )
            atomic_torch_save(
                {
                    "name": name,
                    "start": start,
                    "end": end,
                    "texts_sha256": expected_digest,
                    "encoded": chunk_encoded.cpu(),
                },
                chunk_file,
            )
            encoded_new_chunk = True
        if chunk_encoded.shape[0] != end - start:
            raise RuntimeError(
                f"partial cache shape mismatch for {name} {start}:{end}: "
                f"{tuple(chunk_encoded.shape)}"
            )
        for local, position in enumerate(positions[start:end]):
            values[position] = chunk_encoded[local]
        if encoded_new_chunk and chunk_budget is not None:
            chunk_budget[0] -= 1
            if chunk_budget[0] <= 0:
                raise PartialCacheLimitReached(
                    f"partial cache chunk limit reached after writing {name} "
                    f"chunk {chunk_index}/{total_chunks}"
                )


def build_or_load_cache(
    rows: list[dict],
    path: Path,
    *,
    mode: str,
    hidden_size: int,
    model_path: str,
    encoder_layers: int,
    device: torch.device,
    batch_size: int,
    query_tokens: int,
    candidate_tokens: int,
    evidence_tokens: int,
    base_cache_path: Path | None = None,
    max_new_chunks: int | None = None,
) -> dict:
    settings = {
        "mode": mode,
        "hidden_size": hidden_size,
        "model_path": model_path,
        "encoder_layers": encoder_layers,
        "query_tokens": query_tokens,
        "candidate_tokens": candidate_tokens,
        "evidence_tokens": evidence_tokens,
    }
    key = fingerprint(rows, settings)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") == key:
            return payload
    texts = all_texts(rows)
    base_payload = None
    if base_cache_path and base_cache_path.exists():
        candidate = torch.load(
            base_cache_path, map_location="cpu", weights_only=False
        )
        if candidate.get("settings") == settings:
            base_payload = candidate
    base_indices = {}
    if base_payload:
        base_indices = {
            name: {text: index for index, text in enumerate(items)}
            for name, items in base_payload.get("texts", {}).items()
        }

    def cached_prefix(name: str, items: list[str]) -> tuple[list[torch.Tensor | None], list[str], list[int]]:
        values: list[torch.Tensor | None] = [None] * len(items)
        missing: list[str] = []
        missing_positions: list[int] = []
        source_indices = base_indices.get(name, {})
        source_encoded = (
            base_payload.get("encoded", {}).get(name)
            if base_payload
            else None
        )
        for index, text in enumerate(items):
            cached_index = source_indices.get(text)
            if source_encoded is not None and cached_index is not None:
                values[index] = source_encoded[cached_index]
            else:
                missing.append(text)
                missing_positions.append(index)
        return values, missing, missing_positions

    cached = {
        name: cached_prefix(name, items) for name, items in texts.items()
    }
    if mode == "hash":
        encoded = {}
        for name, items in texts.items():
            values, missing, positions = cached[name]
            if missing:
                missing_encoded = hash_encode(missing, hidden_size)
                for local, position in enumerate(positions):
                    values[position] = missing_encoded[local]
            encoded[name] = torch.stack(
                [value for value in values if value is not None]
            )
    elif mode == "qwen":
        from transformers import AutoModel, AutoTokenizer

        has_missing = any(cached[name][1] for name in texts)
        part_dir = partial_cache_dir(path, key)
        tokenizer = None
        model = None
        if has_missing:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
            backbone = model.model if hasattr(model, "model") else model
            backbone.layers = torch.nn.ModuleList(
                list(backbone.layers[:encoder_layers])
            )
            model.config.num_hidden_layers = encoder_layers
            model = model.to(device)
            hidden_size = int(model.config.hidden_size)
        elif base_payload:
            hidden_size = int(base_payload["hidden_size"])
        lengths = {
            "query": query_tokens,
            "candidate": candidate_tokens,
            "evidence": evidence_tokens,
        }
        encoded = {}
        chunk_budget = [max_new_chunks] if max_new_chunks is not None else None
        try:
            for name, items in texts.items():
                values, missing, positions = cached[name]
                if missing:
                    encode_qwen_with_partial_cache(
                        model=model,
                        tokenizer=tokenizer,
                        missing=missing,
                        positions=positions,
                        values=values,
                        partial_dir=part_dir,
                        name=name,
                        max_length=lengths[name],
                        batch_size=(
                            batch_size
                            if name != "evidence"
                            else max(1, batch_size // 2)
                        ),
                        device=device,
                        chunk_budget=chunk_budget,
                    )
                encoded[name] = torch.stack(
                    [value for value in values if value is not None]
                )
                reused = len(items) - len(missing)
                print(
                    f"encoded {name}: {len(items)} reused={reused} new={len(missing)}",
                    flush=True,
                )
        finally:
            del model, tokenizer
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
    else:
        raise ValueError(f"unknown encoder mode: {mode}")
    payload = {
        "fingerprint": key,
        "hidden_size": hidden_size,
        "texts": texts,
        "encoded": encoded,
        "settings": settings,
    }
    atomic_torch_save(payload, path)
    return payload


class EncoderCache:
    def __init__(self, payload: dict):
        self.encoded = payload["encoded"]
        self.indices = {
            name: {text: index for index, text in enumerate(texts)}
            for name, texts in payload["texts"].items()
        }
        self.hidden_size = int(payload["hidden_size"])

    def get(
        self, name: str, texts: list[str], device: torch.device
    ) -> torch.Tensor:
        indices = torch.tensor(
            [self.indices[name][text] for text in texts], dtype=torch.long
        )
        return self.encoded[name].index_select(0, indices).to(
            device=device, dtype=torch.float32
        )


def runtime_features(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    node_values = []
    for candidate in row["candidates"]:
        runtime = candidate.get("runtime") or {}
        parameters = candidate.get("parameters") or {}
        properties = (
            parameters.get("properties") or {}
            if isinstance(parameters, dict)
            else {}
        )
        failure = str(runtime.get("failure_type") or "")
        node_values.append(
            [
                float(bool(runtime.get("observed"))),
                float(bool(runtime.get("success"))),
                float(bool(failure and failure != "success")),
                min(1.0, float(runtime.get("retry_count", 0)) / 4.0),
                min(1.0, float(runtime.get("latency_ratio", 0.0))),
                min(1.0, len(str(candidate.get("description") or "")) / 1000),
                min(1.0, len(str(candidate.get("evidence") or "")) / 2000),
                min(1.0, len(properties) / 12),
                float("timeout" in failure),
                float("auth" in failure),
                float("invalid" in failure),
                float("blocked" in failure),
                float("unavailable" in failure),
                float("server" in failure),
                float("empty" in failure),
                1.0,
            ]
        )
    observed = [value[0] for value in node_values]
    success = [value[1] for value in node_values]
    failure = [value[2] for value in node_values]
    metadata = row.get("metadata") or {}
    global_values = [
        min(1.0, len(node_values) / 24),
        sum(observed) / max(len(observed), 1),
        sum(success) / max(len(success), 1),
        sum(failure) / max(len(failure), 1),
        min(1.0, float(metadata.get("time_limit", 0) or 0) / 120),
        float(row["source_type"] == "trajectory"),
        float(row["source_type"] == "mmlu"),
        float(row["source_type"] == "swe_bench"),
        float(row["source_type"] == "longbench"),
        float(row["source_type"] == "clawbench"),
        float(any(observed)),
        float(all(observed)),
        float(any(failure)),
        float(any(success)),
        0.0,
        1.0,
    ]
    return torch.tensor(node_values), torch.tensor(global_values)


def encode_row(
    row: dict, cache: EncoderCache, device: torch.device
) -> dict[str, torch.Tensor]:
    candidates = row["candidates"]
    node, global_values = runtime_features(row)
    return {
        "query_hidden": cache.get("query", [query_text(row)], device),
        "candidate_hidden": cache.get(
            "candidate", [candidate_text(item) for item in candidates], device
        ),
        "evidence_hidden": cache.get(
            "evidence", [evidence_text(item) for item in candidates], device
        ),
        "node_features": node.to(device=device, dtype=torch.float32),
        "global_features": global_values.to(
            device=device, dtype=torch.float32
        ),
    }
