"""Replay TRAJECT-Bench golden tool calls and build live coherence labels.

The replay cache is append-only and resumable. Only tools whose registered
implementation uses a read-only GET request are executed. Transient proxy,
network, authentication, and quota failures are retained for diagnosis but are
not used as coherence supervision.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import RequestException


INFRASTRUCTURE_FAILURE_STATUSES = {
    "auth_or_quota",
    "invalid_response",
    "network_error",
    "rate_limited",
    "server_error",
    "timeout",
}
NEGATIVE_STATUSES = {
    "api_error",
    "api_missing",
    "api_not_found",
    "call_invalid",
    "live_empty",
    "provenance_mismatch",
    "service_suspended",
} | INFRASTRUCTURE_FAILURE_STATUSES | {"unresolved_metadata", "unsafe_skipped"}
POSITIVE_STATUSES = {"live_valid"}
TOKEN_PATTERN = re.compile(r"[a-z0-9_\u0080-\uffff]+", re.IGNORECASE)
METHOD_PATTERN = re.compile(r"requests\.(get|post|put|patch|delete)\s*\(", re.IGNORECASE)
AUTH_MARKERS = (
    "invalid api key",
    "invalid key",
    "not subscribed",
    "subscription",
    "unauthorized",
    "forbidden",
    "quota",
)
SUSPENDED_MARKERS = (
    "service is suspended",
    "service suspended",
    "api is suspended",
    "temporarily suspended",
)
MISSING_MARKERS = (
    "api doesn't exist",
    "api does not exist",
    "not found in available tools",
    "tool not found",
    "unknown api",
)
TRANSIENT_MARKERS = (
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "service unavailable",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout after",
)
INVALID_INPUT_MARKERS = (
    "invalid parameter",
    "invalid input",
    "missing required",
    "validation error",
    "unprocessable entity",
    "must be",
    "is required",
)
IDENTITY_KEYS = {
    "asin",
    "domain",
    "email",
    "id",
    "isbn",
    "number",
    "symbol",
    "username",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merged_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for field in ("required parameters", "required_parameters", "optional parameters", "optional_parameters"):
        for parameter in tool.get(field, []) or []:
            if isinstance(parameter, dict) and parameter.get("name") is not None:
                result[str(parameter["name"])] = parameter.get("value", parameter.get("default"))
    return result


def call_id(tool_name: str, parameters: dict[str, Any]) -> str:
    body = canonical_json({"tool_name": tool_name, "parameters": parameters})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def safe_parse(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return stripped


def is_empty(value: Any) -> bool:
    if value in (None, "", [], {}):
        return True
    if isinstance(value, dict):
        if value.get("success") is False or value.get("found") is False:
            return True
        for key in ("articles", "data", "items", "list", "result", "results", "value"):
            if key in value and value[key] in (None, "", [], {}):
                return True
    return False


def flatten_scalars(value: Any, prefix: str = "") -> dict[str, list[str]]:
    found: dict[str, list[str]] = defaultdict(list)
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower().strip()
            nested = flatten_scalars(item, name)
            for nested_key, nested_values in nested.items():
                found[nested_key].extend(nested_values)
    elif isinstance(value, list):
        for item in value[:20]:
            nested = flatten_scalars(item, prefix)
            for nested_key, nested_values in nested.items():
                found[nested_key].extend(nested_values)
    elif prefix and value is not None:
        found[prefix].append(str(value).lower().strip())
    return found


def provenance_mismatch(parameters: dict[str, Any], parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, (dict, list)):
        return []
    scalars = flatten_scalars(parsed)
    mismatches = []
    for key, expected in parameters.items():
        normalized_key = str(key).lower().strip()
        if normalized_key not in IDENTITY_KEYS or normalized_key not in scalars:
            continue
        expected_text = str(expected).lower().strip()
        observed = scalars[normalized_key]
        if expected_text and observed and expected_text not in observed:
            mismatches.append({"parameter": key, "expected": expected, "observed": observed[:5]})
    return mismatches


def text_similarity(left: Any, right: Any) -> float:
    left_tokens = set(TOKEN_PATTERN.findall(str(left).lower()))
    right_tokens = set(TOKEN_PATTERN.findall(str(right).lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def historical_status(output: Any) -> str:
    text = str(output or "").strip()
    signal = error_signal_text(text)
    if any(marker in signal for marker in SUSPENDED_MARKERS):
        return "service_suspended"
    if any(marker in signal for marker in MISSING_MARKERS):
        return "api_missing"
    if any(marker in signal for marker in INVALID_INPUT_MARKERS):
        return "call_invalid"
    if "not found" in signal or re.search(r"\b404\b", signal):
        return "api_not_found"
    if any(marker in signal for marker in TRANSIENT_MARKERS):
        return "transient_error"
    if signal.startswith("error:") or "failed after" in signal:
        return "api_error"
    if is_empty(safe_parse(text)):
        return "live_empty"
    return "apparent_success"


def error_signal_text(output: Any) -> str:
    """Return only fields that plausibly carry API status, not arbitrary content."""
    text = str(output or "").strip()
    parsed = safe_parse(output)
    if isinstance(parsed, dict):
        signal_values = []
        status_keys = {
            "code",
            "detail",
            "error",
            "errors",
            "info",
            "message",
            "messages",
            "reason",
            "status",
            "title",
            "type",
        }

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key).lower() in status_keys:
                        signal_values.append(str(item))
                    visit(item)
            elif isinstance(value, list):
                for item in value[:20]:
                    visit(item)

        visit(parsed)
        return " ".join(signal_values).lower()
    if isinstance(parsed, list):
        return ""
    if len(text) <= 1500:
        return text.lower()
    return ""


def classify_response(
    *,
    http_status: int | None,
    output: Any,
    parameters: dict[str, Any],
    transport_error: str | None = None,
) -> dict[str, Any]:
    text = str(output or "").strip()
    lowered = text.lower()
    signal = error_signal_text(output)

    if transport_error == "timeout":
        return label("timeout", 0, "high", "infrastructure")
    if transport_error:
        return label("network_error", 0, "high", "infrastructure")
    if http_status in (401, 403):
        return label("auth_or_quota", 0, "high", "infrastructure")
    if http_status == 429:
        return label("rate_limited", 0, "high", "infrastructure")
    if http_status is not None and http_status >= 500:
        return label("server_error", 0, "high", "infrastructure")
    if http_status != 200:
        if http_status == 404:
            return label("api_not_found", 0, "high", "endpoint")
        return label("invalid_response", 0, "high", "infrastructure")

    if any(marker in signal for marker in AUTH_MARKERS):
        return label("auth_or_quota", 0, "high", "infrastructure")
    if any(marker in signal for marker in SUSPENDED_MARKERS):
        return label("service_suspended", 0, "high", "endpoint")
    if any(marker in signal for marker in MISSING_MARKERS):
        return label("api_missing", 0, "high", "endpoint")
    if any(marker in signal for marker in TRANSIENT_MARKERS):
        return label("server_error", 0, "high", "infrastructure")
    if any(marker in signal for marker in INVALID_INPUT_MARKERS):
        return label("call_invalid", 0, "medium", "call")
    if "not found" in signal or re.search(r"\b404\b", signal):
        return label("api_not_found", 0, "medium", "call")
    if lowered.startswith("error:") or lowered.startswith("error "):
        return label("api_error", 0, "medium", "call")

    parsed = safe_parse(output)
    if is_empty(parsed):
        return label("live_empty", 0, "medium", "call")
    mismatches = provenance_mismatch(parameters, parsed)
    if mismatches:
        result = label("provenance_mismatch", 0, "medium", "call")
        result["provenance_mismatches"] = mismatches
        return result
    confidence = "high" if isinstance(parsed, (dict, list)) else "medium"
    return label("live_valid", 1, confidence, "call")


def label(status: str, coherence: int | None, confidence: str, failure_scope: str) -> dict[str, Any]:
    return {
        "status": status,
        "coherence_label": coherence,
        "label_confidence": confidence,
        "failure_scope": failure_scope,
        "trainable": coherence is not None,
    }


class RateLimiter:
    def __init__(self, qps: float):
        self.interval = 0.0 if qps <= 0 else 1.0 / qps
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval
        if delay:
            time.sleep(delay)


def registry_method(tool: dict[str, Any]) -> str:
    methods = {match.lower() for match in METHOD_PATTERN.findall(str(tool.get("code", "")))}
    if methods == {"get"}:
        return "GET"
    if not methods:
        return "UNKNOWN"
    return ",".join(sorted(methods)).upper()


def load_native_proxy_defaults(trajectory_root: Path) -> tuple[str | None, str | None]:
    sys.path.insert(0, str(trajectory_root))
    try:
        from utils.tool_exe import execute_trajectory
    finally:
        sys.path.pop(0)
    defaults = execute_trajectory.__defaults__ or ()
    url = defaults[0] if len(defaults) >= 1 else None
    key = defaults[1] if len(defaults) >= 2 else None
    return url, key


def collect_calls(trajectory_root: Path) -> tuple[dict[str, dict[str, Any]], list[Path], dict[str, int]]:
    calls: dict[str, dict[str, Any]] = {}
    source_paths = sorted((trajectory_root / "public_data" / "parallel").glob("**/*.json"))
    source_paths += sorted((trajectory_root / "public_data" / "sequential").glob("**/*.json"))
    stats = Counter()
    for path in source_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue
        relative = str(path.relative_to(trajectory_root / "public_data"))
        for row_index, row in enumerate(rows):
            tool_field = next((field for field in ("tool list", "tool_list") if field in row), None)
            if not tool_field:
                stats["rows_without_concrete_calls"] += 1
                continue
            stats["rows_with_concrete_calls"] += 1
            for tool_index, tool in enumerate(row.get(tool_field, []) or []):
                tool_name = str(tool.get("tool name", "")).strip()
                if not tool_name:
                    stats["calls_without_tool_name"] += 1
                    continue
                parameters = merged_parameters(tool)
                identifier = call_id(tool_name, parameters)
                entry = calls.setdefault(
                    identifier,
                    {
                        "call_id": identifier,
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "historical_outputs": [],
                        "historical_status_counts": Counter(),
                        "occurrences": [],
                    },
                )
                historical_output = tool.get("executed_output")
                entry["historical_status_counts"][historical_status(historical_output)] += 1
                if historical_output is not None and len(entry["historical_outputs"]) < 3:
                    entry["historical_outputs"].append(str(historical_output))
                entry["occurrences"].append(
                    {
                        "source": relative,
                        "row_index": row_index,
                        "tool_index": tool_index,
                        "tool_field": tool_field,
                        "query": row.get("query", ""),
                        "domain": row.get("domain") or path.parent.name,
                        "trajectory_type": row.get("trajectory_type")
                        or ("parallel" if "parallel" in path.parts else "sequential"),
                    }
                )
                stats["concrete_call_occurrences"] += 1
    stats["unique_calls"] = len(calls)
    return calls, source_paths, dict(stats)


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    tools = json.loads(path.read_text(encoding="utf-8"))
    return {str(tool.get("tool name", "")).strip(): tool for tool in tools if tool.get("tool name")}


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record["coherence_label"] = int(
                    record.get("status") == "live_valid"
                )
                record["trainable"] = True
                if record.get("status") != "live_valid":
                    record["label_confidence"] = "high"
                records[record["call_id"]] = record
            except (json.JSONDecodeError, KeyError) as error:
                raise RuntimeError(f"Corrupt replay cache at line {line_number}: {error}") from error
    return records


def make_static_record(call: dict[str, Any], status: str, method: str) -> dict[str, Any]:
    coherence = 0
    confidence = "high"
    scope = "registry" if status == "unresolved_metadata" else "safety"
    result = {
        "call_id": call["call_id"],
        "tool_name": call["tool_name"],
        "parameters": call["parameters"],
        "method": method,
        "observed_at": utc_now(),
        "attempts": 0,
        "latency_ms": 0,
        "http_status": None,
        "output": None,
        "historical_status_counts": dict(call["historical_status_counts"]),
        "historical_similarity": 0.0,
    }
    result.update(label(status, coherence, confidence, scope))
    return result


def replay_call(
    call: dict[str, Any],
    tool: dict[str, Any],
    *,
    service_url: str,
    toolbench_key: str,
    timeout: float,
    retries: int,
    limiter: RateLimiter,
    max_output_chars: int,
) -> dict[str, Any]:
    payload = {
        "category": tool.get("domain name", ""),
        "tool_name": tool.get("parent tool name", call["tool_name"]),
        "api_name": tool.get("API name", call["tool_name"]),
        "tool_input": call["parameters"],
        "strip": "truncate",
        "toolbench_key": toolbench_key,
    }
    headers = {"toolbench_key": toolbench_key}
    started = time.monotonic()
    output = None
    http_status = None
    transport_error = None
    classification = label("invalid_response", None, "low", "infrastructure")
    attempts = 0

    for attempt in range(1, retries + 1):
        attempts = attempt
        limiter.wait()
        try:
            response = requests.post(
                service_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            transport_error = None
            http_status = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and "response" in body:
                output = body["response"]
                if not isinstance(output, str):
                    output = canonical_json(output)
            else:
                output = response.text[:max_output_chars]
            classification = classify_response(
                http_status=http_status,
                output=output,
                parameters=call["parameters"],
            )
        except requests.Timeout:
            transport_error = "timeout"
            classification = classify_response(
                http_status=None,
                output=None,
                parameters=call["parameters"],
                transport_error=transport_error,
            )
        except RequestException as error:
            transport_error = type(error).__name__
            classification = classify_response(
                http_status=None,
                output=None,
                parameters=call["parameters"],
                transport_error=transport_error,
            )

        if classification["status"] not in INFRASTRUCTURE_FAILURE_STATUSES or attempt == retries:
            break
        time.sleep(min(2 ** (attempt - 1), 4))

    output_text = None if output is None else str(output)[:max_output_chars]
    historical = call["historical_outputs"][0] if call["historical_outputs"] else ""
    record = {
        "call_id": call["call_id"],
        "tool_name": call["tool_name"],
        "parameters": call["parameters"],
        "method": "GET",
        "observed_at": utc_now(),
        "attempts": attempts,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "http_status": http_status,
        "transport_error": transport_error,
        "output": output_text,
        "historical_status_counts": dict(call["historical_status_counts"]),
        "historical_similarity": round(text_similarity(historical, output_text), 6),
    }
    record.update(classification)
    return record


def compact_replay(record: dict[str, Any] | None, preview_chars: int = 2000) -> dict[str, Any]:
    if record is None:
        return {
            "status": "not_replayed",
            "coherence_label": None,
            "trainable": False,
        }
    return {
        "call_id": record["call_id"],
        "status": record["status"],
        "coherence_label": record.get("coherence_label"),
        "label_confidence": record.get("label_confidence"),
        "failure_scope": record.get("failure_scope"),
        "trainable": record.get("trainable", False),
        "observed_at": record.get("observed_at"),
        "latency_ms": record.get("latency_ms"),
        "historical_similarity": record.get("historical_similarity"),
        "live_output_preview": (record.get("output") or "")[:preview_chars],
    }


def annotate_public_data(
    trajectory_root: Path,
    output_dir: Path,
    source_paths: list[Path],
    records: dict[str, dict[str, Any]],
) -> None:
    target_root = output_dir / "annotated_public_data"
    for path in source_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            tool_field = next((field for field in ("tool list", "tool_list") if field in row), None)
            if not tool_field:
                continue
            for tool in row.get(tool_field, []) or []:
                identifier = call_id(str(tool.get("tool name", "")).strip(), merged_parameters(tool))
                tool["live_replay"] = compact_replay(records.get(identifier))
        target = target_root / path.relative_to(trajectory_root / "public_data")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(target, rows)


def annotate_rcg_training(
    rcg_training_path: Path,
    output_path: Path,
    records: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with rcg_training_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            for branch in row.get("branches", []):
                tool = {
                    "tool name": branch.get("tool_name", ""),
                    "required_parameters": branch.get("required_parameters", []),
                    "optional_parameters": branch.get("optional_parameters", []),
                }
                identifier = call_id(str(branch.get("tool_name", "")).strip(), merged_parameters(tool))
                replay = compact_replay(records.get(identifier))
                branch["live_replay"] = replay
                counts[replay["status"]] += 1
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output_path)
    return dict(counts)


def write_training_examples(
    calls: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[str, int]:
    counts = Counter()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for identifier in sorted(records):
            record = records[identifier]
            if record.get("coherence_label") is None:
                continue
            call = calls.get(identifier, {})
            occurrence = (call.get("occurrences") or [{}])[0]
            example = {
                "call_id": identifier,
                "query": occurrence.get("query", ""),
                "domain": occurrence.get("domain", ""),
                "trajectory_type": occurrence.get("trajectory_type", ""),
                "tool_name": record["tool_name"],
                "parameters": record["parameters"],
                "live_output": record.get("output"),
                "coherence_target": record["coherence_label"],
                "coherence_status": record["status"],
                "label_confidence": record.get("label_confidence"),
                "failure_scope": record.get("failure_scope"),
                "historical_status_counts": record.get("historical_status_counts", {}),
                "historical_similarity": record.get("historical_similarity", 0.0),
                "occurrence_count": len(call.get("occurrences", [])),
            }
            handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[str(record["coherence_label"])] += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return dict(counts)


def choose_pending(
    pending: list[dict[str, Any]],
    max_calls: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    pending = interleave_by_tool(pending, seed)
    if not max_calls or len(pending) <= max_calls:
        return pending
    good = []
    bad = []
    for call in pending:
        statuses = set(call["historical_status_counts"])
        (bad if statuses - {"apparent_success"} else good).append(call)
    random.Random(seed).shuffle(good)
    random.Random(seed + 1).shuffle(bad)
    selected = []
    while len(selected) < max_calls and (good or bad):
        if bad:
            selected.append(bad.pop())
        if len(selected) < max_calls and good:
            selected.append(good.pop())
    return selected


def interleave_by_tool(pending: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Round-robin tools so one slow provider cannot occupy the whole pool."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in pending:
        groups[call["tool_name"]].append(call)
    randomizer = random.Random(seed)
    tool_names = list(groups)
    randomizer.shuffle(tool_names)
    for tool_name in tool_names:
        randomizer.shuffle(groups[tool_name])
    positions = {tool_name: 0 for tool_name in tool_names}
    ordered = []
    while True:
        progressed = False
        for tool_name in tool_names:
            index = positions[tool_name]
            if index >= len(groups[tool_name]):
                continue
            ordered.append(groups[tool_name][index])
            positions[tool_name] = index + 1
            progressed = True
        if not progressed:
            return ordered


def build_summary(
    *,
    collection_stats: dict[str, int],
    calls: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    rcg_counts: dict[str, int],
    example_counts: dict[str, int],
) -> dict[str, Any]:
    status_counts = Counter(record["status"] for record in records.values())
    coherence_counts = Counter(
        "ignored" if record.get("coherence_label") is None else str(record["coherence_label"])
        for record in records.values()
    )
    domain_counts: dict[str, Counter] = defaultdict(Counter)
    for identifier, record in records.items():
        for occurrence in calls.get(identifier, {}).get("occurrences", []):
            domain_counts[occurrence.get("domain", "unknown")][record["status"]] += 1
    return {
        "generated_at": utc_now(),
        "collection": collection_stats,
        "registry_tools": len(registry),
        "replayed_unique_calls": len(records),
        "remaining_unique_calls": max(0, len(calls) - len(records)),
        "status_counts": dict(status_counts),
        "coherence_counts": dict(coherence_counts),
        "trainable_unique_calls": sum(example_counts.values()),
        "training_example_counts": example_counts,
        "rcg_branch_status_counts": rcg_counts,
        "domain_occurrence_status_counts": {
            domain: dict(counts) for domain, counts in sorted(domain_counts.items())
        },
        "label_policy": {
            "positive": sorted(POSITIVE_STATUSES),
            "negative": sorted(NEGATIVE_STATUSES),
            "infrastructure_negatives": sorted(INFRASTRUCTURE_FAILURE_STATUSES),
            "unavailable_negatives": ["unresolved_metadata", "unsafe_skipped"],
            "safety": "Only registry tools whose implementation is exclusively requests.get are executed.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rcg-training", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--qps", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-output-chars", type=int, default=20000)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--retry-transient", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_root = args.trajectory_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "replay_calls.jsonl"
    progress_path = output_dir / "progress.json"

    calls, source_paths, collection_stats = collect_calls(trajectory_root)
    registry = load_registry(trajectory_root / "public_data" / "tools" / "all_tools.json")
    records = load_cache(cache_path)

    static_records = []
    executable = []
    for identifier, call in calls.items():
        if identifier in records and not (
            args.retry_transient
            and records[identifier]["status"] in INFRASTRUCTURE_FAILURE_STATUSES
        ):
            continue
        tool = registry.get(call["tool_name"])
        if tool is None:
            static_records.append(make_static_record(call, "unresolved_metadata", "UNKNOWN"))
            continue
        method = registry_method(tool)
        if method != "GET":
            static_records.append(make_static_record(call, "unsafe_skipped", method))
            continue
        executable.append(call)

    pending = choose_pending(executable, args.max_calls, args.seed)
    api_url = os.getenv("API_URL")
    toolbench_key = os.getenv("TOOLBENCH_KEY")
    if not args.prepare_only and (not api_url or not toolbench_key):
        native_url, native_key = load_native_proxy_defaults(trajectory_root)
        api_url = api_url or native_url
        toolbench_key = toolbench_key or native_key
    if not args.prepare_only and (not api_url or not toolbench_key):
        raise RuntimeError("API_URL and TOOLBENCH_KEY are unavailable")

    mode = "a" if cache_path.exists() else "w"
    completed_this_run = 0
    run_statuses = Counter()
    limiter = RateLimiter(args.qps)
    with cache_path.open(mode, encoding="utf-8") as cache:
        for record in static_records:
            cache.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records[record["call_id"]] = record
            run_statuses[record["status"]] += 1
            completed_this_run += 1

        if not args.prepare_only and pending:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(
                        replay_call,
                        call,
                        registry[call["tool_name"]],
                        service_url=str(api_url),
                        toolbench_key=str(toolbench_key),
                        timeout=args.timeout,
                        retries=args.retries,
                        limiter=limiter,
                        max_output_chars=args.max_output_chars,
                    ): call["call_id"]
                    for call in pending
                }
                for future in as_completed(futures):
                    record = future.result()
                    cache.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    records[record["call_id"]] = record
                    run_statuses[record["status"]] += 1
                    completed_this_run += 1
                    if completed_this_run % args.flush_every == 0:
                        cache.flush()
                        os.fsync(cache.fileno())
                    if completed_this_run % args.progress_every == 0:
                        progress = {
                            "updated_at": utc_now(),
                            "completed_this_run": completed_this_run,
                            "selected_this_run": len(pending) + len(static_records),
                            "cached_total": len(records),
                            "unique_calls": len(calls),
                            "run_status_counts": dict(run_statuses),
                        }
                        atomic_json(progress_path, progress)
                        print(canonical_json(progress), flush=True)
        cache.flush()
        os.fsync(cache.fileno())

    annotate_public_data(trajectory_root, output_dir, source_paths, records)
    rcg_counts = {}
    if args.rcg_training and args.rcg_training.exists():
        rcg_counts = annotate_rcg_training(
            args.rcg_training,
            output_dir / "traject_swarm_training_live_replay.jsonl",
            records,
        )
    example_counts = write_training_examples(
        calls,
        records,
        output_dir / "coherence_live_api_examples.jsonl",
    )
    summary = build_summary(
        collection_stats=collection_stats,
        calls=calls,
        records=records,
        registry=registry,
        rcg_counts=rcg_counts,
        example_counts=example_counts,
    )
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        progress_path,
        {
            "updated_at": utc_now(),
            "state": "complete" if len(records) >= len(calls) else "partial",
            "completed_this_run": completed_this_run,
            "selected_this_run": len(pending) + len(static_records),
            "cached_total": len(records),
            "unique_calls": len(calls),
            "run_status_counts": dict(run_statuses),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
