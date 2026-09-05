#!/usr/bin/env python3
"""Compare Qwen3.5 and Qwen3.8 on progressively larger review contexts."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

from test_live_qwen_reasoning import (
    SEGMENTS,
    _extract_result,
    _mode_payload,
    _request_json,
)


DEFAULT_LOCAL_MODEL = "RishabhSinha/Qwen3.5-9B-NVFP4"
DEFAULT_REMOTE_MODEL = "Inferact/Qwen3.8-27B-NVFP4"
DEFAULT_LOCAL_URL = "http://127.0.0.1:8001"
DEFAULT_REMOTE_URL = "http://192.168.1.230:8000"
MAX_OUTPUT_TOKENS = 1400
TARGETS = (4_000, 8_000, 16_000, 24_000)

FILLER_TEXT = (
    "The guest described a practical example, and the host asked a follow-up question "
    "about how the implementation would work in everyday use."
)

NEEDLES = [
    {"id": 9101, "speaker": "HOST", "text": "The results was encouraging, and we recieve feedback every week."},
    {"id": 9102, "speaker": "GUEST", "text": "Its a simple change, but it effects the outcome."},
    {"id": 9103, "speaker": "HOST", "text": "We need to seperate the files before the final review."},
    {"id": 9104, "speaker": "GUEST", "text": "That approach is definately worth testing."},
    {"id": 9105, "speaker": "HOST", "text": "The team have already completed the first pass."},
    {"id": 9106, "speaker": "GUEST", "text": "Please send the report when you are ready."},
    {"id": 9107, "speaker": "HOST", "text": "We should preserve the original meaning and speaker labels."},
    {"id": 9108, "speaker": "GUEST", "text": "This is the final example for the comparison."},
]


def _review_payload(model: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _mode_payload("none", model)
    payload["messages"] = copy.deepcopy(payload["messages"])
    payload["messages"][1]["content"] = json.dumps(
        {
            "stage_name": "transcript_cleanup_review",
            "stage_mode": "local_batch",
            "edit_scope": "text_only",
            "return_only_changed_segments": True,
            "changed_segments_only": True,
            "max_changed_segments_hint": 8,
            "segments": segments,
        },
        ensure_ascii=True,
    )
    return payload


def _segments_for_count(filler_count: int) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    needle_positions = {
        round((filler_count - 1) * index / max(len(NEEDLES) - 1, 1))
        for index in range(len(NEEDLES))
    }
    needle_index = 0
    for filler_index in range(filler_count):
        segments.append(
            {
                "id": 10_000 + filler_index,
                "speaker": "GUEST" if filler_index % 2 else "HOST",
                "text": FILLER_TEXT,
            }
        )
        if filler_index in needle_positions and needle_index < len(NEEDLES):
            segments.append(NEEDLES[needle_index])
            needle_index += 1
    while needle_index < len(NEEDLES):
        segments.append(NEEDLES[needle_index])
        needle_index += 1
    return segments


def _token_count(tokenizer: Any, model: str, segments: List[Dict[str, Any]]) -> int:
    payload = _review_payload(model, segments)
    messages = payload["messages"]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    if hasattr(ids, "__getitem__") and not isinstance(ids, (list, tuple)):
        try:
            ids = ids["input_ids"]
        except (KeyError, TypeError, IndexError):
            pass
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def _make_context(tokenizer: Any, model: str, target: int) -> tuple[List[Dict[str, Any]], int]:
    low, high = 1, max(4, target // 8)
    while _token_count(tokenizer, model, _segments_for_count(high)) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if _token_count(tokenizer, model, _segments_for_count(middle)) < target:
            low = middle + 1
        else:
            high = middle
    candidates = [_segments_for_count(max(1, low - 1)), _segments_for_count(low)]
    segments = min(candidates, key=lambda item: abs(_token_count(tokenizer, model, item) - target))
    return segments, _token_count(tokenizer, model, segments)


def _signature(response: Dict[str, Any]) -> Dict[str, Any]:
    extracted = _extract_result(response)
    content = extracted.get("content") or ""
    parsed: Any = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        pass
    normalized = None
    if isinstance(parsed, dict):
        normalized = {
            "reviewed_segments": parsed.get("reviewed_segments"),
            "corrected_segment_count": parsed.get("corrected_segment_count"),
            "episode_notes": parsed.get("episode_notes"),
        }
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "json_valid": bool(extracted.get("json_valid")),
        "schema_valid": bool(extracted.get("schema_valid")),
        "reasoning_chars": len(extracted.get("reasoning") or ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "finish_reason": extracted.get("finish_reason", ""),
        "reviewed_segment_ids": [
            item.get("id")
            for item in (normalized or {}).get("reviewed_segments", [])
            if isinstance(item, dict) and item.get("id") is not None
        ] if normalized else [],
        "normalized_output": normalized,
    }


def _run_request(url: str, model: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    started = time.perf_counter()
    response = _request_json(
        f"{url.rstrip('/')}/v1/chat/completions",
        _review_payload(model, segments),
        timeout=300,
    )
    result = _signature(response)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def _run_endpoint(url: str, model: str, contexts: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for target, segments in contexts.items():
        try:
            measured = _run_request(url, model, segments)
            results[str(target)] = {"measured": measured}
        except Exception as exc:
            results[str(target)] = {"error": str(exc)}
    return {"endpoint": url, "model": model, "contexts": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-url", default=DEFAULT_LOCAL_URL)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE_URL)
    parser.add_argument("--remote-model", default=DEFAULT_REMOTE_MODEL)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.local_model)
    contexts: Dict[int, List[Dict[str, Any]]] = {}
    context_sizes: Dict[str, int] = {}
    for target in TARGETS:
        segments, count = _make_context(tokenizer, args.local_model, target)
        contexts[target] = segments
        context_sizes[str(target)] = count

    report = {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "targets": list(TARGETS),
        "context_token_estimates": context_sizes,
        "needle_ids": [item["id"] for item in NEEDLES],
        "local": _run_endpoint(args.local_url, args.local_model, contexts),
        "remote": _run_endpoint(args.remote_url, args.remote_model, contexts),
    }
    comparisons = {}
    for target in TARGETS:
        local_entry = report["local"]["contexts"][str(target)]
        remote_entry = report["remote"]["contexts"][str(target)]
        local = local_entry.get("measured")
        remote = remote_entry.get("measured")
        comparison: Dict[str, Any] = {}
        if local and remote:
            comparison = {
                "local_seconds": local["elapsed_seconds"],
                "remote_seconds": remote["elapsed_seconds"],
                "remote_to_local_latency_ratio": round(
                    remote["elapsed_seconds"] / max(local["elapsed_seconds"], 0.001), 3
                ),
                "outputs_equal": local["normalized_output"] == remote["normalized_output"],
                "local_reviewed_ids": local["reviewed_segment_ids"],
                "remote_reviewed_ids": remote["reviewed_segment_ids"],
            }
        else:
            comparison["error"] = {
                "local": local_entry.get("error"),
                "remote": remote_entry.get("error"),
            }
        comparisons[str(target)] = comparison
    report["comparisons"] = comparisons
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
