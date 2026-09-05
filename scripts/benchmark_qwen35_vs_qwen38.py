#!/usr/bin/env python3
"""Compare local Qwen3.5-9B NVFP4 against the configured Qwen3.8 vLLM server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from test_live_qwen_reasoning import (
    SEGMENTS,
    SYSTEM_PROMPT,
    _extract_result,
    _mode_payload,
    _request_json,
)


DEFAULT_LOCAL_MODEL = "RishabhSinha/Qwen3.5-9B-NVFP4"
DEFAULT_REMOTE_URL = "http://192.168.1.230:8000"
DEFAULT_REMOTE_MODEL = "Inferact/Qwen3.8-27B-NVFP4"
MAX_OUTPUT_TOKENS = 1400


def _messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": f"/no_think {SYSTEM_PROMPT}"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "stage_name": "transcript_cleanup_review",
                    "stage_mode": "local_batch",
                    "edit_scope": "text_only",
                    "return_only_changed_segments": True,
                    "changed_segments_only": True,
                    "max_changed_segments_hint": 8,
                    "segments": SEGMENTS,
                },
                ensure_ascii=True,
            ),
        },
    ]


def _summarize_response(response: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    extracted = _extract_result(response)
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "elapsed_seconds": round(elapsed, 3),
        "json_valid": bool(extracted.get("json_valid")),
        "schema_valid": bool(extracted.get("schema_valid")),
        "reasoning_chars": len(extracted.get("reasoning") or ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "finish_reason": extracted.get("finish_reason", ""),
    }


def benchmark_endpoint(base_url: str, model: str, repetitions: int) -> Dict[str, Any]:
    payload = _mode_payload("none", model)
    timings = []
    last_response: Dict[str, Any] = {}
    for _ in range(repetitions + 1):
        started = time.perf_counter()
        last_response = _request_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout=180,
        )
        result = _summarize_response(last_response, time.perf_counter() - started)
        if len(timings) > 0:
            timings.append(result)
        else:
            timings.append({"warmup": True, **result})
    measured = timings[1:]
    return {
        "mode": "none",
        "execution": "vllm_http",
        "endpoint": base_url,
        "model": model,
        "warmup": timings[0],
        "measured": measured,
        "mean_seconds": round(sum(x["elapsed_seconds"] for x in measured) / len(measured), 3),
        "min_seconds": round(min(x["elapsed_seconds"] for x in measured), 3),
        "max_seconds": round(max(x["elapsed_seconds"] for x in measured), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--local-url", default="http://127.0.0.1:8001")
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE_URL)
    parser.add_argument("--remote-model", default=DEFAULT_REMOTE_MODEL)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "repetitions": args.repetitions,
        "local": benchmark_endpoint(args.local_url, args.local_model, args.repetitions),
        "remote": benchmark_endpoint(args.remote_url, args.remote_model, args.repetitions),
    }
    local_mean = report["local"]["mean_seconds"]
    remote_mean = report["remote"]["mean_seconds"]
    report["remote_to_local_speedup"] = round(remote_mean / local_mean, 2)
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
