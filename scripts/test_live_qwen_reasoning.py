#!/usr/bin/env python3
"""Probe Qwen reasoning controls against a live OpenAI-compatible vLLM server.

This is intentionally a standalone opt-in test. It does not run as part of the
normal unit-test suite because it requires a reachable vLLM endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "http://192.168.1.230:8000"
DEFAULT_MODEL = "Inferact/Qwen3.8-27B-NVFP4"
MAX_OUTPUT_TOKENS = 1400


SEGMENTS = [
    {"id": 1, "speaker": "HOST", "text": "Welcome back to the show. Today we are talking about sleep and recovery."},
    {"id": 2, "speaker": "GUEST", "text": "Thanks for having me. The most important factor is keeping a regular schedule."},
    {"id": 3, "speaker": "HOST", "text": "So consistency matters more than trying to catch up on weekends?"},
    {"id": 4, "speaker": "GUEST", "text": "Exactly. A stable wake time is usually easier to maintain than forcing an early bedtime."},
    {"id": 5, "speaker": "HOST", "text": "That is a useful distinction, and it gives listeners one practical place to start."},
]

SYSTEM_PROMPT = (
    "You are performing the 'Transcript cleanup review' stage of podcast transcript review. "
    "Correct only obvious transcription or grammar errors. Preserve meaning, segment ids, order, "
    "speaker labels, and unchanged text. Do not invent facts. Return only a JSON object with exactly "
    "these keys: reviewed_segments (array of objects containing id and text only), "
    "corrected_segment_count (integer), and episode_notes (array of strings). "
    "If no correction is needed, return an empty reviewed_segments array and corrected_segment_count 0."
)


def _request_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"connection failed: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"server returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("server returned a non-object JSON response")
    return parsed


def _mode_payload(mode: str, model: str) -> Dict[str, Any]:
    system_prompt = SYSTEM_PROMPT
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
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
        ],
    }

    if mode == "legacy_no_think_prompt":
        payload["messages"][0]["content"] = f"/no_think {system_prompt}"
    elif mode == "none":
        payload["messages"][0]["content"] = f"/no_think {system_prompt}"
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif mode in {"low", "medium", "xhigh"}:
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "reasoning_effort": mode,
        }
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return payload


def _extract_result(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {"content": "", "reasoning": "", "finish_reason": "", "json_valid": False, "schema_valid": False}
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content_text = content if isinstance(content, str) else json.dumps(content)
    reasoning_text = reasoning if isinstance(reasoning, str) else json.dumps(reasoning)
    parsed: Any = None
    parse_source = "content"
    try:
        parsed = json.loads(content_text)
    except json.JSONDecodeError:
        if not content_text.strip() and reasoning_text.strip():
            parse_source = "reasoning"
            try:
                parsed = json.loads(reasoning_text)
            except json.JSONDecodeError:
                parsed = None
    schema_valid = (
        isinstance(parsed, dict)
        and isinstance(parsed.get("reviewed_segments"), list)
        and isinstance(parsed.get("corrected_segment_count"), int)
        and isinstance(parsed.get("episode_notes"), list)
    )
    return {
        "content": content_text,
        "reasoning": reasoning_text,
        "finish_reason": str(choice.get("finish_reason") or choice.get("stop_reason") or ""),
        "json_valid": parsed is not None,
        "schema_valid": schema_valid,
        "parse_source": parse_source if parsed is not None else "",
        "reviewed_segment_count": len(parsed.get("reviewed_segments") or []) if isinstance(parsed, dict) else None,
        "corrected_segment_count": parsed.get("corrected_segment_count") if isinstance(parsed, dict) else None,
    }


def run_probe(base_url: str, model: str, timeout: float) -> Dict[str, Any]:
    base_url = base_url.rstrip("/")
    models = _request_json(f"{base_url}/v1/models", timeout=min(timeout, 30.0))
    model_ids = [
        str(item.get("id"))
        for item in (models.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    model_visible = model in model_ids

    results: List[Dict[str, Any]] = []
    for mode in ("legacy_no_think_prompt", "none", "low", "medium", "xhigh"):
        started = time.perf_counter()
        error = ""
        response: Dict[str, Any] = {}
        try:
            response = _request_json(
                f"{base_url}/v1/chat/completions",
                _mode_payload(mode, model),
                timeout=timeout,
            )
            extracted = _extract_result(response)
        except Exception as exc:  # Report one failed mode while continuing the comparison.
            extracted = {
                "content": "",
                "reasoning": "",
                "finish_reason": "",
                "json_valid": False,
                "schema_valid": False,
            }
            error = str(exc)
        elapsed = time.perf_counter() - started
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        results.append(
            {
                "mode": mode,
                "elapsed_seconds": round(elapsed, 3),
                "content_chars": len(extracted.get("content") or ""),
                "reasoning_chars": len(extracted.get("reasoning") or ""),
                "finish_reason": extracted.get("finish_reason", ""),
                "json_valid": bool(extracted.get("json_valid")),
                "schema_valid": bool(extracted.get("schema_valid")),
                "parse_source": extracted.get("parse_source", ""),
                "reviewed_segment_count": extracted.get("reviewed_segment_count"),
                "corrected_segment_count": extracted.get("corrected_segment_count"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (
                    usage.get("reasoning_tokens")
                    or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                    if isinstance(usage.get("completion_tokens_details"), dict)
                    else usage.get("reasoning_tokens")
                ),
                "error": error,
            }
        )

    return {
        "endpoint": base_url,
        "model": model,
        "model_visible_at_v1_models": model_visible,
        "visible_model_ids": model_ids,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "results": results,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"Endpoint: {report['endpoint']}")
    print(f"Model: {report['model']} (visible: {report['model_visible_at_v1_models']})")
    print(f"Max output tokens: {report['max_output_tokens']}")
    print("")
    print("mode                    seconds  completion  reasoning  JSON  schema  finish  error")
    print("----------------------  -------  ----------  ---------  ----  ------  ------  -----")
    for result in report["results"]:
        print(
            f"{result['mode']:<22}  {result['elapsed_seconds']:>7.3f}  "
            f"{str(result['completion_tokens'] or '-'):>10}  "
            f"{str(result['reasoning_tokens'] or '-'):>9}  "
            f"{'yes' if result['json_valid'] else 'no':>4}  "
            f"{'yes' if result['schema_valid'] else 'no':>6}  "
            f"{str(result['finish_reason'] or '-'):>6}  "
            f"{result['error'][:80]}"
        )

    successful = [item for item in report["results"] if not item["error"] and item["schema_valid"]]
    none_result = next((item for item in successful if item["mode"] == "none"), None)
    low_result = next((item for item in successful if item["mode"] == "low"), None)
    if none_result and low_result and low_result["elapsed_seconds"] > 0:
        speedup = low_result["elapsed_seconds"] / max(none_result["elapsed_seconds"], 0.001)
        print("")
        print(f"Hard-off vs low latency ratio: {speedup:.2f}x (values are single-request measurements).")
    print("")
    print("Assessment:")
    if none_result and none_result["reasoning_chars"] == 0:
        print("- The hard-off request suppressed visible reasoning and returned valid review JSON.")
    else:
        print("- The hard-off request did not produce a valid no-reasoning result; inspect the raw endpoint behavior.")
    if low_result:
        print("- The low-effort request returned valid review JSON; compare its latency and reasoning token count above.")
    else:
        print("- The low-effort request failed schema validation or transport.")
    xhigh_result = next((item for item in report["results"] if item["mode"] == "xhigh"), None)
    if xhigh_result and xhigh_result["reasoning_tokens"] and low_result and low_result["reasoning_tokens"]:
        print(
            f"- xhigh used {xhigh_result['reasoning_tokens']} reasoning tokens versus "
            f"{low_result['reasoning_tokens']} at low."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()
    try:
        report = run_probe(args.base_url, args.model, args.timeout)
    except Exception as exc:
        print(f"Live Qwen reasoning probe failed: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"JSON report: {args.output}")
    return 0 if any(item["schema_valid"] for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
