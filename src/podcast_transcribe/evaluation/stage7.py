"""Stage 7 quality, provider-promotion, and gold-set governance helpers.

The module deliberately operates on JSON reports and small in-memory records.
It is safe to use from the workbench without importing the ASR/diarization
runtime, while still giving the CLI a single auditable promotion gate.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


STAGE7_REPORT_VERSION = 1
SUPPORTED_GOLD_SET_VERSIONS = {1, 2}
CONDITION_TAGS = (
    "crosstalk",
    "noise",
    "music",
    "accent",
    "sponsor_read",
    "short_turn",
    "long_episode",
)


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def load_gold_set_manifest(gold_set_dir: Path) -> Dict[str, object]:
    path = gold_set_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Gold-set manifest must be an object: {path}")
    version = int(payload.get("gold_set_version") or 0)
    if version not in SUPPORTED_GOLD_SET_VERSIONS:
        raise RuntimeError(f"Unsupported gold_set_version {version} in {path}")
    if not isinstance(payload.get("entries"), list):
        raise RuntimeError(f"Gold-set manifest is missing entries: {path}")
    return payload


def gold_set_readiness(gold_set_dir: Path) -> Dict[str, object]:
    """Report which gold-set entries are ready without modifying any files."""

    manifest = load_gold_set_manifest(gold_set_dir)
    entries: List[Dict[str, object]] = []
    for raw in manifest.get("entries") or []:
        if not isinstance(raw, dict) or raw.get("enabled") is False:
            continue
        reference = gold_set_dir / str(raw.get("reference") or "")
        approval = str(raw.get("approval_status") or "pending_review").lower()
        tags = [str(tag) for tag in (raw.get("tags") or raw.get("error_taxonomy") or [])]
        missing = []
        if not reference.exists():
            missing.append("reference_missing")
        if not raw.get("segment_ids"):
            missing.append("segment_ids_missing")
        if approval not in {"approved", "human_approved"}:
            missing.append("human_approval_pending")
        entries.append(
            {
                "id": str(raw.get("id") or reference.stem),
                "reference": str(reference),
                "approval_status": approval,
                "reviewer_id": str(raw.get("reviewer_id") or ""),
                "condition_tags": sorted(set(tags)),
                "ready": not missing,
                "missing": missing,
            }
        )
    return {
        "stage7_report_version": STAGE7_REPORT_VERSION,
        "gold_set_version": int(manifest.get("gold_set_version") or 0),
        "name": str(manifest.get("name") or gold_set_dir.name),
        "entry_count": len(entries),
        "ready_count": sum(bool(item["ready"]) for item in entries),
        "pending_count": sum(not bool(item["ready"]) for item in entries),
        "conditions": sorted({tag for item in entries for tag in item["condition_tags"]}),
        "entries": entries,
        "manifest_fingerprint": _hash_payload(manifest),
    }


def condition_report(benchmark_report: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Aggregate benchmark results by gold-set condition tag."""

    output: Dict[str, Dict[str, object]] = {}
    for item in benchmark_report.get("results") or []:
        if not isinstance(item, dict):
            continue
        tags = item.get("error_taxonomy") or item.get("tags") or ["untagged"]
        for tag in tags:
            name = str(tag)
            bucket = output.setdefault(name, {"entry_count": 0, "reference_words": 0, "wer_errors": 0, "timestamp_errors": []})
            bucket["entry_count"] += 1
            wer = item.get("wer") if isinstance(item.get("wer"), dict) else {}
            bucket["reference_words"] += int(wer.get("reference_words") or 0)
            bucket["wer_errors"] += int(wer.get("errors") or 0)
            timestamp = item.get("timestamp_error") if isinstance(item.get("timestamp_error"), dict) else {}
            if timestamp.get("mean_absolute_error_seconds") is not None:
                bucket["timestamp_errors"].append(float(timestamp["mean_absolute_error_seconds"]))
    for bucket in output.values():
        values = bucket.pop("timestamp_errors")
        bucket["wer"] = bucket["wer_errors"] / bucket["reference_words"] if bucket["reference_words"] else 0.0
        bucket["mean_timestamp_error_seconds"] = sum(values) / len(values) if values else None
    return dict(sorted(output.items()))


def provider_promotion_report(
    baseline_report: Dict[str, object],
    candidate_report: Dict[str, object],
    *,
    provider_stage: str,
    max_runtime_regression: float = 0.25,
    max_memory_regression: float = 0.25,
) -> Dict[str, object]:
    """Compare provider reports and require explicit guardrail approval."""

    baseline = baseline_report.get("aggregate") if isinstance(baseline_report.get("aggregate"), dict) else {}
    candidate = candidate_report.get("aggregate") if isinstance(candidate_report.get("aggregate"), dict) else {}

    def value(payload: Dict[str, object], key: str, nested: Optional[str] = None) -> Optional[float]:
        current = payload.get(key)
        if nested and isinstance(current, dict):
            current = current.get(nested)
        try:
            return None if current is None else float(current)
        except (TypeError, ValueError):
            return None

    deltas = {
        "wer": (value(candidate, "wer", "wer") or 0.0) - (value(baseline, "wer", "wer") or 0.0),
        "speaker_attributed_wer": (value(candidate, "speaker_attributed_wer", "speaker_attributed_wer") or 0.0) - (value(baseline, "speaker_attributed_wer", "speaker_attributed_wer") or 0.0),
        "timestamp_error_seconds": (value(candidate, "mean_timestamp_error_seconds") or 0.0) - (value(baseline, "mean_timestamp_error_seconds") or 0.0),
        "runtime_seconds": (value(candidate, "mean_processing_seconds") or 0.0) - (value(baseline, "mean_processing_seconds") or 0.0),
        "host_precision": (value(candidate, "host_precision") or 0.0) - (value(baseline, "host_precision") or 0.0),
        "host_recall": (value(candidate, "host_recall") or 0.0) - (value(baseline, "host_recall") or 0.0),
    }
    for resource_key in ("peak_cpu_working_set_mib", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib"):
        candidate_resource = candidate.get("resource_usage") if isinstance(candidate.get("resource_usage"), dict) else {}
        baseline_resource = baseline.get("resource_usage") if isinstance(baseline.get("resource_usage"), dict) else {}
        deltas[resource_key] = (value(candidate_resource.get(resource_key) if isinstance(candidate_resource.get(resource_key), dict) else {}, "max") or 0.0) - (value(baseline_resource.get(resource_key) if isinstance(baseline_resource.get(resource_key), dict) else {}, "max") or 0.0)
    failures = []
    if deltas["wer"] > 0:
        failures.append("wer_regression")
    if deltas["speaker_attributed_wer"] > 0:
        failures.append("speaker_attributed_wer_regression")
    if deltas["timestamp_error_seconds"] > 0:
        failures.append("timing_regression")
    baseline_runtime = value(baseline, "mean_processing_seconds") or 0.0
    if baseline_runtime and deltas["runtime_seconds"] / baseline_runtime > max_runtime_regression:
        failures.append("runtime_regression")
    for resource_key in ("peak_cpu_working_set_mib", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib"):
        baseline_resource = baseline.get("resource_usage") if isinstance(baseline.get("resource_usage"), dict) else {}
        baseline_max = value(baseline_resource.get(resource_key) if isinstance(baseline_resource.get(resource_key), dict) else {}, "max") or 0.0
        if baseline_max and deltas[resource_key] / baseline_max > max_memory_regression:
            failures.append(f"{resource_key}_regression")
    return {
        "stage7_report_version": STAGE7_REPORT_VERSION,
        "provider_stage": provider_stage,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_identity": baseline_report.get("candidate_provenance") or baseline_report.get("candidate_dir") or "",
        "candidate_identity": candidate_report.get("candidate_provenance") or candidate_report.get("candidate_dir") or "",
        "deltas": deltas,
        "guardrails": {"max_runtime_regression": max_runtime_regression, "max_memory_regression": max_memory_regression},
        "passed": not failures,
        "failures": failures,
        "conditions": {"baseline": condition_report(baseline_report), "candidate": condition_report(candidate_report)},
    }


def write_stage7_report(output_dir: Path, report: Dict[str, object], stem: str = "stage7_promotion_report") -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stage 7 Promotion Report",
        "",
        f"- Stage: {report.get('provider_stage', '')}",
        f"- Passed: {bool(report.get('passed'))}",
        f"- Failures: {', '.join(report.get('failures') or []) or 'none'}",
        "",
        "## Deltas",
        "",
    ]
    for key, value in (report.get("deltas") or {}).items():
        lines.append(f"- {key}: {float(value):+.6f}")
    if report.get("conditions"):
        lines.extend(["", "## Conditions", ""])
        for name, values in (report["conditions"].get("candidate") or {}).items():
            lines.append(f"- {name}: WER={float(values.get('wer') or 0.0):.4f}, entries={values.get('entry_count', 0)}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def validate_alignment_provider_report(report: Dict[str, object], expected_provider: str) -> List[str]:
    """Return actionable diagnostics for alignment-provider promotion inputs."""

    failures: List[str] = []
    provenance = report.get("candidate_provenance") or {}
    if isinstance(provenance, dict):
        provider = str(provenance.get("provider") or provenance.get("alignment_provider") or "")
        if provider and provider != expected_provider:
            failures.append(f"alignment_provider_mismatch:{provider}")
    if not report.get("results"):
        failures.append("no_gold_results")
    return failures
