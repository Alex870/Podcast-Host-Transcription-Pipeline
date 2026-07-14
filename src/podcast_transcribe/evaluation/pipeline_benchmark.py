"""Gold-set pipeline benchmark runner and reports."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from podcast_transcribe.contract import validate_transcript_payload
from podcast_transcribe.evaluation.metrics import (
    aggregate_metric_counts,
    diarization_error_rate,
    glossary_preservation,
    host_classification,
    speaker_attributed_wer,
    timestamp_error,
    word_error_rate,
)


GOLD_SET_VERSION = 1
PIPELINE_BENCHMARK_VERSION = 2


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def load_gold_manifest(gold_set_dir: Path) -> Dict[str, object]:
    manifest_path = gold_set_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Gold-set manifest not found: {manifest_path}")
    payload = _load_json(manifest_path)
    if int(payload.get("gold_set_version") or 0) != GOLD_SET_VERSION:
        raise RuntimeError(f"Unsupported gold_set_version in {manifest_path}")
    if not isinstance(payload.get("entries"), list):
        raise RuntimeError(f"Gold-set manifest is missing entries: {manifest_path}")
    return payload


def _candidate_path(entry: Dict[str, object], candidate_dir: Optional[Path], gold_set_dir: Path) -> Path:
    explicit = str(entry.get("candidate") or "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else gold_set_dir / path
    if candidate_dir is None:
        raise RuntimeError(f"Gold-set entry {entry.get('id')} has no candidate path and no candidate directory was supplied.")
    stem = str(entry.get("audio_stem") or entry.get("id") or "").strip()
    layer = str(entry.get("candidate_layer") or "cleaned").strip().lower()
    suffix = "_reviewed_speaker_transcript.json" if layer == "reviewed" else "_cleaned_speaker_transcript.json"
    return candidate_dir / f"{stem}{suffix}"


def _transcript_text(payload: Dict[str, object]) -> str:
    return " ".join(str(segment.get("text") or "") for segment in payload.get("segments") or [] if isinstance(segment, dict))


def evaluate_entry(entry: Dict[str, object], gold_set_dir: Path, candidate_dir: Optional[Path]) -> Dict[str, object]:
    reference_path = gold_set_dir / str(entry.get("reference") or "")
    candidate_path = _candidate_path(entry, candidate_dir, gold_set_dir)
    reference = _load_json(reference_path)
    candidate = _load_json(candidate_path)
    candidate_name = candidate_path.name
    for suffix in ("_cleaned_speaker_transcript.json", "_reviewed_speaker_transcript.json", "_speaker_transcript.json"):
        if candidate_name.endswith(suffix):
            candidate_name = candidate_name[: -len(suffix)]
            break
    manifest_path = candidate_path.parent / f"{candidate_name}_manifest.json"
    candidate_manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    reference_errors = validate_transcript_payload(reference)
    candidate_errors = validate_transcript_payload(candidate)
    if reference_errors:
        raise RuntimeError(f"Invalid gold reference {reference_path}: {'; '.join(reference_errors)}")
    if candidate_errors:
        raise RuntimeError(f"Invalid candidate {candidate_path}: {'; '.join(candidate_errors)}")
    segment_ids = {str(value) for value in entry.get("segment_ids") or []}
    reference_segments = [
        segment
        for segment in reference.get("segments") or []
        if isinstance(segment, dict) and (not segment_ids or str(segment.get("id")) in segment_ids)
    ]
    candidate_segments = [
        segment
        for segment in candidate.get("segments") or []
        if isinstance(segment, dict) and (not segment_ids or str(segment.get("id")) in segment_ids)
    ]
    reference_text = " ".join(str(segment.get("text") or "") for segment in reference_segments)
    candidate_text = " ".join(str(segment.get("text") or "") for segment in candidate_segments)
    return {
        "id": str(entry.get("id") or reference_path.stem),
        "tags": list(entry.get("tags") or []),
        "error_taxonomy": list(entry.get("error_taxonomy") or entry.get("tags") or []),
        "segment_ids": sorted(segment_ids),
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "wer": word_error_rate(reference_text, candidate_text),
        "speaker_attributed_wer": speaker_attributed_wer(reference_segments, candidate_segments),
        "timestamp_error": timestamp_error(reference_segments, candidate_segments),
        "diarization": diarization_error_rate(
            [turn for turn in reference.get("diarization_turns") or [] if isinstance(turn, dict)],
            [turn for turn in candidate.get("diarization_turns") or [] if isinstance(turn, dict)],
            collar_seconds=float(entry.get("der_collar_seconds") or 0.25),
        ),
        "host_classification": host_classification(reference_segments, candidate_segments),
        "glossary": glossary_preservation(entry.get("preferred_terms") or [], reference_text, candidate_text),
        "candidate_provenance": candidate.get("metadata", {}).get("stage_provenance", {}) if isinstance(candidate.get("metadata"), dict) else {},
        "performance": {
            "timings_seconds": candidate_manifest.get("timings_seconds") or {},
            "resource_usage": candidate_manifest.get("resource_usage") or {},
            "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        },
    }


def run_pipeline_benchmark(
    gold_set_dir: Path,
    candidate_dir: Optional[Path] = None,
    baseline_dir: Optional[Path] = None,
) -> Dict[str, object]:
    manifest = load_gold_manifest(gold_set_dir)
    results: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for entry in manifest.get("entries") or []:
        if not isinstance(entry, dict) or entry.get("enabled") is False:
            continue
        try:
            results.append(evaluate_entry(entry, gold_set_dir, candidate_dir))
        except Exception as exc:
            failures.append({"id": str(entry.get("id") or "unknown"), "error": str(exc)})

    wer_items = [item["wer"] for item in results]
    sa_items = [item["speaker_attributed_wer"] for item in results]
    host_tp = sum(int(item["host_classification"]["true_positive"]) for item in results)
    host_fp = sum(int(item["host_classification"]["false_positive"]) for item in results)
    host_fn = sum(int(item["host_classification"]["false_negative"]) for item in results)
    glossary_expected = sum(int(item["glossary"]["expected_count"]) for item in results)
    glossary_preserved = sum(int(item["glossary"]["preserved_count"]) for item in results)
    timestamp_values = [
        float(item["timestamp_error"]["mean_absolute_error_seconds"])
        for item in results
        if item["timestamp_error"]["mean_absolute_error_seconds"] is not None
    ]
    diarization_scored = sum(float(item["diarization"]["scored_seconds"] or 0.0) for item in results)
    diarization_errors = sum(float(item["diarization"]["error_seconds"] or 0.0) for item in results)
    total_timings = [
        float((item.get("performance", {}).get("timings_seconds") or {}).get("total") or 0.0)
        for item in results
        if float((item.get("performance", {}).get("timings_seconds") or {}).get("total") or 0.0) > 0
    ]
    taxonomy: Dict[str, Dict[str, object]] = {}
    for tag in sorted({tag for item in results for tag in item.get("error_taxonomy") or []}):
        tagged = [item for item in results if tag in (item.get("error_taxonomy") or [])]
        tagged_words = sum(int(item["wer"]["reference_words"]) for item in tagged)
        tagged_errors = sum(int(item["wer"]["errors"]) for item in tagged)
        taxonomy[tag] = {
            "entry_count": len(tagged),
            "wer": tagged_errors / tagged_words if tagged_words else 0.0,
            "mean_timestamp_error_seconds": (
                sum(float(item["timestamp_error"]["mean_absolute_error_seconds"]) for item in tagged if item["timestamp_error"]["mean_absolute_error_seconds"] is not None)
                / max(1, sum(item["timestamp_error"]["mean_absolute_error_seconds"] is not None for item in tagged))
            ),
        }
    report = {
        "pipeline_benchmark_version": PIPELINE_BENCHMARK_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gold_set": {"path": str(gold_set_dir), "name": manifest.get("name", gold_set_dir.name), "entry_count": len(results)},
        "candidate_dir": str(candidate_dir) if candidate_dir else "",
        "aggregate": {
            "wer": aggregate_metric_counts(wer_items, "errors", "reference_words", "wer"),
            "speaker_attributed_wer": aggregate_metric_counts(sa_items, "errors", "reference_words", "speaker_attributed_wer"),
            "host_precision": host_tp / (host_tp + host_fp) if host_tp + host_fp else 1.0,
            "host_recall": host_tp / (host_tp + host_fn) if host_tp + host_fn else 1.0,
            "glossary_preservation_rate": glossary_preserved / glossary_expected if glossary_expected else 1.0,
            "mean_timestamp_error_seconds": sum(timestamp_values) / len(timestamp_values) if timestamp_values else None,
            "diarization_error_rate": diarization_errors / diarization_scored if diarization_scored else None,
            "completion_rate": len(results) / (len(results) + len(failures)) if results or failures else 0.0,
            "mean_processing_seconds": sum(total_timings) / len(total_timings) if total_timings else None,
        },
        "error_taxonomy": taxonomy,
        "results": results,
        "failures": failures,
    }
    if baseline_dir is not None:
        baseline = run_pipeline_benchmark(gold_set_dir, baseline_dir)
        candidate_aggregate = report["aggregate"]
        baseline_aggregate = baseline["aggregate"]
        comparison = {
            "baseline_dir": str(baseline_dir),
            "wer_delta": float(candidate_aggregate["wer"]["wer"]) - float(baseline_aggregate["wer"]["wer"]),
            "speaker_attributed_wer_delta": float(candidate_aggregate["speaker_attributed_wer"]["speaker_attributed_wer"]) - float(baseline_aggregate["speaker_attributed_wer"]["speaker_attributed_wer"]),
            "diarization_error_rate_delta": (float(candidate_aggregate["diarization_error_rate"] or 0.0) - float(baseline_aggregate["diarization_error_rate"] or 0.0)),
            "processing_seconds_delta": (float(candidate_aggregate["mean_processing_seconds"] or 0.0) - float(baseline_aggregate["mean_processing_seconds"] or 0.0)),
        }
        thresholds = manifest.get("promotion_thresholds") if isinstance(manifest.get("promotion_thresholds"), dict) else {}
        failures_to_promote = []
        if comparison["wer_delta"] > float(thresholds.get("max_wer_regression") or 0.0):
            failures_to_promote.append("wer_regression")
        if comparison["speaker_attributed_wer_delta"] > float(thresholds.get("max_speaker_attributed_wer_regression") or 0.0):
            failures_to_promote.append("speaker_attributed_wer_regression")
        minimum_completion = float(thresholds.get("minimum_completion_rate") or 1.0)
        if float(candidate_aggregate["completion_rate"]) < minimum_completion:
            failures_to_promote.append("completion_rate")
        report["baseline"] = baseline["aggregate"]
        report["comparison"] = comparison
        report["promotion"] = {"passed": not failures_to_promote, "failures": failures_to_promote, "thresholds": thresholds}
    return report


def write_pipeline_benchmark_reports(output_dir: Path, report: Dict[str, object]) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pipeline_quality_benchmark_report.json"
    markdown_path = output_dir / "pipeline_quality_benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    aggregate = report.get("aggregate") or {}
    lines = [
        "# Pipeline Quality Benchmark",
        "",
        f"- Gold set: {report.get('gold_set', {}).get('name', '')}",
        f"- Completed entries: {report.get('gold_set', {}).get('entry_count', 0)}",
        f"- WER: {float((aggregate.get('wer') or {}).get('wer') or 0.0):.4f}",
        f"- Speaker-attributed WER: {float((aggregate.get('speaker_attributed_wer') or {}).get('speaker_attributed_wer') or 0.0):.4f}",
        f"- Diarization error rate: {float(aggregate.get('diarization_error_rate') or 0.0):.4f}",
        f"- Host precision: {float(aggregate.get('host_precision') or 0.0):.4f}",
        f"- Host recall: {float(aggregate.get('host_recall') or 0.0):.4f}",
        f"- Glossary preservation: {float(aggregate.get('glossary_preservation_rate') or 0.0):.4f}",
        f"- Completion rate: {float(aggregate.get('completion_rate') or 0.0):.4f}",
        "",
        "## Entries",
        "",
        "| Entry | WER | Speaker-attributed WER | Host precision | Host recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report.get("results") or []:
        lines.append(
            f"| {item.get('id', '')} | {float(item['wer']['wer']):.4f} | "
            f"{float(item['speaker_attributed_wer']['speaker_attributed_wer']):.4f} | "
            f"{float(item['host_classification']['precision']):.4f} | {float(item['host_classification']['recall']):.4f} |"
        )
    failures = report.get("failures") or []
    if report.get("comparison"):
        comparison = report["comparison"]
        lines.extend([
            "", "## Baseline Comparison", "",
            f"- WER delta: {float(comparison.get('wer_delta') or 0.0):+.4f}",
            f"- Speaker-attributed WER delta: {float(comparison.get('speaker_attributed_wer_delta') or 0.0):+.4f}",
            f"- Promotion passed: {bool((report.get('promotion') or {}).get('passed'))}",
        ])
    if report.get("error_taxonomy"):
        lines.extend(["", "## Error Taxonomy", "", "| Category | Entries | WER |", "|---|---:|---:|"])
        for name, values in report["error_taxonomy"].items():
            lines.append(f"| {name} | {values.get('entry_count', 0)} | {float(values.get('wer') or 0.0):.4f} |")
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item.get('id')}: {item.get('error')}" for item in failures)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
