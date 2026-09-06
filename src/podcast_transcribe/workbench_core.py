"""Core helpers for the transcript review workbench.

This module stays intentionally lightweight so the workbench can inspect
processed artifacts without importing the heavy transcription runtime.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from podcast_transcribe.config import resolve_review_runtime_config
from podcast_transcribe.contract import validate_reviewed_transcript_payload, validate_transcript_payload
from podcast_transcribe.ecosystem_contracts import (
    build_correction_manifest_history,
    canonical_id,
    correction_identity_payload,
)
from podcast_transcribe.learned_rules import (
    LEARNED_RULE_ALLOWED_FAMILIES,
    LEARNED_RULE_ALLOWED_STAGES,
    approved_review_rules,
    get_review_rule as load_single_review_rule,
    load_review_rule_library,
    normalize_learned_rule,
    save_review_rule_library,
    upsert_review_rule,
)
from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.outputs import (
    write_batch_report_md,
    write_json_output,
    write_review_run_report,
    write_speaker_workflow_report,
    write_text_transcript,
)
from podcast_transcribe.review import _chat_template_request_controls, review_segments
from podcast_transcribe.speaker_workflow import (
    assert_write_revision,
    build_cross_episode_speaker_view,
    file_revision,
)


WORKBENCH_DIRNAME = "_workbench"
SCAN_CACHE_SUBDIR = "semantic_scan"
AUDIT_LOG_SUBDIR = ".workbench"
AUDIT_LOG_FILENAME = "audit_log.jsonl"
ISSUE_RESOLUTION_SUBDIR = "issue_resolution"
DEFAULT_CORRECTIONS_DIRNAME = "corrections"
TEACH_ME_SUBDIR = "teach_me"
TEACH_ME_CONTROL_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "review_fixtures" / "teach_me_controls.json"
PIPELINE_GOLD_SET_DIR = Path("benchmarks") / "pipeline_gold_set"
PIPELINE_GOLD_ANNOTATIONS_DIRNAME = "annotations"
CORRECTION_MANIFEST_DIRNAME = "_correction_manifests"

_GLOSSARY_WRITE_LOCK = threading.Lock()


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _resolve_under_root(root: Path, raw_path: Optional[str], default_relative: Optional[str] = None) -> Path:
    value = (raw_path or "").strip()
    if value:
        candidate = Path(value)
        path = candidate if candidate.is_absolute() else root / candidate
    elif default_relative:
        path = root / default_relative
    else:
        path = root
    return path.resolve()


def _cli_helpers():
    try:
        from podcast_transcribe.cli import (
            build_review_backfill_summary_row,
            normalize_episode_summary_row,
            segment_items_from_cleaned_payload,
            write_episode_summary_csv,
            write_reviewed_output_bundle,
            write_run_reports,
        )

        return {
            "build_review_backfill_summary_row": build_review_backfill_summary_row,
            "normalize_episode_summary_row": normalize_episode_summary_row,
            "segment_items_from_cleaned_payload": segment_items_from_cleaned_payload,
            "write_episode_summary_csv": write_episode_summary_csv,
            "write_reviewed_output_bundle": write_reviewed_output_bundle,
            "write_run_reports": write_run_reports,
        }
    except ModuleNotFoundError:
        return {
            "build_review_backfill_summary_row": _fallback_build_review_backfill_summary_row,
            "normalize_episode_summary_row": _fallback_normalize_episode_summary_row,
            "segment_items_from_cleaned_payload": _fallback_segment_items_from_cleaned_payload,
            "write_episode_summary_csv": _fallback_write_episode_summary_csv,
            "write_reviewed_output_bundle": _fallback_write_reviewed_output_bundle,
            "write_run_reports": _fallback_write_run_reports,
        }


def _state_helpers():
    from podcast_transcribe.state import SUMMARY_FILENAME, load_episode_summary_rows

    return {
        "SUMMARY_FILENAME": SUMMARY_FILENAME,
        "load_episode_summary_rows": load_episode_summary_rows,
    }


def _coerce_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _fallback_segment_items_from_cleaned_payload(payload: Dict[str, object]) -> List[SegmentItem]:
    rebuilt_segments: List[SegmentItem] = []
    for index, raw_segment in enumerate(payload.get("segments") or []):
        if not isinstance(raw_segment, dict):
            raise RuntimeError(f"Segment {index} in cleaned transcript JSON is not an object.")
        words_payload = raw_segment.get("words") or []
        words = [
            WordItem(
                start=word.get("start") if isinstance(word, dict) else None,
                end=word.get("end") if isinstance(word, dict) else None,
                word=str(word.get("word") or "") if isinstance(word, dict) else "",
                speaker=str(word.get("speaker") or raw_segment.get("speaker") or "") if isinstance(word, dict) else "",
            )
            for word in words_payload
            if isinstance(word, dict)
        ]
        rebuilt_segments.append(
            SegmentItem(
                id=int(raw_segment["id"]),
                start=float(raw_segment["start"]),
                end=float(raw_segment["end"]),
                text=str(raw_segment["text"]),
                speaker=str(raw_segment.get("speaker") or ""),
                avg_logprob=raw_segment.get("avg_logprob"),
                no_speech_prob=raw_segment.get("no_speech_prob"),
                words=words,
                original_text=raw_segment.get("original_text"),
                cleanup_applied=_coerce_bool(raw_segment.get("cleanup_applied"), False),
                cleanup_level=str(raw_segment.get("cleanup_level") or ""),
                manual_correction_applied=_coerce_bool(raw_segment.get("manual_correction_applied"), False),
                original_speaker=raw_segment.get("original_speaker"),
            )
        )
    return rebuilt_segments


def _fallback_write_episode_summary_csv(path: Path, rows: List[Dict[str, object]]):
    fieldnames = sorted({key for row in rows for key in row.keys()} | {"episode"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fallback_normalize_episode_summary_row(row: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(row)
    for key in ("processing_seconds", "review_priority_score", "host_duration_seconds", "host_share_of_speech"):
        if key in normalized and normalized[key] not in ("", None):
            normalized[key] = _coerce_float(normalized[key], 0.0)
    for key in (
        "transcript_segments",
        "reviewed_segment_count",
        "review_corrected_segment_count",
        "review_unique_stage_count",
        "preferred_term_intervention_count",
    ):
        if key in normalized:
            normalized[key] = _coerce_int(normalized[key], 0)
    for key in ("review_attempted", "reviewed_output_written", "review_material_change", "episode_qa_added_value"):
        if key in normalized:
            normalized[key] = _coerce_bool(normalized[key], False)
    return normalized


def _fallback_write_run_reports(output_dir: Path, rows: List[Dict[str, object]], elapsed_seconds: Optional[float] = None):
    write_batch_report_md(output_dir / "_batch_report.md", rows, elapsed_seconds=elapsed_seconds)
    write_review_run_report(output_dir, rows, elapsed_seconds=elapsed_seconds)
    write_speaker_workflow_report(output_dir, rows)


def _fallback_write_reviewed_output_bundle(
    audio_path: Path,
    output_dir: Path,
    reviewed_segments: List[SegmentItem],
    review_metadata: Dict[str, object],
    host_output_labels: set[str],
    episode_metadata: Dict[str, object],
    info_payload: Dict[str, object],
    diarized_turns: List[Dict[str, object]],
    speaker_mapping: Dict[str, str],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    known_assignments: Dict[str, Dict[str, object]],
    runtime_config: Optional[Dict[str, object]],
) -> List[Path]:
    if not reviewed_segments:
        return []
    reviewed_text_version = (
        "reviewed_llm_high_context"
        if review_metadata.get("review_runtime_profile") == "high_context_5090"
        else "reviewed_llm"
    )
    reviewed_metadata = {**episode_metadata, "text_version": reviewed_text_version}
    base_name = audio_path.stem
    reviewed_paths = [
        output_dir / f"{base_name}_reviewed_speaker_transcript.txt",
        output_dir / f"{base_name}_reviewed_host_only.txt",
        output_dir / f"{base_name}_reviewed_speaker_transcript.json",
    ]
    write_text_transcript(
        reviewed_paths[0],
        reviewed_segments,
        _format_timestamp,
        host_only=False,
        metadata=reviewed_metadata,
    )
    write_text_transcript(
        reviewed_paths[1],
        reviewed_segments,
        _format_timestamp,
        host_only=True,
        host_labels=host_output_labels,
        metadata=reviewed_metadata,
    )
    write_json_output(
        reviewed_paths[2],
        source_file=str(audio_path),
        info_payload=info_payload,
        diarized_turns=diarized_turns,
        segments=reviewed_segments,
        speaker_mapping=speaker_mapping,
        host_speaker=host_speaker,
        durations=durations,
        known_assignments=known_assignments,
        metadata=reviewed_metadata,
        text_version=reviewed_text_version,
        pipeline_version=runtime_config.get("model", "") if runtime_config else "",
        review_metadata=review_metadata,
    )
    return reviewed_paths


def _fallback_build_review_backfill_summary_row(
    audio_path: Path,
    cleaned_payload: Dict[str, object],
    cleaned_segments: List[SegmentItem],
    review_result: Dict[str, object],
    existing_summary_row: Optional[Dict[str, object]] = None,
    processing_seconds: float = 0.0,
) -> Dict[str, object]:
    metadata = cleaned_payload.get("metadata") if isinstance(cleaned_payload.get("metadata"), dict) else {}
    review_metadata = review_result.get("metadata") if isinstance(review_result.get("metadata"), dict) else {}
    stage_results = review_metadata.get("review_stage_results") if isinstance(review_metadata.get("review_stage_results"), dict) else {}
    change_summary = review_metadata.get("review_change_summary") if isinstance(review_metadata.get("review_change_summary"), dict) else {}
    row = dict(existing_summary_row or {})
    row.update(
        {
            "episode": audio_path.name,
            "episode_date": metadata.get("episode_date", ""),
            "episode_date_compact": metadata.get("episode_date_compact", ""),
            "episode_year": metadata.get("episode_year", ""),
            "episode_month": metadata.get("episode_month", ""),
            "episode_day": metadata.get("episode_day", ""),
            "episode_sort_key": metadata.get("episode_sort_key", ""),
            "transcript_segments": len(cleaned_segments),
            "processing_seconds": processing_seconds,
            "review_attempted": bool(review_result.get("attempted")),
            "review_status": str(review_metadata.get("review_status") or ""),
            "review_skip_reason": str(review_metadata.get("review_skip_reason") or ""),
            "review_runtime_profile": str(review_metadata.get("review_runtime_profile") or ""),
            "review_backend": str(review_metadata.get("review_backend") or ""),
            "review_model_name": str(review_metadata.get("review_model_name") or ""),
            "reviewed_segment_count": int(review_metadata.get("reviewed_segment_count") or 0),
            "review_corrected_segment_count": int(review_metadata.get("corrected_segment_count") or 0),
            "reviewed_output_written": bool(review_result.get("segments")),
            "review_pipeline_version": str(review_metadata.get("review_pipeline_version") or ""),
            "review_enabled_stages": ";".join(str(item) for item in review_metadata.get("review_enabled_stages") or []),
            "review_completed_stages": ";".join(str(item) for item in review_metadata.get("review_completed_stages") or []),
            "review_skipped_stages": ";".join(str(item) for item in review_metadata.get("review_skipped_stages") or []),
            "review_input_source": str(review_metadata.get("review_input_source") or ""),
            "review_episode_qa_mode": str(review_metadata.get("episode_qa_mode") or ""),
            "active_learned_rule_ids": ";".join(str(item) for item in review_metadata.get("active_learned_rule_ids") or []),
            "contributing_learned_rule_ids": ";".join(str(item) for item in review_metadata.get("contributing_learned_rule_ids") or []),
            "cleanup_review_corrected_count": int(((stage_results.get("transcript_cleanup_review") or {}).get("corrected_segment_count")) or 0),
            "glossary_review_corrected_count": int(((stage_results.get("glossary_correction_review") or {}).get("corrected_segment_count")) or 0),
            "speaker_consistency_review_corrected_count": int(((stage_results.get("speaker_consistency_review") or {}).get("corrected_segment_count")) or 0),
            "episode_qa_review_corrected_count": int(((stage_results.get("episode_qa_review") or {}).get("corrected_segment_count")) or 0),
            "review_material_change": bool(change_summary.get("material_change")),
            "review_unique_stage_count": int(change_summary.get("unique_stage_count") or 0),
            "preferred_term_intervention_count": int(change_summary.get("protected_term_intervention_count") or 0),
            "processing_mode": "tier2-only backfill",
            "tier1_reused_from_existing": True,
            "review_backfilled_from_cleaned_json": True,
        }
    )
    return row


def _assert_within_root(root: Path, path: Path):
    root_text = str(root.resolve()).rstrip("\\/").lower()
    path_text = str(path.resolve()).lower()
    if path_text != root_text and not path_text.startswith(root_text + "\\") and not path_text.startswith(root_text + "/"):
        raise RuntimeError(f"Refusing to access path outside project root: {path}")


def _episode_stem_from_cleaned_json(path: Path) -> str:
    suffix = "_cleaned_speaker_transcript.json"
    if not path.name.endswith(suffix):
        raise RuntimeError(f"Unrecognized cleaned transcript filename: {path.name}")
    return path.name[: -len(suffix)]


def _segment_issue_candidates(cleaned_segments: List[Dict[str, object]], reviewed_segments: Optional[List[Dict[str, object]]] = None) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    reviewed_by_id = {
        int(segment["id"]): segment
        for segment in (reviewed_segments or [])
        if isinstance(segment, dict) and segment.get("id") is not None
    }
    for segment in cleaned_segments:
        if not isinstance(segment, dict):
            continue
        segment_id = int(segment.get("id") or 0)
        text = str(segment.get("text") or "")
        lowered = text.lower()
        evidence_ids = [segment_id]
        if ", and we, we " in lowered or ", you know," in lowered or " they're not, they're not " in lowered:
            findings.append(
                {
                    "finding_id": f"deterministic-restart-{segment_id}",
                    "source": "deterministic",
                    "issue_type": "restart_or_disfluency",
                    "severity": "medium",
                    "reason": "This segment still contains a repeated restart or filler pattern.",
                    "segment_ids": evidence_ids,
                    "suggested_text": None,
                }
            )
        reviewed = reviewed_by_id.get(segment_id)
        if reviewed is not None and str(reviewed.get("text") or "") != text:
            findings.append(
                {
                    "finding_id": f"review-diff-{segment_id}",
                    "source": "reviewed_diff",
                    "issue_type": "reviewed_text_differs",
                    "severity": "low",
                    "reason": "Reviewed transcript text differs from the cleaned transcript for this segment.",
                    "segment_ids": evidence_ids,
                    "suggested_text": str(reviewed.get("text") or ""),
                }
            )
    return findings


def _segment_summary(segment: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": int(segment.get("id") or 0),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker": str(segment.get("speaker") or ""),
        "text": str(segment.get("text") or ""),
        "original_text": segment.get("original_text"),
        "llm_reviewed_text": segment.get("llm_reviewed_text"),
    }


def load_project_config(project_root: Path) -> Dict[str, object]:
    config_path = project_root / "podcast_transcribe_config.json"
    example_path = project_root / "examples" / "podcast_transcribe_config.example.json"
    if config_path.exists():
        return _load_json(config_path)
    if example_path.exists():
        return _load_json(example_path)
    return {}


def resolve_workbench_paths(project_root: Path, output_dir: Path) -> Dict[str, Path]:
    config = load_project_config(project_root)
    corrections_dir = _resolve_under_root(
        project_root,
        str(config.get("corrections_dir") or ""),
        DEFAULT_CORRECTIONS_DIRNAME,
    )
    preferred_terms_path = _resolve_under_root(
        project_root,
        str(config.get("preferred_terms_file") or "examples/preferred_terms.txt"),
    )
    replacement_map_path = _resolve_under_root(
        project_root,
        str(config.get("replacement_map_json") or "examples/preferred_replacements.json"),
    )
    workbench_dir = output_dir / WORKBENCH_DIRNAME
    evaluation_pack_path = _resolve_under_root(
        project_root,
        str(config.get("evaluation_pack_path") or ""),
        str(PIPELINE_GOLD_SET_DIR),
    )
    return {
        "corrections_dir": corrections_dir,
        "preferred_terms_path": preferred_terms_path,
        "replacement_map_path": replacement_map_path,
        "workbench_dir": workbench_dir,
        "scan_cache_dir": workbench_dir / SCAN_CACHE_SUBDIR,
        "issue_resolution_dir": workbench_dir / ISSUE_RESOLUTION_SUBDIR,
        "audit_log_path": project_root / AUDIT_LOG_SUBDIR / AUDIT_LOG_FILENAME,
        "evaluation_pack_path": evaluation_pack_path,
    }


def discover_episode_bundles(output_dir: Path) -> List[Dict[str, object]]:
    episodes: List[Dict[str, object]] = []
    for cleaned_path in sorted(output_dir.glob("*_cleaned_speaker_transcript.json")):
        stem = _episode_stem_from_cleaned_json(cleaned_path)
        reviewed_path = output_dir / f"{stem}_reviewed_speaker_transcript.json"
        manifest_path = output_dir / f"{stem}_manifest.json"
        summary = {
            "episode_id": stem,
            "episode_name": stem,
            "cleaned_json_path": str(cleaned_path),
            "reviewed_json_path": str(reviewed_path) if reviewed_path.exists() else "",
            "manifest_path": str(manifest_path) if manifest_path.exists() else "",
            "has_reviewed": reviewed_path.exists(),
        }
        try:
            cleaned_payload = _load_json(cleaned_path)
            errors = validate_transcript_payload(cleaned_payload)
            if errors:
                summary["load_error"] = "; ".join(errors[:5])
            metadata = cleaned_payload.get("metadata") if isinstance(cleaned_payload.get("metadata"), dict) else {}
            summary["episode_date"] = metadata.get("episode_date", "")
            summary["host_detected"] = bool(cleaned_payload.get("host_detected"))
            summary["segment_count"] = len(cleaned_payload.get("segments") or [])
        except Exception as exc:
            summary["load_error"] = str(exc)
        episodes.append(summary)
    return episodes


def load_episode_bundle(project_root: Path, output_dir: Path, episode_id: str) -> Dict[str, object]:
    cleaned_path = output_dir / f"{episode_id}_cleaned_speaker_transcript.json"
    if not cleaned_path.exists():
        raise RuntimeError(f"Episode cleaned transcript not found: {cleaned_path.name}")
    reviewed_path = output_dir / f"{episode_id}_reviewed_speaker_transcript.json"
    manifest_path = output_dir / f"{episode_id}_manifest.json"
    summary_path = output_dir / "_episode_review_summary.csv"
    review_run_report_path = output_dir / "_review_run_report.json"
    speaker_workflow_report_path = output_dir / "_speaker_workflow_report.json"

    cleaned_payload = _load_json(cleaned_path)
    cleaned_errors = validate_transcript_payload(cleaned_payload)
    if cleaned_errors:
        raise RuntimeError(f"Cleaned transcript payload is invalid: {'; '.join(cleaned_errors[:5])}")

    reviewed_payload = None
    if reviewed_path.exists():
        reviewed_payload = _load_json(reviewed_path)
        reviewed_errors = validate_reviewed_transcript_payload(reviewed_payload)
        if reviewed_errors:
            raise RuntimeError(f"Reviewed transcript payload is invalid: {'; '.join(reviewed_errors[:5])}")

    manifest_payload = _load_json(manifest_path) if manifest_path.exists() else {}
    summary_row = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("episode") or "") == f"{episode_id}{cleaned_path.suffix.replace('_cleaned_speaker_transcript.json', '')}":
                    summary_row = row
                    break
                if str(row.get("episode") or "") == f"{episode_id}.mp3" or str(row.get("episode") or "").startswith(episode_id):
                    summary_row = row
                    break

    review_run_report = _load_json(review_run_report_path) if review_run_report_path.exists() else {}
    speaker_workflow_report = _load_json(speaker_workflow_report_path) if speaker_workflow_report_path.exists() else {}
    cleaned_segments = [_segment_summary(segment) for segment in (cleaned_payload.get("segments") or []) if isinstance(segment, dict)]
    reviewed_segments = [_segment_summary(segment) for segment in (reviewed_payload.get("segments") or []) if isinstance(segment, dict)] if reviewed_payload else []
    deterministic_findings = _segment_issue_candidates(cleaned_segments, reviewed_segments)
    paths = resolve_workbench_paths(project_root, output_dir)
    scan_cache_path = paths["scan_cache_dir"] / f"{episode_id}.semantic_scan.json"
    scan_cache = _load_json(scan_cache_path) if scan_cache_path.exists() else {}
    return {
        "episode_id": episode_id,
        "cleaned": {
            "path": str(cleaned_path),
            "source_revision": file_revision(cleaned_path),
            "metadata": cleaned_payload.get("metadata") or {},
            "segments": cleaned_segments,
            "host_detected": bool(cleaned_payload.get("host_detected")),
            "host_original_speaker_id": cleaned_payload.get("host_original_speaker_id"),
            "speaker_mapping": cleaned_payload.get("speaker_mapping") or {},
            "speaker_durations_seconds": cleaned_payload.get("speaker_durations_seconds") or {},
        },
        "reviewed": {
            "present": reviewed_payload is not None,
            "path": str(reviewed_path) if reviewed_payload else "",
            "metadata": (reviewed_payload or {}).get("review_metadata") or {},
            "segments": reviewed_segments,
        },
        "manifest": manifest_payload,
        "summary_row": summary_row,
        "review_run_report": review_run_report,
        "speaker_workflow_report": speaker_workflow_report,
        "deterministic_findings": deterministic_findings,
        "semantic_scan": scan_cache,
        "gold_annotation": load_gold_annotation(project_root, episode_id, output_dir),
    }


def _gold_set_paths(
    project_root: Path,
    episode_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    if output_dir is not None:
        gold_root = resolve_workbench_paths(project_root, output_dir)["evaluation_pack_path"]
    else:
        config = load_project_config(project_root)
        gold_root = _resolve_under_root(
            project_root,
            str(config.get("evaluation_pack_path") or ""),
            str(PIPELINE_GOLD_SET_DIR),
        )
    paths = {
        "root": gold_root,
        "manifest": gold_root / "manifest.json",
        "annotations": gold_root / PIPELINE_GOLD_ANNOTATIONS_DIRNAME,
    }
    if episode_id:
        safe_episode_id = Path(episode_id).name
        paths["annotation"] = paths["annotations"] / f"{safe_episode_id}.reference.json"
    return paths


def load_gold_annotation(project_root: Path, episode_id: str, output_dir: Optional[Path] = None) -> Dict[str, object]:
    annotation_path = _gold_set_paths(project_root, episode_id, output_dir)["annotation"]
    if not annotation_path.exists():
        return {"present": False, "episode_id": episode_id, "path": str(annotation_path), "segments": []}
    payload = _load_json(annotation_path)
    return {
        "present": True,
        "episode_id": episode_id,
        "path": str(annotation_path),
        "segments": payload.get("segments") or [],
        "annotation_metadata": payload.get("annotation_metadata") or {},
    }


def save_gold_segment_annotation(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    segment_id: int,
    reference_text: str,
    reference_speaker: str,
    tags: Optional[List[str]] = None,
    notes: str = "",
    reviewer_id: str = "",
    approval_status: str = "pending_review",
) -> Dict[str, object]:
    cleaned_path = output_dir / f"{episode_id}_cleaned_speaker_transcript.json"
    if not cleaned_path.exists():
        raise RuntimeError(f"Episode cleaned transcript not found: {cleaned_path.name}")
    cleaned_payload = _load_json(cleaned_path)
    errors = validate_transcript_payload(cleaned_payload)
    if errors:
        raise RuntimeError(f"Cannot annotate invalid cleaned transcript: {'; '.join(errors[:5])}")
    paths = _gold_set_paths(project_root, episode_id, output_dir)
    paths["annotations"].mkdir(parents=True, exist_ok=True)
    if paths["annotation"].exists():
        reference_payload = _load_json(paths["annotation"])
    else:
        reference_payload = deepcopy(cleaned_payload)
        reference_payload["text_version"] = "human_reference"
        reference_payload["annotation_metadata"] = {
            "gold_set_version": 1,
            "created_at_epoch_seconds": int(time.time()),
            "source_cleaned_path": str(cleaned_path),
            "annotated_segment_ids": [],
            "tags": [],
            "notes": {},
            "reviewer_id": str(reviewer_id).strip(),
            "approval_status": str(approval_status or "pending_review").strip().lower(),
        }
    target = next(
        (segment for segment in reference_payload.get("segments") or [] if int(segment.get("id", -1)) == int(segment_id)),
        None,
    )
    if not isinstance(target, dict):
        raise RuntimeError(f"Segment {segment_id} was not found in {episode_id}.")
    target["text"] = str(reference_text).strip()
    target["speaker"] = str(reference_speaker).strip() or str(target.get("speaker") or "UNKNOWN")
    metadata = reference_payload.setdefault("annotation_metadata", {})
    annotated_ids = {int(value) for value in metadata.get("annotated_segment_ids") or []}
    annotated_ids.add(int(segment_id))
    metadata["annotated_segment_ids"] = sorted(annotated_ids)
    metadata["updated_at_epoch_seconds"] = int(time.time())
    metadata["tags"] = sorted({str(item).strip() for item in (metadata.get("tags") or []) + list(tags or []) if str(item).strip()})
    notes_payload = metadata.get("notes") if isinstance(metadata.get("notes"), dict) else {}
    if notes.strip():
        notes_payload[str(segment_id)] = notes.strip()
    metadata["notes"] = notes_payload
    if reviewer_id.strip():
        metadata["reviewer_id"] = reviewer_id.strip()
    metadata["approval_status"] = str(approval_status or metadata.get("approval_status") or "pending_review").strip().lower()
    metadata["version"] = int(metadata.get("version") or 1)
    history = metadata.get("adjudication_history") if isinstance(metadata.get("adjudication_history"), list) else []
    history.append(
        {
            "timestamp_epoch_seconds": int(time.time()),
            "segment_id": int(segment_id),
            "reviewer_id": str(reviewer_id).strip(),
            "approval_status": metadata["approval_status"],
            "reference_text": str(reference_text).strip(),
            "reference_speaker": str(reference_speaker).strip(),
            "tags": sorted({str(item).strip() for item in tags or [] if str(item).strip()}),
            "notes": notes.strip(),
        }
    )
    metadata["adjudication_history"] = history
    paths["annotation"].write_text(json.dumps(reference_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if paths["manifest"].exists():
        manifest = _load_json(paths["manifest"])
    else:
        manifest = {"gold_set_version": 1, "name": "Podcast Pipeline Gold Set", "entries": []}
    entries = manifest.setdefault("entries", [])
    entry = next((item for item in entries if isinstance(item, dict) and str(item.get("id")) == episode_id), None)
    relative_reference = str(paths["annotation"].relative_to(paths["root"])).replace("\\", "/")
    if entry is None:
        entry = {"id": episode_id, "audio_stem": episode_id, "reference": relative_reference, "enabled": True}
        entries.append(entry)
    entry["reference"] = relative_reference
    entry["tags"] = metadata["tags"]
    entry["segment_ids"] = metadata["annotated_segment_ids"]
    entry["reviewer_id"] = metadata.get("reviewer_id", "")
    entry["approval_status"] = metadata.get("approval_status", "pending_review")
    entry["condition_tags"] = metadata["tags"]
    manifest["entries"] = sorted(entries, key=lambda item: str(item.get("id") or ""))
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _append_audit_log(
        project_root,
        output_dir,
        {
            "timestamp_epoch_seconds": int(time.time()),
            "action": "gold_segment_annotation_saved",
            "episode_id": episode_id,
            "segment_id": int(segment_id),
            "target_path": str(paths["annotation"]),
        },
    )
    return load_gold_annotation(project_root, episode_id, output_dir)


def evaluation_queues(project_root: Path, output_dir: Path) -> Dict[str, object]:
    paths = _gold_set_paths(project_root, output_dir=output_dir)
    manifest = _load_json(paths["manifest"]) if paths["manifest"].exists() else {
        "gold_set_version": 2,
        "name": "Private Podcast Evaluation Pack",
        "entries": [],
    }
    entries_by_id = {
        str(item.get("id") or ""): item
        for item in manifest.get("entries") or []
        if isinstance(item, dict)
    }
    queues: Dict[str, List[Dict[str, object]]] = {
        "unlabelled": [],
        "pending_review": [],
        "adjudication_required": [],
        "human_approved": [],
    }
    for episode in discover_episode_bundles(output_dir):
        episode_id = str(episode["episode_id"])
        entry = entries_by_id.get(episode_id)
        if entry is None:
            queues["unlabelled"].append(episode)
            continue
        status = str(entry.get("approval_status") or "pending_review").lower()
        queue = "human_approved" if status in {"approved", "human_approved"} else status
        if queue not in queues:
            queue = "pending_review"
        queues[queue].append({**episode, "evaluation_entry": entry})
    return {
        "evaluation_pack_path": str(paths["root"]),
        "manifest_path": str(paths["manifest"]),
        "queues": queues,
        "counts": {key: len(value) for key, value in queues.items()},
    }


def propose_quality_campaign(project_root: Path, output_dir: Path) -> Dict[str, object]:
    episodes = discover_episode_bundles(output_dir)
    enriched = []
    for episode in episodes:
        cleaned = _load_json(Path(str(episode["cleaned_json_path"])))
        transcription = cleaned.get("transcription") if isinstance(cleaned.get("transcription"), dict) else {}
        duration = float(transcription.get("duration") or 0.0)
        enriched.append({**episode, "duration_seconds": duration})
    ordered = sorted(enriched, key=lambda item: (float(item["duration_seconds"]), str(item["episode_id"])))
    if not ordered:
        return {"target_count": 12, "selected": [], "requirements": {}}
    short = ordered[:3]
    long = ordered[-3:] if len(ordered) > 3 else []
    used = {str(item["episode_id"]) for item in short + long}
    middle_pool = [item for item in ordered if str(item["episode_id"]) not in used]
    if len(middle_pool) <= 6:
        typical = middle_pool
    else:
        step = (len(middle_pool) - 1) / 5
        typical = [middle_pool[round(index * step)] for index in range(6)]
    selected = short + typical + long
    return {
        "campaign_version": 1,
        "target_count": 12,
        "selected": [
            {
                **item,
                "duration_class": "short" if item in short else "long" if item in long else "typical",
                "required_operator_tags": [],
            }
            for item in selected[:12]
        ],
        "requirements": {
            "duration_mix": {"short": 3, "typical": 6, "long": 3},
            "minimum_condition_counts": {
                "crosstalk": 2,
                "recurring_guest_or_cohost": 2,
                "noise_or_music": 2,
            },
            "human_approval_required": True,
        },
    }


def initialize_quality_campaign(project_root: Path, output_dir: Path) -> Dict[str, object]:
    proposal = propose_quality_campaign(project_root, output_dir)
    if len(proposal.get("selected") or []) < 12:
        raise RuntimeError("At least 12 processed episodes are required to initialize the guided campaign.")
    paths = _gold_set_paths(project_root, output_dir=output_dir)
    manifest = _load_json(paths["manifest"]) if paths["manifest"].exists() else {
        "gold_set_version": 2,
        "name": "Private Podcast Evaluation Pack",
        "entries": [],
    }
    manifest["gold_set_version"] = 2
    manifest["campaign_selection"] = {
        "campaign_version": 1,
        "created_at_epoch_seconds": int(time.time()),
        "requirements": proposal["requirements"],
        "episodes": [
            {
                "id": item["episode_id"],
                "duration_class": item["duration_class"],
                "duration_seconds": item["duration_seconds"],
            }
            for item in proposal["selected"]
        ],
    }
    entries = {
        str(item.get("id") or ""): item
        for item in manifest.get("entries") or []
        if isinstance(item, dict)
    }
    for item in proposal["selected"]:
        episode_id = str(item["episode_id"])
        entry = entries.setdefault(
            episode_id,
            {
                "id": episode_id,
                "audio_stem": episode_id,
                "reference": f"annotations/{episode_id}.reference.json",
                "enabled": True,
                "approval_status": "pending_review",
                "segment_ids": [],
                "tags": [],
            },
        )
        entry["duration_class"] = item["duration_class"]
    manifest["entries"] = sorted(entries.values(), key=lambda item: str(item.get("id") or ""))
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "initialized", "manifest_path": str(paths["manifest"]), **proposal}


def accept_evaluation_baseline(project_root: Path, output_dir: Path, reviewer_id: str) -> Dict[str, object]:
    queues = evaluation_queues(project_root, output_dir)
    approved = int(queues["counts"]["human_approved"])
    if approved < 12:
        raise RuntimeError(f"The guided evaluation campaign requires 12 human-approved episodes; found {approved}.")
    pack_root = Path(str(queues["evaluation_pack_path"]))
    manifest = _load_json(pack_root / "manifest.json")
    approved_entries = [
        item for item in manifest.get("entries") or []
        if isinstance(item, dict)
        and str(item.get("approval_status") or "").lower() in {"approved", "human_approved"}
    ]
    duration_counts = {
        name: sum(str(item.get("duration_class") or "") == name for item in approved_entries)
        for name in ("short", "typical", "long")
    }
    if duration_counts != {"short": 3, "typical": 6, "long": 3}:
        raise RuntimeError(f"Approved campaign duration mix is incomplete: {duration_counts}")
    tags = [[str(tag).lower() for tag in item.get("tags") or []] for item in approved_entries]
    condition_counts = {
        "crosstalk": sum("crosstalk" in item for item in tags),
        "recurring_guest_or_cohost": sum(
            bool({"recurring_guest", "cohost", "co-host", "recurring_guest_or_cohost"} & set(item))
            for item in tags
        ),
        "noise_or_music": sum(bool({"noise", "music", "noise_or_music"} & set(item)) for item in tags),
    }
    shortfalls = [name for name, count in condition_counts.items() if count < 2]
    if shortfalls:
        raise RuntimeError("Approved campaign is missing condition coverage: " + ", ".join(shortfalls))
    report_path = output_dir / "pipeline_quality_benchmark_report.json"
    if not report_path.exists():
        raise RuntimeError("Run the pipeline benchmark before accepting a baseline.")
    report = _load_json(report_path)
    baseline_dir = pack_root / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    target = baseline_dir / "accepted_baseline.json"
    if target.exists():
        raise RuntimeError("An accepted baseline already exists. Baseline replacement requires a separate adjudicated promotion.")
    payload = {
        "baseline_contract_version": 1,
        "accepted_at_epoch_seconds": int(time.time()),
        "reviewer_id": str(reviewer_id).strip() or "anonymous",
        "evaluation_pack_path": str(pack_root),
        "episode_contract_versions": sorted(
            {
                str(item.get("episode_contract_version") or "legacy")
                for item in report.get("results") or []
                if isinstance(item, dict)
            }
        ),
        "report": report,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "evaluation_baseline_accepted",
            "reviewer_id": payload["reviewer_id"],
            "target_path": str(target),
        },
    )
    return {"status": "accepted", "path": str(target)}


def _workbench_scan_system_prompt() -> str:
    return (
        "You are reviewing cleaned podcast transcript segments for likely transcription problems. "
        "Return strict JSON with a top-level 'findings' array. "
        "Each finding must include finding_id, issue_type, severity, reason, segment_ids, and may include suggested_text. "
        "Do not rewrite the whole transcript. Only flag likely problems. "
        "Keep findings compact and evidence-based."
    )


def _openai_compatible_request(
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1600,
    *,
    backend_name: str = "",
    reasoning_effort: str = "none",
) -> Dict[str, object]:
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request_controls = _chat_template_request_controls(backend_name, model_name, reasoning_effort)
    if request_controls is not None:
        payload["chat_template_kwargs"] = request_controls

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Semantic scan backend HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Semantic scan backend connection failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Semantic scan backend returned a non-object response.")
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Semantic scan backend returned no choices.")
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
    content = str(content or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Semantic scan backend returned a non-object JSON payload.")
    return parsed


def run_semantic_scan(project_root: Path, output_dir: Path, episode_id: str, force: bool = False) -> Dict[str, object]:
    paths = resolve_workbench_paths(project_root, output_dir)
    cache_path = paths["scan_cache_dir"] / f"{episode_id}.semantic_scan.json"
    if cache_path.exists() and not force:
        return _load_json(cache_path)

    config = load_project_config(project_root)
    resolved = resolve_review_runtime_config(config)
    if not resolved.get("backend_ready"):
        raise RuntimeError("Semantic scan requires a configured review backend and model.")
    bundle = load_episode_bundle(project_root, output_dir, episode_id)
    segments = bundle["cleaned"]["segments"]
    prompt_payload = {
        "episode_id": episode_id,
        "segments": [
            {
                "id": segment["id"],
                "speaker": segment["speaker"],
                "text": segment["text"],
            }
            for segment in segments
        ],
        "reviewed_differences_present": bundle["reviewed"]["present"],
    }
    parsed = _openai_compatible_request(
        str(resolved.get("review_base_url") or ""),
        str(resolved.get("review_model_name") or ""),
        _workbench_scan_system_prompt(),
        json.dumps(prompt_payload, ensure_ascii=True),
        backend_name=str(resolved.get("effective_backend") or resolved.get("backend") or ""),
        reasoning_effort=str(resolved.get("review_reasoning_effort") or "none"),
    )
    findings = parsed.get("findings")
    if not isinstance(findings, list):
        raise RuntimeError("Semantic scan payload is missing a findings list.")
    result = {
        "episode_id": episode_id,
        "generated_at_epoch_ms": int(time.time() * 1000),
        "backend": str(resolved.get("backend") or ""),
        "review_model_name": str(resolved.get("review_model_name") or ""),
        "finding_count": len(findings),
        "findings": [
            {
                "finding_id": str(item.get("finding_id") or f"finding-{index+1}"),
                "issue_type": str(item.get("issue_type") or "other"),
                "severity": str(item.get("severity") or "medium"),
                "reason": str(item.get("reason") or ""),
                "segment_ids": [int(value) for value in (item.get("segment_ids") or []) if str(value).strip()],
                "suggested_text": item.get("suggested_text"),
                "source": "semantic_scan",
            }
            for index, item in enumerate(findings)
            if isinstance(item, dict)
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _teach_me_session_path(output_dir: Path, episode_id: str, session_id: str) -> Path:
    paths = resolve_workbench_paths(output_dir.parent if output_dir.name == "output" else output_dir, output_dir)
    return paths["workbench_dir"] / TEACH_ME_SUBDIR / episode_id / f"{session_id}.json"


def _teach_me_workbench_dir(output_dir: Path) -> Path:
    return output_dir / WORKBENCH_DIRNAME / TEACH_ME_SUBDIR


def _review_rule_prompt() -> str:
    return (
        "You are extracting a reusable, narrow review rule from a single operator-approved transcript edit. "
        "Return strict JSON. Do not write code. Do not propose broad paraphrase rules. "
        "Keep the rule local, conservative, and suitable for an LLM transcript review stage. "
        "Never mutate deterministic cleanup behavior. Never violate protected preferred terms. "
        "Only choose from these rule families: cleanup_preference, glossary_naming_preference, "
        "speaker_label_preference, style_phrasing_preference, do_not_change_constraint. "
        "Only choose from these stage targets: transcript_cleanup_review, glossary_correction_review, "
        "speaker_consistency_review, episode_qa_review."
    )


def _choose_rule_family(candidate: Dict[str, object]) -> str:
    family = str(candidate.get("rule_family") or "").strip()
    return family if family in LEARNED_RULE_ALLOWED_FAMILIES else "style_phrasing_preference"


def _choose_stage_target(candidate: Dict[str, object]) -> str:
    stage_target = str(candidate.get("stage_target") or "").strip()
    return stage_target if stage_target in LEARNED_RULE_ALLOWED_STAGES else "transcript_cleanup_review"


def _load_teach_me_controls() -> List[Dict[str, object]]:
    if not TEACH_ME_CONTROL_FIXTURE_PATH.exists():
        return []
    payload = _load_json(TEACH_ME_CONTROL_FIXTURE_PATH)
    fixtures = payload.get("fixtures")
    return [item for item in fixtures if isinstance(item, dict)] if isinstance(fixtures, list) else []


def _teaching_source_example(bundle: Dict[str, object], segment_id: int, desired_reviewed_text: str) -> Dict[str, object]:
    cleaned_segment = next((segment for segment in bundle["cleaned"]["segments"] if int(segment["id"]) == int(segment_id)), None)
    if cleaned_segment is None:
        raise RuntimeError(f"Segment {segment_id} was not found in cleaned transcript data.")
    reviewed_segment = next((segment for segment in bundle["reviewed"]["segments"] if int(segment["id"]) == int(segment_id)), None)
    return {
        "segment_id": int(segment_id),
        "speaker": str(cleaned_segment.get("speaker") or ""),
        "cleaned_text": str(cleaned_segment.get("text") or ""),
        "reviewed_text": str((reviewed_segment or {}).get("text") or cleaned_segment.get("text") or ""),
        "desired_reviewed_text": str(desired_reviewed_text).strip(),
        "start": cleaned_segment.get("start"),
        "end": cleaned_segment.get("end"),
    }


def _nearby_example_segments(bundle: Dict[str, object], segment_id: int) -> List[Dict[str, object]]:
    cleaned_segments = list(bundle["cleaned"]["segments"])
    indexes = [index for index, segment in enumerate(cleaned_segments) if int(segment["id"]) == int(segment_id)]
    if not indexes:
        return []
    index = indexes[0]
    examples: List[Dict[str, object]] = []
    for offset in (-1, 1):
        nearby_index = index + offset
        if 0 <= nearby_index < len(cleaned_segments):
            segment = cleaned_segments[nearby_index]
            examples.append(
                {
                    "segment_id": int(segment["id"]),
                    "speaker": str(segment.get("speaker") or ""),
                    "cleaned_text": str(segment.get("text") or ""),
                    "expected_text": str(segment.get("text") or ""),
                    "expected_changed": False,
                    "source": "nearby_episode_control",
                }
            )
    return examples


def _induce_learned_rule_candidate(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    source_example: Dict[str, object],
    supersedes_rule_id: str = "",
) -> Dict[str, object]:
    config = load_project_config(project_root)
    runtime_review = resolve_review_runtime_config(config)
    if not runtime_review.get("backend_ready"):
        raise RuntimeError("Teach Me requires a configured review backend and model.")
    existing_rules = load_review_rule_library(project_root)["rules"]
    prompt_payload = {
        "episode_id": episode_id,
        "source_example": source_example,
        "existing_rules": [
            {
                "rule_id": rule.get("rule_id"),
                "rule_family": rule.get("rule_family"),
                "stage_target": rule.get("stage_target"),
                "summary": rule.get("summary"),
                "status": rule.get("status"),
            }
            for rule in existing_rules
        ],
        "preferred_terms": _read_preferred_terms(resolve_workbench_paths(project_root, output_dir)["preferred_terms_path"]),
        "replacement_map": _read_replacement_map(resolve_workbench_paths(project_root, output_dir)["replacement_map_path"]),
        "review_stage_flags": {
            "transcript_cleanup_review": bool(runtime_review.get("transcript_cleanup_review")),
            "glossary_correction_review": bool(runtime_review.get("glossary_correction_review")),
            "speaker_consistency_review": bool(runtime_review.get("speaker_consistency_review")),
            "episode_qa_review": bool(runtime_review.get("episode_qa_review")),
        },
        "supersedes_rule_id": supersedes_rule_id,
    }
    parsed = _openai_compatible_request(
        str(runtime_review.get("review_base_url") or ""),
        str(runtime_review.get("review_model_name") or ""),
        _review_rule_prompt(),
        json.dumps(prompt_payload, ensure_ascii=True),
        max_tokens=1800,
        backend_name=str(runtime_review.get("effective_backend") or runtime_review.get("backend") or ""),
        reasoning_effort=str(runtime_review.get("review_reasoning_effort") or "none"),
    )
    candidate = parsed.get("rule_candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("Teach Me rule induction did not return a rule_candidate object.")
    family = _choose_rule_family(candidate)
    stage_target = _choose_stage_target(candidate)
    instruction_payload = candidate.get("instruction_payload") if isinstance(candidate.get("instruction_payload"), dict) else {}
    return {
        "rule_id": f"rule_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "status": "draft",
        "activation_status": "pending_approval",
        "rule_family": family,
        "stage_target": stage_target,
        "summary": str(candidate.get("summary") or "").strip(),
        "explanation": str(candidate.get("explanation") or "").strip(),
        "instruction_payload": {
            "directive": str(instruction_payload.get("directive") or "").strip(),
            "avoid": [str(item).strip() for item in (instruction_payload.get("avoid") or []) if str(item).strip()],
            "positive_examples": [item for item in (instruction_payload.get("positive_examples") or []) if isinstance(item, dict)],
            "negative_examples": [item for item in (instruction_payload.get("negative_examples") or []) if isinstance(item, dict)],
        },
        "confidence": float(candidate.get("confidence") or 0.0),
        "ambiguity_notes": [str(item).strip() for item in (candidate.get("ambiguity_notes") or []) if str(item).strip()],
        "source_examples": [source_example],
        "validation": {},
        "activation_scope": "project_review_layer",
        "supersedes_rule_id": supersedes_rule_id,
        "superseded_by_rule_id": "",
        "provenance": {
            "created_at_epoch_ms": int(time.time() * 1000),
            "updated_at_epoch_ms": int(time.time() * 1000),
            "backend": str(runtime_review.get("backend") or ""),
            "review_model_name": str(runtime_review.get("review_model_name") or ""),
            "validation_evidence": {},
        },
        "audit": {
            "approvals": [],
            "reruns": [],
            "backfills": [],
        },
    }


def _rule_runtime_config(config: Dict[str, object], stage_target: str) -> Dict[str, object]:
    payload = dict(config)
    payload["transcript_cleanup_review"] = stage_target == "transcript_cleanup_review"
    payload["glossary_correction_review"] = stage_target == "glossary_correction_review"
    payload["speaker_consistency_review"] = stage_target == "speaker_consistency_review"
    payload["episode_qa_review"] = stage_target == "episode_qa_review"
    return payload


def _segment_item_from_text(segment_id: int, speaker: str, text: str) -> SegmentItem:
    return SegmentItem(
        id=segment_id,
        start=float(segment_id),
        end=float(segment_id) + 1.0,
        text=text,
        speaker=speaker,
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        words=[WordItem(start=float(segment_id), end=float(segment_id) + 0.1, word=(text.split() or [""])[0], speaker=speaker)],
        original_text=text,
        cleanup_applied=False,
        cleanup_level="normal",
        manual_correction_applied=False,
        original_speaker=speaker,
    )


def _run_rule_candidate_once(
    config: Dict[str, object],
    stage_target: str,
    segments: List[SegmentItem],
    rule_candidate: Dict[str, object],
) -> Dict[str, object]:
    runtime_config = _rule_runtime_config(config, stage_target)
    return review_segments(
        segments,
        runtime_config,
        review_input_source="teach_me_validation",
        learned_rules=[{**rule_candidate, "status": "approved"}],
    )


def _extract_result_text(review_result: Dict[str, object], segment_id: int, fallback_text: str) -> str:
    for segment in review_result.get("segments") or []:
        if int(getattr(segment, "id", 0)) == int(segment_id):
            return str(getattr(segment, "text", fallback_text) or fallback_text)
    return str(fallback_text)


def _validation_feedback(rule_candidate: Dict[str, object], validation: Dict[str, object]) -> str:
    failures = []
    taught = validation.get("taught_example") if isinstance(validation.get("taught_example"), dict) else {}
    if taught and not taught.get("exact_match"):
        failures.append("taught example did not match desired reviewed text exactly")
    controls = validation.get("controls") if isinstance(validation.get("controls"), list) else []
    over_edits = [item.get("fixture_id") for item in controls if isinstance(item, dict) and not item.get("passed")]
    if over_edits:
        failures.append(f"control failures: {', '.join(str(item) for item in over_edits)}")
    if not failures:
        return ""
    return (
        "Refine the rule candidate conservatively so it matches the taught example and avoids overreach. "
        f"Current failures: {'; '.join(failures)}."
    )


def _refine_rule_candidate(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    source_example: Dict[str, object],
    current_candidate: Dict[str, object],
    validation: Dict[str, object],
) -> Dict[str, object]:
    config = load_project_config(project_root)
    runtime_review = resolve_review_runtime_config(config)
    feedback = _validation_feedback(current_candidate, validation)
    if not feedback:
        return current_candidate
    prompt_payload = {
        "episode_id": episode_id,
        "source_example": source_example,
        "current_candidate": current_candidate,
        "validation": validation,
        "feedback": feedback,
    }
    parsed = _openai_compatible_request(
        str(runtime_review.get("review_base_url") or ""),
        str(runtime_review.get("review_model_name") or ""),
        _review_rule_prompt(),
        json.dumps(prompt_payload, ensure_ascii=True),
        max_tokens=1600,
        backend_name=str(runtime_review.get("effective_backend") or runtime_review.get("backend") or ""),
        reasoning_effort=str(runtime_review.get("review_reasoning_effort") or "none"),
    )
    candidate = parsed.get("rule_candidate")
    if not isinstance(candidate, dict):
        return current_candidate
    current_candidate["rule_family"] = _choose_rule_family(candidate)
    current_candidate["stage_target"] = _choose_stage_target(candidate)
    current_candidate["summary"] = str(candidate.get("summary") or current_candidate.get("summary") or "").strip()
    current_candidate["explanation"] = str(candidate.get("explanation") or current_candidate.get("explanation") or "").strip()
    instruction_payload = candidate.get("instruction_payload") if isinstance(candidate.get("instruction_payload"), dict) else {}
    current_candidate["instruction_payload"] = {
        "directive": str(instruction_payload.get("directive") or current_candidate.get("instruction_payload", {}).get("directive") or "").strip(),
        "avoid": [str(item).strip() for item in (instruction_payload.get("avoid") or []) if str(item).strip()],
        "positive_examples": [item for item in (instruction_payload.get("positive_examples") or []) if isinstance(item, dict)],
        "negative_examples": [item for item in (instruction_payload.get("negative_examples") or []) if isinstance(item, dict)],
    }
    current_candidate["confidence"] = float(candidate.get("confidence") or current_candidate.get("confidence") or 0.0)
    current_candidate["ambiguity_notes"] = [str(item).strip() for item in (candidate.get("ambiguity_notes") or []) if str(item).strip()]
    current_candidate["provenance"]["updated_at_epoch_ms"] = int(time.time() * 1000)
    return current_candidate


def validate_teach_me_rule_candidate(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    source_example: Dict[str, object],
    rule_candidate: Dict[str, object],
) -> Dict[str, object]:
    config = load_project_config(project_root)
    stage_target = str(rule_candidate.get("stage_target") or "transcript_cleanup_review")
    validation = {
        "stage_target": stage_target,
        "taught_example": {},
        "nearby_examples": [],
        "controls": [],
        "pass": False,
        "warnings": [],
        "refinement_iterations": 0,
    }
    bundle = load_episode_bundle(project_root, output_dir, episode_id)
    nearby_examples = _nearby_example_segments(bundle, int(source_example["segment_id"]))
    controls = [fixture for fixture in _load_teach_me_controls() if str(fixture.get("stage_target") or "") in {"", stage_target}]

    current_candidate = normalize_learned_rule(rule_candidate)
    for attempt in range(3):
        taught_result = _run_rule_candidate_once(
            config,
            stage_target,
            [_segment_item_from_text(int(source_example["segment_id"]), str(source_example["speaker"]), str(source_example["cleaned_text"]))],
            current_candidate,
        )
        taught_text = _extract_result_text(taught_result, int(source_example["segment_id"]), str(source_example["cleaned_text"]))
        validation["taught_example"] = {
            "expected_text": str(source_example["desired_reviewed_text"]),
            "produced_text": taught_text,
            "exact_match": taught_text == str(source_example["desired_reviewed_text"]),
        }

        validation["nearby_examples"] = []
        for example in nearby_examples:
            result = _run_rule_candidate_once(
                config,
                stage_target,
                [_segment_item_from_text(int(example["segment_id"]), str(example["speaker"]), str(example["cleaned_text"]))],
                current_candidate,
            )
            produced = _extract_result_text(result, int(example["segment_id"]), str(example["cleaned_text"]))
            validation["nearby_examples"].append(
                {
                    "segment_id": int(example["segment_id"]),
                    "expected_text": str(example["expected_text"]),
                    "produced_text": produced,
                    "passed": produced == str(example["expected_text"]),
                }
            )

        validation["controls"] = []
        for index, fixture in enumerate(controls, start=1):
            result = _run_rule_candidate_once(
                config,
                stage_target,
                [_segment_item_from_text(index, str(fixture.get("speaker") or "HOST"), str(fixture.get("input_text") or ""))],
                current_candidate,
            )
            produced = _extract_result_text(result, index, str(fixture.get("input_text") or ""))
            expected = str(fixture.get("expected_text") or fixture.get("input_text") or "")
            validation["controls"].append(
                {
                    "fixture_id": str(fixture.get("fixture_id") or f"fixture_{index}"),
                    "expected_text": expected,
                    "produced_text": produced,
                    "passed": produced == expected,
                }
            )

        validation["pass"] = bool(validation["taught_example"].get("exact_match")) and all(
            bool(item.get("passed")) for item in validation["controls"]
        )
        if validation["pass"] or attempt >= 2:
            break
        validation["refinement_iterations"] = attempt + 1
        current_candidate = _refine_rule_candidate(project_root, output_dir, episode_id, source_example, current_candidate, validation)

    warnings = []
    if not validation["taught_example"].get("exact_match"):
        warnings.append("taught_example_mismatch")
    warnings.extend(
        f"control_failure:{item['fixture_id']}"
        for item in validation["controls"]
        if not item.get("passed")
    )
    validation["warnings"] = warnings
    validation["refinement_iterations"] = int(validation["refinement_iterations"])
    current_candidate["validation"] = validation
    current_candidate["provenance"]["validation_evidence"] = validation
    return current_candidate


def _teach_me_session_artifact_path(output_dir: Path, episode_id: str, session_id: str) -> Path:
    return _teach_me_workbench_dir(output_dir) / episode_id / f"{session_id}.json"


def propose_teach_me_rule(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    segment_id: int,
    desired_reviewed_text: str,
    supersedes_rule_id: str = "",
) -> Dict[str, object]:
    bundle = load_episode_bundle(project_root, output_dir, episode_id)
    desired = str(desired_reviewed_text).strip()
    if not desired:
        raise RuntimeError("Desired reviewed text cannot be blank.")
    source_example = _teaching_source_example(bundle, segment_id, desired)
    candidate = _induce_learned_rule_candidate(project_root, output_dir, episode_id, source_example, supersedes_rule_id=supersedes_rule_id)
    candidate = validate_teach_me_rule_candidate(project_root, output_dir, episode_id, source_example, candidate)
    session_id = f"teach_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    artifact = {
        "session_id": session_id,
        "episode_id": episode_id,
        "segment_id": int(segment_id),
        "source_example": source_example,
        "rule_candidate": candidate,
    }
    artifact_path = _teach_me_session_artifact_path(output_dir, episode_id, session_id)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    upsert_review_rule(project_root, candidate)
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "teach_me_rule_proposed",
            "episode_id": episode_id,
            "segment_id": int(segment_id),
            "session_id": session_id,
            "rule_id": candidate["rule_id"],
            "target_path": str(artifact_path),
        },
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "episode_id": episode_id,
        "segment_id": int(segment_id),
        "source_example": source_example,
        "rule_candidate": candidate,
    }


def _update_summary_for_episode(output_dir: Path, summary_row: Dict[str, object]):
    cli = _cli_helpers()
    state = _state_helpers()
    summary_path = output_dir / state["SUMMARY_FILENAME"]
    existing_rows = state["load_episode_summary_rows"](summary_path, cli["normalize_episode_summary_row"])
    existing_rows[str(summary_row.get("episode") or "")] = cli["normalize_episode_summary_row"](summary_row)
    cli["write_episode_summary_csv"](summary_path, list(existing_rows.values()))
    cli["write_run_reports"](output_dir, list(existing_rows.values()))


def rerun_review_with_approved_rules(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    focus_rule_id: str = "",
) -> Dict[str, object]:
    approved_rules = approved_review_rules(project_root)
    if focus_rule_id and not any(str(rule.get("rule_id") or "") == str(focus_rule_id) for rule in approved_rules):
        raise RuntimeError(f"Approved learned rule not found: {focus_rule_id}")
    cleaned_path = output_dir / f"{episode_id}_cleaned_speaker_transcript.json"
    cleaned_payload = _load_json(cleaned_path)
    cli = _cli_helpers()
    cleaned_segments = cli["segment_items_from_cleaned_payload"](cleaned_payload)
    config = load_project_config(project_root)
    review_result = review_segments(
        cleaned_segments,
        config,
        review_input_source="cleaned_json_backfill",
        learned_rules=approved_rules,
    )
    review_metadata = review_result["metadata"]
    approved_rule_ids = [str(rule.get("rule_id") or "") for rule in approved_rules if str(rule.get("rule_id") or "")]
    review_metadata["active_learned_rule_ids"] = approved_rule_ids
    contributing_rule_ids = [
        rule_id
        for rule_id in (review_metadata.get("contributing_learned_rule_ids") or [])
        if str(rule_id) in approved_rule_ids
    ]
    if focus_rule_id and focus_rule_id in approved_rule_ids and focus_rule_id not in contributing_rule_ids:
        contributing_rule_ids.append(focus_rule_id)
    review_metadata["contributing_learned_rule_ids"] = contributing_rule_ids
    episode_metadata = cleaned_payload.get("metadata") if isinstance(cleaned_payload.get("metadata"), dict) else {}
    source_file = str(cleaned_payload.get("source_file") or f"{episode_id}.mp3")
    audio_path = Path(source_file)
    host_speaker = cleaned_payload.get("host_original_speaker_id")
    speaker_mapping = {
        str(key): str(value)
        for key, value in (cleaned_payload.get("speaker_mapping") or {}).items()
        if value not in ("", None)
    }
    resolved_host_label = speaker_mapping.get(str(host_speaker), "HOST") if host_speaker else "HOST"
    host_output_labels = {resolved_host_label, "HOST"}
    durations = {
        str(key): float(value)
        for key, value in (cleaned_payload.get("speaker_durations_seconds") or {}).items()
        if value not in ("", None)
    }
    known_assignments = {
        str(key): value
        for key, value in (cleaned_payload.get("known_speaker_assignments") or {}).items()
        if isinstance(value, dict)
    }
    diarized_turns = [turn for turn in (cleaned_payload.get("diarization_turns") or []) if isinstance(turn, dict)]
    info_payload = cleaned_payload.get("transcription") if isinstance(cleaned_payload.get("transcription"), dict) else {}
    reviewed_paths = cli["write_reviewed_output_bundle"](
        audio_path=audio_path,
        output_dir=output_dir,
        reviewed_segments=review_result["segments"],
        review_metadata=review_metadata,
        host_output_labels=host_output_labels,
        episode_metadata=episode_metadata,
        info_payload=info_payload,
        diarized_turns=diarized_turns,
        speaker_mapping=speaker_mapping,
        host_speaker=str(host_speaker) if host_speaker not in ("", None) else None,
        durations=durations,
        known_assignments=known_assignments,
        runtime_config=config,
    )
    summary_row = cli["build_review_backfill_summary_row"](
        audio_path=audio_path,
        cleaned_payload=cleaned_payload,
        cleaned_segments=cleaned_segments,
        review_result=review_result,
        processing_seconds=0.0,
    )
    _update_summary_for_episode(output_dir, summary_row)
    if focus_rule_id:
        rule = get_review_rule(project_root, focus_rule_id)
        if rule:
            reruns = rule.get("audit", {}).get("reruns") if isinstance(rule.get("audit"), dict) else []
            reruns = list(reruns or [])
            reruns.append({"episode_id": episode_id, "created_at_epoch_ms": int(time.time() * 1000)})
            rule["audit"]["reruns"] = reruns
            rule["provenance"]["updated_at_epoch_ms"] = int(time.time() * 1000)
            upsert_review_rule(project_root, rule)
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "teach_me_current_episode_rerun",
            "episode_id": episode_id,
            "rule_id": focus_rule_id,
            "reviewed_output_paths": [str(path) for path in reviewed_paths],
        },
    )
    return {
        "status": "ok",
        "episode_id": episode_id,
        "reviewed_output_paths": [str(path) for path in reviewed_paths],
        "review_metadata": review_metadata,
        "summary_row": summary_row,
    }


def approve_review_rule(project_root: Path, output_dir: Path, rule_id: str, episode_id: str) -> Dict[str, object]:
    rule = get_review_rule(project_root, rule_id)
    if rule is None:
        raise RuntimeError(f"Learned rule not found: {rule_id}")
    if rule.get("supersedes_rule_id"):
        superseded = get_review_rule(project_root, str(rule.get("supersedes_rule_id") or ""))
        if superseded:
            superseded["status"] = "superseded"
            superseded["superseded_by_rule_id"] = rule_id
            superseded["provenance"]["updated_at_epoch_ms"] = int(time.time() * 1000)
            upsert_review_rule(project_root, superseded)
    rule["status"] = "approved"
    rule["activation_status"] = "approved"
    approvals = rule.get("audit", {}).get("approvals") if isinstance(rule.get("audit"), dict) else []
    approvals = list(approvals or [])
    approvals.append({"episode_id": episode_id, "created_at_epoch_ms": int(time.time() * 1000)})
    rule["audit"]["approvals"] = approvals
    rule["provenance"]["updated_at_epoch_ms"] = int(time.time() * 1000)
    upsert_review_rule(project_root, rule)
    rerun_result = rerun_review_with_approved_rules(project_root, output_dir, episode_id, focus_rule_id=rule_id)
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "teach_me_rule_approved",
            "episode_id": episode_id,
            "rule_id": rule_id,
        },
    )
    return {
        "status": "ok",
        "rule": get_review_rule(project_root, rule_id),
        "rerun": rerun_result,
    }


def reject_review_rule(project_root: Path, output_dir: Path, rule_id: str) -> Dict[str, object]:
    rule = get_review_rule(project_root, rule_id)
    if rule is None:
        raise RuntimeError(f"Learned rule not found: {rule_id}")
    rule["status"] = "disabled"
    rule["activation_status"] = "rejected"
    rule["provenance"]["updated_at_epoch_ms"] = int(time.time() * 1000)
    upsert_review_rule(project_root, rule)
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "teach_me_rule_rejected",
            "rule_id": rule_id,
        },
    )
    return {"status": "ok", "rule": rule}


def disable_review_rule(project_root: Path, output_dir: Path, rule_id: str) -> Dict[str, object]:
    rule = get_review_rule(project_root, rule_id)
    if rule is None:
        raise RuntimeError(f"Learned rule not found: {rule_id}")
    rule["status"] = "disabled"
    rule["activation_status"] = "disabled"
    rule["provenance"]["updated_at_epoch_ms"] = int(time.time() * 1000)
    upsert_review_rule(project_root, rule)
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "teach_me_rule_disabled",
            "rule_id": rule_id,
        },
    )
    return {"status": "ok", "rule": rule}


def backfill_review_rule(project_root: Path, output_dir: Path, rule_id: str) -> Dict[str, object]:
    rule = get_review_rule(project_root, rule_id)
    if rule is None or str(rule.get("status") or "") != "approved":
        raise RuntimeError(f"Approved learned rule not found: {rule_id}")
    episodes = discover_episode_bundles(output_dir)
    completed = []
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            continue
        rerun_review_with_approved_rules(project_root, output_dir, episode_id, focus_rule_id=rule_id)
        completed.append(episode_id)
    backfills = rule.get("audit", {}).get("backfills") if isinstance(rule.get("audit"), dict) else []
    backfills = list(backfills or [])
    backfills.append({"episode_ids": completed, "created_at_epoch_ms": int(time.time() * 1000)})
    rule["audit"]["backfills"] = backfills
    rule["provenance"]["updated_at_epoch_ms"] = int(time.time() * 1000)
    upsert_review_rule(project_root, rule)
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "teach_me_rule_backfill",
            "rule_id": rule_id,
            "episode_ids": completed,
        },
    )
    return {"status": "ok", "rule_id": rule_id, "episode_ids": completed, "episode_count": len(completed)}


def preview_text_correction(project_root: Path, output_dir: Path, episode_id: str, segment_id: int, corrected_text: str) -> Dict[str, object]:
    bundle = load_episode_bundle(project_root, output_dir, episode_id)
    segment = next((item for item in bundle["cleaned"]["segments"] if int(item["id"]) == int(segment_id)), None)
    if segment is None:
        raise RuntimeError(f"Segment {segment_id} was not found in episode {episode_id}.")
    new_text = str(corrected_text).strip()
    if not new_text:
        raise RuntimeError("Corrected text cannot be blank.")
    config = load_project_config(project_root)
    downstream = []
    for key in ("podcast_rag_project_dir", "ragscope_project_dir"):
        if str(config.get(key) or "").strip():
            downstream.append({"consumer": key, "status": "plan_required"})
    return {
        "episode_id": episode_id,
        "segment_id": int(segment_id),
        "original_text": segment["text"],
        "corrected_text": new_text,
        "changes": [{"field": "corrected_text", "before": segment["text"], "after": new_text}],
        "source_revision": file_revision(output_dir / f"{episode_id}_cleaned_speaker_transcript.json"),
        "target_files": [
            f"{episode_id}_corrected_speaker_transcript.json",
            f"{CORRECTION_MANIFEST_DIRNAME}/{episode_id}.correction-manifest-v2.json",
            f"corrections/{episode_id}_corrections.csv",
        ],
        "downstream_plans": downstream,
    }


def _write_authoritative_pair_atomic(first_path: Path, first_value: Dict[str, object], second_path: Path, second_value: Dict[str, object]) -> None:
    """Commit transcript and correction manifest together, restoring both on failure."""
    staged = []
    backups: Dict[Path, Optional[bytes]] = {}
    try:
        for path, value in ((first_path, first_value), (second_path, second_value)):
            path.parent.mkdir(parents=True, exist_ok=True)
            backups[path] = path.read_bytes() if path.exists() else None
            handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".txn")
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((Path(temporary), path))
        for temporary, path in staged:
            os.replace(temporary, path)
    except Exception:
        for path, previous in backups.items():
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _read_existing_corrections(correction_path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    if not correction_path.exists():
        return ["segment_id", "corrected_text", "speaker"], {}
    with correction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ["segment_id", "corrected_text", "speaker"])
        rows = {}
        for row in reader:
            segment_id = str(row.get("segment_id") or row.get("id") or "").strip()
            if segment_id:
                rows[segment_id] = row
        return fieldnames, rows


def _append_audit_log(project_root: Path, output_dir: Path, entry: Dict[str, object]):
    audit_log_path = resolve_workbench_paths(project_root, output_dir)["audit_log_path"]
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def apply_text_correction(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    segment_id: int,
    corrected_text: str,
    expected_revision: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    preview = preview_text_correction(project_root, output_dir, episode_id, segment_id, corrected_text)
    cleaned_path = output_dir / f"{episode_id}_cleaned_speaker_transcript.json"
    assert_write_revision(cleaned_path, expected_revision)
    cleaned_payload = _load_json(cleaned_path)
    transcript_for_contract = deepcopy(cleaned_payload)
    for segment in transcript_for_contract.get("segments") or []:
        if isinstance(segment, dict):
            segment.setdefault("source_span_id", str(segment.get("id") or ""))
    manifest_dir = output_dir / CORRECTION_MANIFEST_DIRNAME
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{episode_id}.correction-manifest-v2.json"
    existing_history: List[Dict[str, object]] = []
    if manifest_path.exists():
        existing_manifest = _load_json(manifest_path)
        existing_history = [
            dict(item)
            for item in existing_manifest.get("corrections") or []
            if isinstance(item, dict)
        ]
    for item in existing_history:
        if (
            str(item.get("source_span_id") or "") == str(segment_id)
            and str(item.get("field") or "") == "text"
            and str(item.get("status") or "") == "approved"
        ):
            item["status"] = "superseded"
            item["adjudication_state"] = "superseded"
    source_segment = next(
        item for item in transcript_for_contract["segments"]
        if str(item.get("source_span_id")) == str(segment_id)
    )
    new_correction = {
        "source_span_id": str(segment_id),
        "source_anchor": {
            "episode_id": episode_id,
            "source_span_id": str(segment_id),
            "start": float(source_segment.get("start") or 0.0),
            "end": float(source_segment.get("end") or 0.0),
            "field": "text",
        },
        "field": "text",
        "before": source_segment.get("text"),
        "after": preview["corrected_text"],
        "correction_kind": "transcript_text",
        "scope": "episode_segment",
        "reason_code": "operator_correction",
        "status": "approved",
        "adjudication_state": "accepted",
        "supersedes": [
            str(item.get("correction_id") or "")
            for item in existing_history
            if str(item.get("source_span_id") or "") == str(segment_id)
            and str(item.get("status") or "") == "superseded"
        ],
        "superseded_by": [],
        "provenance": {
            "source": "transcript_review_workbench",
            "approved_at_epoch_ms": int(time.time() * 1000),
        },
    }
    manifest, corrected_payload = build_correction_manifest_history(
        transcript_for_contract,
        [*existing_history, new_correction],
        reviewer="workbench-operator",
        producer={"name": "podcast-host-transcription-pipeline", "contract_version": "2"},
    )
    new_id = str(manifest["corrections"][-1]["correction_id"])
    for item in manifest["corrections"]:
        if str(item.get("status") or "") == "superseded" and not item.get("superseded_by"):
            item["superseded_by"] = [new_id]
    manifest["correction_set_id"] = canonical_id(correction_identity_payload(manifest), prefix="correction")
    corrected_payload["contract_version"] = "episode-contract-v2"
    corrected_payload["text_version"] = "corrected_human"
    corrected_payload["correction_lineage"] = {
        "correction_set_ids": [manifest["correction_set_id"]],
        "applied_correction_ids": [
            str(item["correction_id"])
            for item in manifest["corrections"]
            if item.get("status") == "approved"
        ],
    }
    corrected_json_path = output_dir / f"{episode_id}_corrected_speaker_transcript.json"
    _write_authoritative_pair_atomic(corrected_json_path, corrected_payload, manifest_path, manifest)

    corrections_dir = resolve_workbench_paths(project_root, output_dir)["corrections_dir"]
    _assert_within_root(project_root, corrections_dir)
    corrections_dir.mkdir(parents=True, exist_ok=True)
    correction_path = corrections_dir / f"{episode_id}_corrections.csv"
    fieldnames, rows = _read_existing_corrections(correction_path)
    fieldnames = list(dict.fromkeys(["segment_id", "corrected_text", "speaker", *fieldnames]))
    row = deepcopy(rows.get(str(segment_id), {}))
    row["segment_id"] = str(segment_id)
    row["corrected_text"] = preview["corrected_text"]
    row.setdefault("speaker", "")
    rows[str(segment_id)] = row
    correction_temp = correction_path.with_name(correction_path.name + ".tmp")
    with correction_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys(), key=lambda value: int(value)):
            writer.writerow(rows[key])
    correction_temp.replace(correction_path)
    corrected_segments = [
        _segment_summary(item)
        for item in corrected_payload.get("segments") or []
        if isinstance(item, dict)
    ]
    speaker_lines = [
        f"[{float(item.get('start') or 0.0):.2f}][{item.get('speaker') or 'UNKNOWN'}] {item.get('text') or ''}"
        for item in corrected_segments
    ]
    host_labels = {"HOST"}
    speaker_mapping = corrected_payload.get("speaker_mapping") or {}
    host_original = corrected_payload.get("host_original_speaker_id")
    if host_original not in (None, ""):
        host_labels.add(str(speaker_mapping.get(str(host_original)) or "HOST"))
    host_lines = [
        f"[{float(item.get('start') or 0.0):.2f}][{item.get('speaker') or 'UNKNOWN'}] {item.get('text') or ''}"
        for item in corrected_segments
        if str(item.get("speaker") or "") in host_labels
    ]
    (output_dir / f"{episode_id}_corrected_speaker_transcript.txt").write_text(
        "\n".join(speaker_lines), encoding="utf-8"
    )
    (output_dir / f"{episode_id}_corrected_host_only.txt").write_text(
        "\n".join(host_lines), encoding="utf-8"
    )
    downstream_event = _notify_downstream_correction(
        project_root,
        output_dir,
        episode_id,
        manifest,
    )
    audit_entry = {
        "created_at_epoch_ms": int(time.time() * 1000),
        "action": "apply_text_correction",
        "episode_id": episode_id,
        "segment_id": int(segment_id),
        "target_path": str(correction_path),
        "before": preview["original_text"],
        "after": preview["corrected_text"],
        "correction_id": new_id,
        "correction_set_id": manifest["correction_set_id"],
        "downstream_status": downstream_event["status"],
    }
    _append_audit_log(project_root, output_dir, audit_entry)
    return {
        "status": "ok",
        "target_path": str(correction_path),
        "manifest_path": str(manifest_path),
        "corrected_json_path": str(corrected_json_path),
        "correction_id": new_id,
        "correction_set_id": manifest["correction_set_id"],
        "downstream": downstream_event,
        "preview": preview,
    }


def _notify_downstream_correction(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    manifest: Dict[str, object],
) -> Dict[str, object]:
    config = load_project_config(project_root)
    event = {
        "contract_version": "correction-notification-v1",
        "episode_id": episode_id,
        "correction_set_id": manifest.get("correction_set_id"),
        "correction_manifest_path": str(
            output_dir / CORRECTION_MANIFEST_DIRNAME / f"{episode_id}.correction-manifest-v2.json"
        ),
        "stale_source_span_ids": sorted(
            {
                str(item.get("source_span_id") or "")
                for item in manifest.get("corrections") or []
                if isinstance(item, dict) and item.get("status") == "approved"
            }
        ),
        "created_at_epoch_ms": int(time.time() * 1000),
    }
    delivered = []
    missing = []
    delivery_errors = []
    for key, inbox_name in (
        ("podcast_rag_project_dir", "transcription_corrections"),
        ("ragscope_project_dir", "transcription_corrections"),
    ):
        raw = str(config.get(key) or "").strip()
        if not raw:
            missing.append(key)
            continue
        root = Path(raw)
        if not root.is_absolute():
            root = project_root / root
        if not root.exists():
            missing.append(key)
            continue
        inbox = root / "state" / inbox_name
        inbox.mkdir(parents=True, exist_ok=True)
        target = inbox / f"{manifest.get('correction_set_id')}.json"
        try:
            _write_json_atomic(target, event)
            delivered.append(str(target))
        except OSError as exc:
            delivery_errors.append({"consumer": key, "error": str(exc)})
    pending_dir = output_dir / "_downstream_corrections"
    pending_dir.mkdir(parents=True, exist_ok=True)
    pending_path = pending_dir / f"{manifest.get('correction_set_id')}.json"
    event["status"] = "downstream_failed" if delivery_errors else "delivered" if delivered and not missing else "downstream_pending"
    event["delivered_paths"] = delivered
    event["missing_consumers"] = missing
    event["delivery_errors"] = delivery_errors
    _write_json_atomic(pending_path, event)
    return {**event, "pending_path": str(pending_path)}


def list_episode_corrections(output_dir: Path, episode_id: str) -> Dict[str, object]:
    path = output_dir / CORRECTION_MANIFEST_DIRNAME / f"{episode_id}.correction-manifest-v2.json"
    if not path.exists():
        return {"episode_id": episode_id, "present": False, "corrections": []}
    payload = _load_json(path)
    return {"episode_id": episode_id, "present": True, "path": str(path), **payload}


def rollback_text_correction(
    project_root: Path,
    output_dir: Path,
    episode_id: str,
    correction_id: str,
) -> Dict[str, object]:
    current = list_episode_corrections(output_dir, episode_id)
    if not current.get("present"):
        raise RuntimeError(f"No correction history exists for {episode_id}.")
    corrections = [dict(item) for item in current.get("corrections") or [] if isinstance(item, dict)]
    target = next((item for item in corrections if str(item.get("correction_id")) == correction_id), None)
    if target is None:
        raise RuntimeError(f"Correction was not found: {correction_id}")
    target["status"] = "disabled"
    target["adjudication_state"] = "rejected"
    cleaned_path = output_dir / f"{episode_id}_cleaned_speaker_transcript.json"
    source = _load_json(cleaned_path)
    for segment in source.get("segments") or []:
        if isinstance(segment, dict):
            segment.setdefault("source_span_id", str(segment.get("id") or ""))
    manifest, corrected = build_correction_manifest_history(
        source,
        corrections,
        reviewer="workbench-operator",
        producer={"name": "podcast-host-transcription-pipeline", "contract_version": "2"},
    )
    manifest_path = output_dir / CORRECTION_MANIFEST_DIRNAME / f"{episode_id}.correction-manifest-v2.json"
    _write_json_atomic(manifest_path, manifest)
    corrected["contract_version"] = "episode-contract-v2"
    corrected["text_version"] = "corrected_human"
    corrected["correction_lineage"] = {
        "correction_set_ids": [manifest["correction_set_id"]],
        "applied_correction_ids": [
            str(item["correction_id"]) for item in manifest["corrections"] if item.get("status") == "approved"
        ],
    }
    corrected_path = output_dir / f"{episode_id}_corrected_speaker_transcript.json"
    _write_json_atomic(corrected_path, corrected)
    active = [item for item in manifest["corrections"] if item.get("status") == "approved"]
    correction_path = resolve_workbench_paths(project_root, output_dir)["corrections_dir"] / f"{episode_id}_corrections.csv"
    correction_path.parent.mkdir(parents=True, exist_ok=True)
    correction_temp = correction_path.with_name(correction_path.name + ".tmp")
    with correction_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment_id", "corrected_text", "speaker"])
        writer.writeheader()
        for item in sorted(active, key=lambda value: int(value.get("source_span_id") or 0)):
            writer.writerow(
                {
                    "segment_id": item.get("source_span_id"),
                    "corrected_text": item.get("after"),
                    "speaker": "",
                }
            )
    correction_temp.replace(correction_path)
    corrected_segments = [item for item in corrected.get("segments") or [] if isinstance(item, dict)]
    speaker_mapping = corrected.get("speaker_mapping") or {}
    host_original = corrected.get("host_original_speaker_id")
    host_labels = {"HOST"}
    if host_original not in (None, ""):
        host_labels.add(str(speaker_mapping.get(str(host_original)) or "HOST"))
    (output_dir / f"{episode_id}_corrected_speaker_transcript.txt").write_text(
        "\n".join(
            f"[{float(item.get('start') or 0.0):.2f}][{item.get('speaker') or 'UNKNOWN'}] {item.get('text') or ''}"
            for item in corrected_segments
        ),
        encoding="utf-8",
    )
    (output_dir / f"{episode_id}_corrected_host_only.txt").write_text(
        "\n".join(
            f"[{float(item.get('start') or 0.0):.2f}][{item.get('speaker') or 'UNKNOWN'}] {item.get('text') or ''}"
            for item in corrected_segments
            if str(item.get("speaker") or "") in host_labels
        ),
        encoding="utf-8",
    )
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "rollback_text_correction",
            "episode_id": episode_id,
            "correction_id": correction_id,
            "correction_set_id": manifest["correction_set_id"],
        },
    )
    return {"status": "rolled_back", "correction_id": correction_id, "manifest": manifest}


def _read_preferred_terms(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preview_preferred_term_addition(project_root: Path, output_dir: Path, term: str) -> Dict[str, object]:
    paths = resolve_workbench_paths(project_root, output_dir)
    target = paths["preferred_terms_path"]
    _assert_within_root(project_root, target)
    new_term = str(term).strip()
    if not new_term:
        raise RuntimeError("Preferred term cannot be blank.")
    existing = _read_preferred_terms(target)
    already_present = new_term in existing
    return {
        "target_path": str(target),
        "term": new_term,
        "already_present": already_present,
        "line_will_be_added": None if already_present else new_term,
    }


def apply_preferred_term_addition(project_root: Path, output_dir: Path, term: str) -> Dict[str, object]:
    preview = preview_preferred_term_addition(project_root, output_dir, term)
    target = Path(preview["target_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with _GLOSSARY_WRITE_LOCK:
        existing = _read_preferred_terms(target)
        if preview["term"] not in existing:
            existing.append(preview["term"])
            target.write_text("\n".join(existing) + "\n", encoding="utf-8")
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "apply_preferred_term_addition",
            "target_path": str(target),
            "after": preview["term"],
        },
    )
    return {"status": "ok", "preview": preview}


def _read_replacement_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    result = {}
    for preferred, aliases in payload.items():
        if isinstance(aliases, list):
            result[str(preferred)] = [str(alias).strip() for alias in aliases if str(alias).strip()]
    return result


def preview_replacement_map_update(project_root: Path, output_dir: Path, preferred_term: str, alias: str) -> Dict[str, object]:
    paths = resolve_workbench_paths(project_root, output_dir)
    target = paths["replacement_map_path"]
    _assert_within_root(project_root, target)
    preferred = str(preferred_term).strip()
    alias_text = str(alias).strip()
    if not preferred or not alias_text:
        raise RuntimeError("Preferred term and alias are both required.")
    payload = _read_replacement_map(target)
    aliases = list(payload.get(preferred) or [])
    already_present = alias_text in aliases
    if not already_present:
        aliases.append(alias_text)
    return {
        "target_path": str(target),
        "preferred_term": preferred,
        "alias": alias_text,
        "already_present": already_present,
        "updated_aliases": aliases,
    }


def apply_replacement_map_update(project_root: Path, output_dir: Path, preferred_term: str, alias: str) -> Dict[str, object]:
    preview = preview_replacement_map_update(project_root, output_dir, preferred_term, alias)
    target = Path(preview["target_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with _GLOSSARY_WRITE_LOCK:
        payload = _read_replacement_map(target)
        aliases = list(payload.get(preview["preferred_term"]) or [])
        if preview["alias"] not in aliases:
            aliases.append(preview["alias"])
            payload[preview["preferred_term"]] = aliases
            target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _append_audit_log(
        project_root,
        output_dir,
        {
            "created_at_epoch_ms": int(time.time() * 1000),
            "action": "apply_replacement_map_update",
            "target_path": str(target),
            "preferred_term": preview["preferred_term"],
            "alias": preview["alias"],
        },
    )
    return {"status": "ok", "preview": preview}


def load_audit_log(project_root: Path, output_dir: Path, limit: int = 200) -> List[Dict[str, object]]:
    audit_log_path = resolve_workbench_paths(project_root, output_dir)["audit_log_path"]
    if not audit_log_path.exists():
        return []
    entries: List[Dict[str, object]] = []
    for line in audit_log_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            entries.append(payload)
    return entries[-limit:]


def list_review_rules(project_root: Path) -> List[Dict[str, object]]:
    return load_review_rule_library(project_root)["rules"]


def get_review_rule(project_root: Path, rule_id: str) -> Optional[Dict[str, object]]:
    return load_single_review_rule(project_root, rule_id)
