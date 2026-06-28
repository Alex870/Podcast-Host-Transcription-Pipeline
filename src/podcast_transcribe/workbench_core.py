"""Core helpers for the transcript review workbench.

This module stays intentionally lightweight so the workbench can inspect
processed artifacts without importing the heavy transcription runtime.
"""

from __future__ import annotations

import csv
import json
import threading
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from podcast_transcribe.config import resolve_review_runtime_config
from podcast_transcribe.contract import validate_reviewed_transcript_payload, validate_transcript_payload


WORKBENCH_DIRNAME = "_workbench"
SCAN_CACHE_SUBDIR = "semantic_scan"
AUDIT_LOG_SUBDIR = ".workbench"
AUDIT_LOG_FILENAME = "audit_log.jsonl"
ISSUE_RESOLUTION_SUBDIR = "issue_resolution"
DEFAULT_CORRECTIONS_DIRNAME = "corrections"

_GLOSSARY_WRITE_LOCK = threading.Lock()


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


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
    return {
        "corrections_dir": corrections_dir,
        "preferred_terms_path": preferred_terms_path,
        "replacement_map_path": replacement_map_path,
        "workbench_dir": workbench_dir,
        "scan_cache_dir": workbench_dir / SCAN_CACHE_SUBDIR,
        "issue_resolution_dir": workbench_dir / ISSUE_RESOLUTION_SUBDIR,
        "audit_log_path": project_root / AUDIT_LOG_SUBDIR / AUDIT_LOG_FILENAME,
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
    }


def _workbench_scan_system_prompt() -> str:
    return (
        "You are reviewing cleaned podcast transcript segments for likely transcription problems. "
        "Return strict JSON with a top-level 'findings' array. "
        "Each finding must include finding_id, issue_type, severity, reason, segment_ids, and may include suggested_text. "
        "Do not rewrite the whole transcript. Only flag likely problems. "
        "Keep findings compact and evidence-based."
    )


def _openai_compatible_request(base_url: str, model_name: str, system_prompt: str, user_prompt: str, max_tokens: int = 1600) -> Dict[str, object]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model_name,
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            ensure_ascii=True,
        ).encode("utf-8"),
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


def preview_text_correction(project_root: Path, output_dir: Path, episode_id: str, segment_id: int, corrected_text: str) -> Dict[str, object]:
    bundle = load_episode_bundle(project_root, output_dir, episode_id)
    segment = next((item for item in bundle["cleaned"]["segments"] if int(item["id"]) == int(segment_id)), None)
    if segment is None:
        raise RuntimeError(f"Segment {segment_id} was not found in episode {episode_id}.")
    new_text = str(corrected_text).strip()
    if not new_text:
        raise RuntimeError("Corrected text cannot be blank.")
    return {
        "episode_id": episode_id,
        "segment_id": int(segment_id),
        "original_text": segment["text"],
        "corrected_text": new_text,
        "changes": [{"field": "corrected_text", "before": segment["text"], "after": new_text}],
    }


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


def apply_text_correction(project_root: Path, output_dir: Path, episode_id: str, segment_id: int, corrected_text: str) -> Dict[str, object]:
    preview = preview_text_correction(project_root, output_dir, episode_id, segment_id, corrected_text)
    corrections_dir = resolve_workbench_paths(project_root, output_dir)["corrections_dir"]
    _assert_within_root(project_root, corrections_dir)
    corrections_dir.mkdir(parents=True, exist_ok=True)
    correction_path = corrections_dir / f"{episode_id}_corrections.csv"
    fieldnames, rows = _read_existing_corrections(correction_path)
    if "segment_id" not in fieldnames:
        fieldnames = ["segment_id", "corrected_text", "speaker"]
    row = deepcopy(rows.get(str(segment_id), {}))
    row["segment_id"] = str(segment_id)
    row["corrected_text"] = preview["corrected_text"]
    row.setdefault("speaker", "")
    rows[str(segment_id)] = row
    with correction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys(), key=lambda value: int(value)):
            writer.writerow(rows[key])
    audit_entry = {
        "created_at_epoch_ms": int(time.time() * 1000),
        "action": "apply_text_correction",
        "episode_id": episode_id,
        "segment_id": int(segment_id),
        "target_path": str(correction_path),
        "before": preview["original_text"],
        "after": preview["corrected_text"],
    }
    _append_audit_log(project_root, output_dir, audit_entry)
    return {"status": "ok", "target_path": str(correction_path), "preview": preview}


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
