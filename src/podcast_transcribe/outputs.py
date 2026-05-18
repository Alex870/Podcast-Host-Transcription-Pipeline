import csv
import datetime as dt
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Set

from podcast_transcribe.contract import PIPELINE_NAME, TRANSCRIPT_SCHEMA_VERSION, validate_transcript_payload
from podcast_transcribe.quality import classify_segment_text, summarize_content_quality

DATE_FORMAT_SPECS = {
    "YYYYMMDD": {"regex": r"(?<!\d)(\d{8})(?!\d)", "parser": "%Y%m%d"},
    "YYYY-MM-DD": {"regex": r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", "parser": "%Y-%m-%d"},
    "YYYY_MM_DD": {"regex": r"(?<!\d)(\d{4}_\d{2}_\d{2})(?!\d)", "parser": "%Y_%m_%d"},
    "YYYY.MM.DD": {"regex": r"(?<!\d)(\d{4}\.\d{2}\.\d{2})(?!\d)", "parser": "%Y.%m.%d"},
    "MM-DD-YYYY": {"regex": r"(?<!\d)(\d{2}-\d{2}-\d{4})(?!\d)", "parser": "%m-%d-%Y"},
    "MM_DD_YYYY": {"regex": r"(?<!\d)(\d{2}_\d{2}_\d{4})(?!\d)", "parser": "%m_%d_%Y"},
    "MM.DD.YYYY": {"regex": r"(?<!\d)(\d{2}\.\d{2}\.\d{4})(?!\d)", "parser": "%m.%d.%Y"},
    "DD-MM-YYYY": {"regex": r"(?<!\d)(\d{2}-\d{2}-\d{4})(?!\d)", "parser": "%d-%m-%Y"},
    "DD_MM_YYYY": {"regex": r"(?<!\d)(\d{2}_\d{2}_\d{4})(?!\d)", "parser": "%d_%m_%Y"},
    "DD.MM.YYYY": {"regex": r"(?<!\d)(\d{2}\.\d{2}\.\d{4})(?!\d)", "parser": "%d.%m.%Y"},
}

DATE_FORMAT_PRESETS = {
    "strict_iso": ["YYYYMMDD", "YYYY-MM-DD", "YYYY_MM_DD", "YYYY.MM.DD"],
    "american_podcast": [
        "YYYYMMDD",
        "YYYY-MM-DD",
        "YYYY_MM_DD",
        "YYYY.MM.DD",
        "MM-DD-YYYY",
        "MM_DD_YYYY",
        "MM.DD.YYYY",
    ],
    "mixed_common": [
        "YYYYMMDD",
        "YYYY-MM-DD",
        "YYYY_MM_DD",
        "YYYY.MM.DD",
        "MM-DD-YYYY",
        "MM_DD_YYYY",
        "MM.DD.YYYY",
        "DD-MM-YYYY",
        "DD_MM_YYYY",
        "DD.MM.YYYY",
    ],
}

DEFAULT_FILENAME_DATE_PRESET = "strict_iso"
DEFAULT_FILENAME_DATE_POSITION = "last"


def stable_json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def resolve_filename_date_config(date_config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    payload = date_config or {}
    enabled = payload.get("enabled", True)
    preset = str(payload.get("preset") or DEFAULT_FILENAME_DATE_PRESET).strip() or DEFAULT_FILENAME_DATE_PRESET
    configured_formats = payload.get("formats")
    position = str(payload.get("position") or DEFAULT_FILENAME_DATE_POSITION).strip().lower()
    if position not in {"first", "last"}:
        position = DEFAULT_FILENAME_DATE_POSITION

    if configured_formats:
        formats = [str(item).strip() for item in configured_formats if str(item).strip() in DATE_FORMAT_SPECS]
    else:
        formats = DATE_FORMAT_PRESETS.get(preset, DATE_FORMAT_PRESETS[DEFAULT_FILENAME_DATE_PRESET])

    return {
        "enabled": bool(enabled),
        "preset": preset,
        "position": position,
        "formats": formats,
    }


def build_episode_metadata(source_file: str, date_config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Extract stable episode-date metadata from a source filename using configured date formats."""

    source_path = Path(source_file)
    source_name = (
        PureWindowsPath(source_file).name
        if "\\" in source_file or re.match(r"^[A-Za-z]:", source_file)
        else source_path.name
    )
    metadata: Dict[str, object] = {
        "source_file": source_file,
        "source_filename": source_name,
        "episode_date": "",
        "episode_date_compact": "",
        "episode_year": "",
        "episode_month": "",
        "episode_day": "",
        "episode_sort_key": "",
    }

    resolved_config = resolve_filename_date_config(date_config)
    if not resolved_config["enabled"]:
        return metadata

    stem = Path(source_name).stem
    for format_name in resolved_config["formats"]:
        spec = DATE_FORMAT_SPECS.get(format_name)
        if spec is None:
            continue

        valid_matches = []
        for match in re.finditer(spec["regex"], stem):
            raw_value = match.group(1)
            try:
                episode_date = dt.datetime.strptime(raw_value, spec["parser"]).date()
            except ValueError:
                continue
            valid_matches.append((match.start(), episode_date))

        if not valid_matches:
            continue

        _, resolved_date = valid_matches[0 if resolved_config["position"] == "first" else -1]
        compact = resolved_date.strftime("%Y%m%d")
        metadata.update(
            {
                "episode_date": resolved_date.isoformat(),
                "episode_date_compact": compact,
                "episode_year": resolved_date.year,
                "episode_month": resolved_date.month,
                "episode_day": resolved_date.day,
                "episode_sort_key": int(compact),
            }
        )
        break

    return metadata


def write_review_csv(path: Path, rows: List[Dict[str, object]]):
    fieldnames = [
        "issue_type",
        "speaker",
        "start",
        "end",
        "score",
        "details",
        "text",
        "source_file",
        "episode_date",
        "episode_date_compact",
        "episode_sort_key",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def segment_confidence(avg_logprob: Optional[float], no_speech_prob: Optional[float]) -> Dict[str, object]:
    warnings = []
    score = 1.0

    if avg_logprob is not None:
        try:
            avg_logprob_value = float(avg_logprob)
            if avg_logprob_value < -1.0:
                warnings.append("low_avg_logprob")
            score = min(score, max(0.0, min(1.0, (avg_logprob_value + 1.5) / 1.5)))
        except (TypeError, ValueError):
            warnings.append("invalid_avg_logprob")

    if no_speech_prob is not None:
        try:
            no_speech_value = float(no_speech_prob)
            if no_speech_value > 0.6:
                warnings.append("high_no_speech_prob")
            score = min(score, max(0.0, 1.0 - no_speech_value))
        except (TypeError, ValueError):
            warnings.append("invalid_no_speech_prob")

    if avg_logprob is None and no_speech_prob is None:
        return {
            "score": "",
            "quality": "unknown",
            "warnings": ["missing_confidence_inputs"],
        }

    if score >= 0.75:
        quality = "high"
    elif score >= 0.45:
        quality = "medium"
    else:
        quality = "low"

    return {
        "score": round(score, 4),
        "quality": quality,
        "warnings": warnings,
    }


def write_text_transcript(
    path: Path,
    segments,
    format_timestamp,
    host_only: bool = False,
    host_labels: Optional[Set[str]] = None,
    metadata: Optional[Dict[str, object]] = None,
):
    lines = []
    if metadata:
        for key in (
            "source_file",
            "source_filename",
            "episode_date",
            "episode_date_compact",
            "episode_sort_key",
            "text_version",
        ):
            value = metadata.get(key)
            if value not in ("", None):
                lines.append(f"# {key}: {value}")
        if lines:
            lines.append("")

    host_labels = host_labels or {"HOST"}
    for segment in segments:
        if host_only and segment.speaker not in host_labels:
            continue
        label = segment.speaker or "UNKNOWN"
        lines.append(f"[{format_timestamp(segment.start)}][{label}] {segment.text}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_output(
    path: Path,
    source_file: str,
    info_payload: Dict[str, object],
    diarized_turns: List[Dict[str, object]],
    segments,
    speaker_mapping: Dict[str, str],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    known_assignments: Dict[str, Dict[str, object]],
    metadata: Optional[Dict[str, object]] = None,
    text_version: str = "original",
    pipeline_version: str = PIPELINE_NAME,
):
    """Write the RAG-ready transcript JSON payload after schema validation."""

    metadata = metadata or build_episode_metadata(source_file)
    metadata = {
        **metadata,
        "pipeline": PIPELINE_NAME,
        "pipeline_version": pipeline_version,
        "transcript_schema_version": TRANSCRIPT_SCHEMA_VERSION,
    }
    segment_metadata = {
        "source_file": metadata.get("source_file", source_file),
        "source_filename": metadata.get("source_filename", Path(source_file).name),
        "episode_date": metadata.get("episode_date", ""),
        "episode_date_compact": metadata.get("episode_date_compact", ""),
        "episode_sort_key": metadata.get("episode_sort_key", ""),
        "transcript_schema_version": TRANSCRIPT_SCHEMA_VERSION,
    }
    payload = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "pipeline_version": pipeline_version,
        "source_file": source_file,
        "metadata": metadata,
        "episode_date": metadata.get("episode_date", ""),
        "episode_date_compact": metadata.get("episode_date_compact", ""),
        "episode_sort_key": metadata.get("episode_sort_key", ""),
        "text_version": text_version,
        "transcription": info_payload,
        "host_detected": host_speaker is not None,
        "host_original_speaker_id": host_speaker,
        "speaker_mapping": speaker_mapping,
        "known_speaker_assignments": known_assignments,
        "speaker_durations_seconds": durations,
        "diarization_turns": diarized_turns,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
                **(
                    {
                        "original_text": segment.original_text,
                        "cleanup_applied": segment.cleanup_applied,
                    }
                    if hasattr(segment, "original_text")
                    else {}
                ),
                **(
                    {
                        "cleanup_level": segment.cleanup_level,
                    }
                    if hasattr(segment, "cleanup_level")
                    else {}
                ),
                **(
                    {
                        "manual_correction_applied": segment.manual_correction_applied,
                        "original_speaker": getattr(segment, "original_speaker", ""),
                    }
                    if hasattr(segment, "manual_correction_applied")
                    else {}
                ),
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "transcription_confidence": segment_confidence(segment.avg_logprob, segment.no_speech_prob),
                "content_quality": classify_segment_text(
                    segment.text,
                    duration_seconds=max(0.0, float(segment.end or 0.0) - float(segment.start or 0.0)),
                ),
                **segment_metadata,
                "words": [asdict(word) for word in segment.words],
            }
            for segment in segments
        ],
    }
    payload["content_quality_summary"] = summarize_content_quality(
        segment.get("content_quality", {}) for segment in payload["segments"]
    )
    errors = validate_transcript_payload(payload)
    if errors:
        raise ValueError(f"Transcript JSON schema validation failed for {path}: {'; '.join(errors[:10])}")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def write_speaker_identity_review_csv(
    path: Path,
    speaker_mapping: Dict[str, str],
    durations: Dict[str, float],
    similarity_scores: Dict[str, float],
    known_assignments: Dict[str, Dict[str, object]],
    host_speaker: Optional[str],
):
    fieldnames = [
        "speaker_id",
        "assigned_label",
        "duration_seconds",
        "host_similarity",
        "similarity_margin_from_top",
        "is_host",
        "known_speaker_name",
        "known_speaker_score",
        "known_speaker_is_host",
        "review_recommended",
        "review_reason",
    ]
    sorted_scores = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1] if sorted_scores else None
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for speaker_id in sorted(speaker_mapping, key=lambda key: speaker_mapping.get(key, key)):
            score = similarity_scores.get(speaker_id)
            assignment = known_assignments.get(speaker_id, {})
            reasons = []
            if score is None:
                reasons.append("no host similarity score")
            elif score < 0.55:
                reasons.append("low host similarity")
            if top_score is not None and score is not None and top_score - score < 0.05 and speaker_id != host_speaker:
                reasons.append("near top host score")
            if durations.get(speaker_id, 0.0) < 20.0:
                reasons.append("short speaker duration")
            writer.writerow(
                {
                    "speaker_id": speaker_id,
                    "assigned_label": speaker_mapping.get(speaker_id, ""),
                    "duration_seconds": round(float(durations.get(speaker_id, 0.0)), 2),
                    "host_similarity": round(score, 4) if score is not None else "",
                    "similarity_margin_from_top": round(top_score - score, 4) if top_score is not None and score is not None else "",
                    "is_host": speaker_id == host_speaker,
                    "known_speaker_name": assignment.get("name", ""),
                    "known_speaker_score": round(float(assignment["score"]), 4) if assignment.get("score") is not None else "",
                    "known_speaker_is_host": assignment.get("is_host", ""),
                    "review_recommended": bool(reasons),
                    "review_reason": "; ".join(reasons),
                }
            )


def write_output_manifest(
    path: Path,
    source_file: str,
    source_fingerprint: Dict[str, object],
    config: Dict[str, object],
    outputs: List[Path],
    timings: Dict[str, float],
    summary: Dict[str, object],
):
    """Write a manifest that fingerprints inputs, config, outputs, timings, and summary state."""

    output_records = []
    for output_path in outputs:
        if output_path.exists():
            output_records.append(
                {
                    "path": str(output_path),
                    "filename": output_path.name,
                    "size_bytes": output_path.stat().st_size,
                    "sha1": hashlib.sha1(output_path.read_bytes()).hexdigest(),
                }
            )

    manifest = {
        "manifest_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pipeline": PIPELINE_NAME,
        "transcript_schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "source_file": source_file,
        "source_fingerprint": source_fingerprint,
        "config_fingerprint": stable_json_fingerprint(config),
        "config": config,
        "timings_seconds": {key: round(float(value), 3) for key, value in timings.items()},
        "summary": summary,
        "outputs": output_records,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_batch_report_md(path: Path, rows: List[Dict[str, object]], elapsed_seconds: Optional[float] = None):
    total_episodes = len(rows)
    total_segments = sum(int(row.get("transcript_segments") or 0) for row in rows)
    total_review_rows = sum(int(row.get("review_row_count") or 0) for row in rows)
    detected_hosts = sum(1 for row in rows if str(row.get("host_detected")).lower() in {"true", "1", "yes"})
    average_priority = (
        sum(float(row.get("review_priority_score") or 0.0) for row in rows) / total_episodes
        if total_episodes
        else 0.0
    )
    sorted_rows = sorted(rows, key=lambda row: float(row.get("review_priority_score") or 0.0), reverse=True)
    lines = [
        "# Podcast Transcription Batch Report",
        "",
        f"- Generated at: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- Episodes: {total_episodes}",
        f"- Hosts detected: {detected_hosts}/{total_episodes}",
        f"- Transcript segments: {total_segments}",
        f"- Review rows: {total_review_rows}",
        f"- Average review priority score: {average_priority:.2f}",
    ]
    if elapsed_seconds is not None:
        lines.append(f"- Batch elapsed time: {dt.timedelta(seconds=int(elapsed_seconds))}")
    lines.extend(["", "## Highest Review Priority Episodes", "", "| Episode | Date | Score | Reason |", "|---|---:|---:|---|"])
    for row in sorted_rows[:10]:
        reason = str(row.get("review_priority_reason", "")).replace("|", "\\|")
        lines.append(
            f"| {row.get('episode', '')} | {row.get('episode_date', '')} | "
            f"{row.get('review_priority_score', '')} | {reason} |"
        )
    speaker_stats = _speaker_aggregate_stats_for_report(rows)
    if speaker_stats:
        lines.extend(["", "## Speaker Aggregates", "", "| Speaker | Episodes | Seconds | Avg Similarity | Avg Review Priority |", "|---|---:|---:|---:|---:|"])
        for speaker, item in sorted(speaker_stats.items()):
            lines.append(
                f"| {speaker} | {item['episode_count']} | {item['total_duration_seconds']} | "
                f"{item['average_similarity']} | {item['average_review_priority']} |"
            )
    candidates = _promotion_candidates_for_report(rows)
    if candidates:
        lines.extend(["", "## Recurring Speaker Promotion Candidates", ""])
        for candidate in candidates:
            lines.append(
                f"- `{candidate['speaker']}` appears in {candidate['episode_count']} episode(s) "
                f"with {candidate['total_duration_seconds']} seconds of speech: {candidate['recommendation']}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _speaker_aggregate_stats_for_report(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for row in rows:
        speaker = str(row.get("host_label") or "").strip()
        if not speaker:
            continue
        item = stats.setdefault(
            speaker,
            {
                "episode_count": 0,
                "total_duration_seconds": 0.0,
                "similarity_scores": [],
                "review_priority_scores": [],
            },
        )
        item["episode_count"] += 1
        item["total_duration_seconds"] += float(row.get("host_duration_seconds") or 0.0)
        if row.get("top_host_similarity") not in ("", None):
            item["similarity_scores"].append(float(row["top_host_similarity"]))
        if row.get("review_priority_score") not in ("", None):
            item["review_priority_scores"].append(float(row["review_priority_score"]))

    result = {}
    for speaker, item in stats.items():
        scores = item.pop("similarity_scores")
        priorities = item.pop("review_priority_scores")
        item["average_similarity"] = round(sum(scores) / len(scores), 4) if scores else ""
        item["average_review_priority"] = round(sum(priorities) / len(priorities), 2) if priorities else ""
        item["total_duration_seconds"] = round(item["total_duration_seconds"], 2)
        result[speaker] = item
    return result


def _promotion_candidates_for_report(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    candidates = []
    for speaker, item in _speaker_aggregate_stats_for_report(rows).items():
        if speaker.upper().startswith("SPEAKER_") and (
            item["episode_count"] >= 3
            or item["total_duration_seconds"] >= 600.0
        ):
            candidates.append(
                {
                    "speaker": speaker,
                    "episode_count": item["episode_count"],
                    "total_duration_seconds": item["total_duration_seconds"],
                    "recommendation": "review as recurring known speaker",
                }
            )
    return candidates

