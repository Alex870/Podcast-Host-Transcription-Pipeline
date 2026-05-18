import re
from typing import Dict, List


TRANSCRIPT_SCHEMA_VERSION = 2
PIPELINE_NAME = "podcast-host-transcription-pipeline"
REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "pipeline",
    "source_file",
    "metadata",
    "segments",
}
REQUIRED_SEGMENT_FIELDS = {
    "id",
    "start",
    "end",
    "speaker",
    "text",
    "episode_date",
    "episode_sort_key",
    "transcription_confidence",
}


def validate_transcript_payload(payload: Dict[str, object]) -> List[str]:
    """Validate the transcript JSON contract shared with downstream RAG tooling."""

    errors = []
    missing_top = sorted(field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in payload)
    if missing_top:
        errors.append(f"missing top-level fields: {', '.join(missing_top)}")

    if payload.get("schema_version") != TRANSCRIPT_SCHEMA_VERSION:
        errors.append("missing or unsupported schema_version")
    if payload.get("pipeline") != PIPELINE_NAME:
        errors.append("missing or unsupported pipeline")
    if not payload.get("source_file"):
        errors.append("missing source_file")
    if not isinstance(payload.get("metadata"), dict):
        errors.append("missing metadata object")
    if not isinstance(payload.get("segments"), list):
        errors.append("missing segments list")
        return errors

    seen_ids = set()
    previous_start = None
    for index, segment in enumerate(payload["segments"]):
        if not isinstance(segment, dict):
            errors.append(f"segment {index} is not an object")
            continue

        missing_segment = sorted(field for field in REQUIRED_SEGMENT_FIELDS if field not in segment)
        if missing_segment:
            errors.append(f"segment {index} missing fields: {', '.join(missing_segment)}")

        segment_id = segment.get("id")
        if segment_id in seen_ids:
            errors.append(f"duplicate segment id {segment_id}")
        seen_ids.add(segment_id)

        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            errors.append(f"segment {index} missing start/end")
        else:
            try:
                start_float = float(start)
                end_float = float(end)
                if end_float < start_float:
                    errors.append(f"segment {index} ends before it starts")
                if previous_start is not None and start_float < previous_start:
                    errors.append(f"segment {index} starts before previous segment")
                previous_start = start_float
            except (TypeError, ValueError):
                errors.append(f"segment {index} has non-numeric start/end")

        if not str(segment.get("text", "")).strip():
            errors.append(f"segment {index} has empty text")
        if not segment.get("speaker"):
            errors.append(f"segment {index} missing speaker")
        confidence = segment.get("transcription_confidence")
        if not isinstance(confidence, dict):
            errors.append(f"segment {index} missing transcription_confidence")
        if segment.get("episode_date") != payload.get("episode_date"):
            errors.append(f"segment {index} episode_date does not match top-level episode_date")
        if segment.get("episode_date") and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(segment["episode_date"])):
            errors.append(f"segment {index} has invalid episode_date format")

    return errors


def transcript_contract_summary() -> Dict[str, object]:
    return {
        "pipeline": PIPELINE_NAME,
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "required_top_level_fields": sorted(REQUIRED_TOP_LEVEL_FIELDS),
        "required_segment_fields": sorted(REQUIRED_SEGMENT_FIELDS),
    }

