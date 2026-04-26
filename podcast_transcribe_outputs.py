import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set


def write_review_csv(path: Path, rows: List[Dict[str, object]]):
    fieldnames = ["issue_type", "speaker", "start", "end", "score", "details", "text", "source_file"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text_transcript(
    path: Path,
    segments,
    format_timestamp,
    host_only: bool = False,
    host_labels: Optional[Set[str]] = None,
):
    lines = []
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
):
    payload = {
        "source_file": source_file,
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
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "words": [asdict(word) for word in segment.words],
            }
            for segment in segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
