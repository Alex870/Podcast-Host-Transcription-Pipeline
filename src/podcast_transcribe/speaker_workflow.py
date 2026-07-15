"""Dependency-light cross-episode speaker review and safe-write helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def file_revision(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False, "sha256": ""}
    data = path.read_bytes()
    return {"exists": True, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assert_write_revision(path: Path, expected_revision: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Raise a useful conflict error when a workbench write raced another writer."""

    actual = file_revision(path)
    if expected_revision is not None and actual != expected_revision:
        raise RuntimeError(
            f"Write conflict for {path}: the source changed after preview. "
            "Reload the episode and preview the correction again."
        )
    return actual


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _is_unknown(label: object) -> bool:
    value = str(label or "").strip().upper()
    return not value or value == "UNKNOWN" or value.startswith("SPEAKER_")


def collect_speaker_evidence(output_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cleaned_path in sorted(output_dir.glob("*_cleaned_speaker_transcript.json")):
        episode_id = cleaned_path.name[: -len("_cleaned_speaker_transcript.json")]
        cleaned = _load_json(cleaned_path)
        reviewed_path = output_dir / f"{episode_id}_reviewed_speaker_transcript.json"
        reviewed = _load_json(reviewed_path) if reviewed_path.exists() else {}
        reviewed_by_id = {int(item.get("id")): item for item in reviewed.get("segments") or [] if isinstance(item, dict) and item.get("id") is not None}
        for segment in cleaned.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            segment_id = int(segment.get("id") or 0)
            text = str(segment.get("text") or "")
            reviewed_text = str((reviewed_by_id.get(segment_id) or {}).get("text") or "")
            rows.append(
                {
                    "episode_id": episode_id,
                    "segment_id": segment_id,
                    "speaker": str(segment.get("speaker") or "UNKNOWN"),
                    "text": text,
                    "changed": bool(reviewed_text and reviewed_text != text),
                    "reviewed_text": reviewed_text,
                    "evidence_clip": {
                        "audio_path": str(cleaned_path),
                        "start": float(segment.get("start") or 0.0),
                        "end": float(segment.get("end") or 0.0),
                    },
                    "source_revision": file_revision(cleaned_path),
                }
            )
    return rows


def group_recurring_unknown_speakers(rows: List[Dict[str, object]], min_episode_count: int = 2) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = defaultdict(lambda: {"episodes": set(), "evidence_clips": []})
    for row in rows:
        if not _is_unknown(row.get("speaker")):
            continue
        label = str(row.get("speaker") or "UNKNOWN")
        grouped[label]["episodes"].add(str(row.get("episode_id") or ""))
        grouped[label]["evidence_clips"].append(
            {
                "episode_id": row.get("episode_id"),
                "segment_id": row.get("segment_id"),
                "clip": row.get("evidence_clip"),
                "text": row.get("text"),
            }
        )
    result = []
    for speaker, item in grouped.items():
        episodes = sorted(value for value in item["episodes"] if value)
        if len(episodes) >= min_episode_count:
            result.append({"speaker": speaker, "episode_count": len(episodes), "episodes": episodes, "evidence_clips": item["evidence_clips"]})
    return sorted(result, key=lambda item: (-int(item["episode_count"]), str(item["speaker"])))


def build_cross_episode_speaker_view(output_dir: Path, view: str = "all") -> Dict[str, object]:
    rows = collect_speaker_evidence(output_dir)
    normalized_view = str(view or "all").lower()
    if normalized_view == "changed":
        rows = [row for row in rows if row["changed"]]
    elif normalized_view == "speaker":
        rows = [row for row in rows if str(row.get("speaker") or "").upper() not in {"HOST", ""}]
    elif normalized_view not in {"all", "changed", "speaker"}:
        raise ValueError("view must be all, changed, or speaker")
    return {
        "workflow_version": 1,
        "view": normalized_view,
        "row_count": len(rows),
        "rows": rows,
        "recurring_unknown_speakers": group_recurring_unknown_speakers(collect_speaker_evidence(output_dir)),
        "changed_count": sum(bool(row["changed"]) for row in collect_speaker_evidence(output_dir)),
    }
