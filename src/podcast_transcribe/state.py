import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


RESUME_STATE_FILENAME = "_processed_files.json"
SUMMARY_FILENAME = "_episode_review_summary.csv"
REVIEW_CALIBRATION_FILENAME = "_review_calibration_state.json"
DIARIZATION_HISTORY_FILENAME = "_diarization_history_state.json"
CHECKPOINT_DIRNAME = "_processing_checkpoints"
ARTIFACT_DIRNAME = "_processing_artifacts"
STAGE_ARTIFACT_VERSION = 2


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8"):
    """Replace a state file atomically so interruption cannot leave partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def audio_file_fingerprint(audio_path: Path) -> Dict[str, object]:
    stat = audio_path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def expected_output_paths(audio_path: Path, output_dir: Path) -> List[Path]:
    base_name = audio_path.stem
    return [
        output_dir / f"{base_name}_speaker_transcript.txt",
        output_dir / f"{base_name}_host_only.txt",
        output_dir / f"{base_name}_review.csv",
        output_dir / f"{base_name}_speaker_transcript.json",
    ]


def stage_artifact_path(output_dir: Path, audio_path: Path, stage: str) -> Path:
    return output_dir / ARTIFACT_DIRNAME / audio_path.stem / f"{stage}.json"


def save_stage_artifact(
    output_dir: Path,
    audio_path: Path,
    stage: str,
    payload: Dict[str, object],
    source_fingerprint: Optional[Dict[str, object]] = None,
    stage_fingerprint: Optional[Dict[str, object]] = None,
    dependencies: Optional[List[Dict[str, object]]] = None,
):
    """Persist resumable intermediate data for a single heavy processing stage."""

    artifact_path = stage_artifact_path(output_dir, audio_path, stage)
    artifact_payload = {
        "artifact_version": STAGE_ARTIFACT_VERSION,
        "stage": stage,
        "audio_file": audio_path.name,
        "source_fingerprint": source_fingerprint or audio_file_fingerprint(audio_path),
        "stage_fingerprint": stage_fingerprint or {},
        "dependencies": list(dependencies or []),
        "payload": payload,
    }
    atomic_write_text(artifact_path, json.dumps(artifact_payload, indent=2))


def inspect_stage_artifact(
    output_dir: Path,
    audio_path: Path,
    stage: str,
    expected_stage_fingerprint: Optional[Dict[str, object]] = None,
    allow_legacy: bool = True,
) -> Dict[str, object]:
    artifact_path = stage_artifact_path(output_dir, audio_path, stage)
    if not artifact_path.exists():
        return {"reusable": False, "reason": "missing", "path": str(artifact_path), "payload": None}
    try:
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"reusable": False, "reason": "invalid_json", "path": str(artifact_path), "payload": None}
    if artifact_payload.get("source_fingerprint") != audio_file_fingerprint(audio_path):
        return {"reusable": False, "reason": "source_changed", "path": str(artifact_path), "payload": None}
    payload = artifact_payload.get("payload")
    if not isinstance(payload, dict):
        return {"reusable": False, "reason": "invalid_payload", "path": str(artifact_path), "payload": None}

    actual_stage_fingerprint = artifact_payload.get("stage_fingerprint")
    if expected_stage_fingerprint:
        if isinstance(actual_stage_fingerprint, dict) and actual_stage_fingerprint:
            if actual_stage_fingerprint.get("hash") != expected_stage_fingerprint.get("hash"):
                return {
                    "reusable": False,
                    "reason": "stage_fingerprint_changed",
                    "path": str(artifact_path),
                    "payload": None,
                    "actual_stage_fingerprint": actual_stage_fingerprint,
                }
        elif not allow_legacy:
            return {"reusable": False, "reason": "legacy_fingerprint_missing", "path": str(artifact_path), "payload": None}
        else:
            return {
                "reusable": True,
                "reason": "legacy_assumed_compatible",
                "path": str(artifact_path),
                "payload": payload,
                "stage_fingerprint": {},
            }
    expected_dependency_hashes = list(expected_stage_fingerprint.get("dependency_hashes") or []) if expected_stage_fingerprint else []
    actual_dependencies = artifact_payload.get("dependencies") if isinstance(artifact_payload.get("dependencies"), list) else []
    if expected_dependency_hashes:
        expected_hashes = [str(value) for value in expected_dependency_hashes]
        actual_hashes = [str(item.get("hash") or "") for item in actual_dependencies if isinstance(item, dict)]
        if actual_hashes != expected_hashes:
            return {"reusable": False, "reason": "dependency_fingerprint_changed", "path": str(artifact_path), "payload": None}
    return {
        "reusable": True,
        "reason": "fingerprint_match" if expected_stage_fingerprint else "source_match",
        "path": str(artifact_path),
        "payload": payload,
        "stage_fingerprint": actual_stage_fingerprint if isinstance(actual_stage_fingerprint, dict) else {},
        "dependencies": artifact_payload.get("dependencies") if isinstance(artifact_payload.get("dependencies"), list) else [],
    }


def load_stage_artifact(
    output_dir: Path,
    audio_path: Path,
    stage: str,
    expected_stage_fingerprint: Optional[Dict[str, object]] = None,
    allow_legacy: bool = True,
) -> Optional[Dict[str, object]]:
    inspection = inspect_stage_artifact(
        output_dir,
        audio_path,
        stage,
        expected_stage_fingerprint=expected_stage_fingerprint,
        allow_legacy=allow_legacy,
    )
    payload = inspection.get("payload")
    return payload if inspection.get("reusable") and isinstance(payload, dict) else None


def clear_stage_artifacts(output_dir: Path, audio_path: Path):
    artifact_dir = output_dir / ARTIFACT_DIRNAME / audio_path.stem
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def clear_debug_artifacts(output_dir: Path, audio_path: Path):
    artifact_dir = output_dir / ARTIFACT_DIRNAME / audio_path.stem
    if not artifact_dir.exists():
        return
    for child in artifact_dir.iterdir():
        if child.is_dir() and child.name in {"review_debug", "debug"}:
            shutil.rmtree(child)


def load_processed_files(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    processed = payload.get("processed_files", {})
    if isinstance(processed, dict):
        return {
            str(name): record
            for name, record in processed.items()
            if isinstance(name, str) and isinstance(record, dict)
        }

    if isinstance(processed, list):
        return {str(item): {} for item in processed if isinstance(item, str)}

    return {}


def save_processed_files(path: Path, processed_files: Dict[str, Dict[str, object]]):
    payload = {
        "processed_files": {
            name: processed_files[name]
            for name in sorted(processed_files)
        },
    }
    atomic_write_text(path, json.dumps(payload, indent=2))


def load_review_calibration_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_review_calibration_state(path: Path, payload: Dict[str, object]):
    atomic_write_text(path, json.dumps(payload, indent=2))


def load_diarization_history_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_diarization_history_state(path: Path, payload: Dict[str, object]):
    atomic_write_text(path, json.dumps(payload, indent=2))


def load_episode_summary_rows(path: Path, normalize_row) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            episode = row.get("episode")
            if episode:
                rows[episode] = normalize_row(row)
        return rows


def is_file_already_processed(
    audio_path: Path,
    output_dir: Path,
    processed_files: Dict[str, Dict[str, object]],
    existing_summary_rows: Dict[str, Dict[str, object]],
) -> bool:
    """Return true only when resume state, source fingerprint, summary row, and outputs agree."""

    expected_outputs = expected_output_paths(audio_path, output_dir)
    if not all(path.exists() for path in expected_outputs):
        return False

    record = processed_files.get(audio_path.name)
    if record is not None:
        if not record:
            return audio_path.name in existing_summary_rows
        return record == audio_file_fingerprint(audio_path)

    return audio_path.name in existing_summary_rows
