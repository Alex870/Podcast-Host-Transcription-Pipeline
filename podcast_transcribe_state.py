import csv
import json
from pathlib import Path
from typing import Dict, List


RESUME_STATE_FILENAME = "_processed_files.json"
SUMMARY_FILENAME = "_episode_review_summary.csv"
CHECKPOINT_DIRNAME = "_processing_checkpoints"


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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    expected_outputs = expected_output_paths(audio_path, output_dir)
    if not all(path.exists() for path in expected_outputs):
        return False

    record = processed_files.get(audio_path.name)
    if record is not None:
        if not record:
            return audio_path.name in existing_summary_rows
        return record == audio_file_fingerprint(audio_path)

    return audio_path.name in existing_summary_rows
