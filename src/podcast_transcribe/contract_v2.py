"""Episode-contract-v2 upgrade and archival helpers.

The module stays dependency-light so classification and delta upgrades do not
initialize transcription, diarization, or review providers.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


EPISODE_CONTRACT_V2 = "episode-contract-v2"
CONTRACT_ARCHIVE_DIRNAME = "_contract_archive"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_episode_uid(source_file: str, source_fingerprint: Optional[Mapping[str, object]] = None) -> str:
    identity = {
        "source_filename": Path(source_file).name,
        "source_fingerprint": dict(source_fingerprint or {}),
    }
    return f"episode_{_sha256_payload(identity)}"


def episode_contract_status(cleaned_payload: Optional[Mapping[str, object]], manifest: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if not isinstance(cleaned_payload, Mapping):
        return {"status": "missing", "reason": "cleaned transcript is unavailable"}
    contract = str(cleaned_payload.get("contract_version") or "")
    manifest_contract = str((manifest or {}).get("contract_version") or "")
    if contract != EPISODE_CONTRACT_V2 or manifest_contract != EPISODE_CONTRACT_V2:
        return {"status": "v1", "reason": "episode artifacts do not declare episode-contract-v2"}
    required = (
        "episode_id",
        "episode_uid",
        "artifact_provenance",
        "completed_processing_stages",
        "speaker_identity_evidence_complete",
    )
    missing = [key for key in required if not cleaned_payload.get(key)]
    if missing:
        return {"status": "incomplete_v2", "reason": f"missing v2 evidence: {', '.join(missing)}"}
    return {"status": "v2_complete", "reason": "episode-contract-v2 evidence is complete"}


def _artifact_record(path: Path) -> Dict[str, object]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _completed_stages(manifest: Mapping[str, object], payload: Mapping[str, object]) -> list[str]:
    provenance = manifest.get("stage_provenance")
    if not isinstance(provenance, Mapping):
        metadata = payload.get("metadata")
        provenance = metadata.get("stage_provenance") if isinstance(metadata, Mapping) else {}
    stages = [str(key) for key, value in (provenance or {}).items() if isinstance(value, Mapping)]
    if payload.get("segments"):
        stages.extend(["output_write"])
    review = payload.get("review_metadata")
    if isinstance(review, Mapping):
        for item in review.get("review_stage_results") or []:
            if isinstance(item, Mapping) and str(item.get("status") or "") == "completed":
                stages.append(str(item.get("stage") or ""))
    return sorted({stage for stage in stages if stage})


def load_correction_lineage(output_dir: Path, episode_stem: str) -> Dict[str, object]:
    manifest_path = output_dir / "_correction_manifests" / f"{episode_stem}.correction-manifest-v2.json"
    if not manifest_path.exists():
        return {"correction_set_ids": [], "applied_correction_ids": []}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"correction_set_ids": [], "applied_correction_ids": []}
    corrections = payload.get("corrections") if isinstance(payload, Mapping) else []
    return {
        "correction_set_ids": [str(payload.get("correction_set_id") or "")],
        "applied_correction_ids": [
            str(item.get("correction_id") or "")
            for item in corrections or []
            if isinstance(item, Mapping) and str(item.get("status") or "") == "approved"
        ],
    }


def build_v2_payload(
    payload: Mapping[str, object],
    *,
    episode_stem: str,
    source_fingerprint: Mapping[str, object],
    manifest: Mapping[str, object],
    upgrade_method: str,
    output_dir: Path,
) -> Dict[str, object]:
    result = dict(payload)
    result["segments"] = [
        {
            **dict(segment),
            "source_span_id": str(segment.get("source_span_id", segment.get("id", ""))),
        }
        for segment in payload.get("segments") or []
        if isinstance(segment, Mapping)
    ]
    source_file = str(result.get("source_file") or manifest.get("source_file") or episode_stem)
    episode_uid = stable_episode_uid(source_file, source_fingerprint)
    result.update(
        {
            "contract_version": EPISODE_CONTRACT_V2,
            "episode_id": episode_stem,
            "episode_uid": episode_uid,
            "source_fingerprint": dict(source_fingerprint),
            "completed_processing_stages": _completed_stages(manifest, result),
            "correction_lineage": load_correction_lineage(output_dir, episode_stem),
            "speaker_identity_evidence": (
                result.get("speaker_identity_evidence")
                or ((result.get("metadata") or {}).get("speaker_identity_evidence") if isinstance(result.get("metadata"), Mapping) else [])
                or []
            ),
            "speaker_identity_evidence_complete": bool(
                result.get("speaker_identity_evidence_complete")
                or result.get("speaker_identity_evidence")
                or (
                    (result.get("metadata") or {}).get("speaker_identity_evidence_complete")
                    if isinstance(result.get("metadata"), Mapping)
                    else False
                )
                or (
                    (result.get("metadata") or {}).get("speaker_identity_evidence")
                    if isinstance(result.get("metadata"), Mapping)
                    else []
                )
                or not result.get("segments")
            ),
            "contract_upgrade": {
                "target_contract": EPISODE_CONTRACT_V2,
                "method": upgrade_method,
                "upgraded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_contract": str(payload.get("contract_version") or payload.get("schema_version") or "legacy"),
            },
        }
    )
    result["artifact_provenance"] = {
        "manifest_version": manifest.get("manifest_version"),
        "stage_provenance": manifest.get("stage_provenance") or ((result.get("metadata") or {}).get("stage_provenance") if isinstance(result.get("metadata"), Mapping) else {}),
        "source_manifest_hash": _sha256_payload(manifest) if manifest else "",
    }
    metadata = dict(result.get("metadata") or {})
    metadata["episode_contract_version"] = EPISODE_CONTRACT_V2
    metadata["episode_uid"] = episode_uid
    metadata["speaker_identity_evidence_complete"] = bool(result["speaker_identity_evidence_complete"])
    result["metadata"] = metadata
    return result


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def _archive_paths(output_dir: Path, episode_stem: str, paths: Iterable[Path]) -> Path:
    archive_root = output_dir / CONTRACT_ARCHIVE_DIRNAME / episode_stem / "v1"
    archive_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists() and path.is_file():
            target = archive_root / path.name
            if not target.exists():
                shutil.copy2(path, target)
    return archive_root


def archive_legacy_episode_bundle(audio_path: Path, output_dir: Path) -> Optional[Path]:
    stem = audio_path.stem
    paths = [
        *sorted(output_dir.glob(f"{stem}_*speaker_transcript.json")),
        output_dir / f"{stem}_manifest.json",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    current_cleaned = output_dir / f"{stem}_cleaned_speaker_transcript.json"
    current_manifest = output_dir / f"{stem}_manifest.json"
    try:
        cleaned_payload = json.loads(current_cleaned.read_text(encoding="utf-8-sig")) if current_cleaned.exists() else {}
        manifest_payload = json.loads(current_manifest.read_text(encoding="utf-8-sig")) if current_manifest.exists() else {}
    except json.JSONDecodeError:
        cleaned_payload, manifest_payload = {}, {}
    if episode_contract_status(cleaned_payload, manifest_payload)["status"] == "v2_complete":
        return None
    return _archive_paths(output_dir, stem, existing)


def upgrade_episode_bundle_v2(audio_path: Path, output_dir: Path, *, method: str = "delta_from_existing_outputs") -> Dict[str, object]:
    stem = audio_path.stem
    manifest_path = output_dir / f"{stem}_manifest.json"
    json_paths = sorted(output_dir.glob(f"{stem}_*speaker_transcript.json"))
    if not json_paths:
        raise RuntimeError(f"No transcript JSON artifacts are available for contract upgrade: {stem}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cannot upgrade corrupt manifest: {manifest_path}") from exc
    source_fingerprint = manifest.get("source_fingerprint")
    if not isinstance(source_fingerprint, Mapping):
        stat = audio_path.stat() if audio_path.exists() else None
        source_fingerprint = {
            "name": audio_path.name,
            "size_bytes": stat.st_size if stat else None,
            "mtime_ns": stat.st_mtime_ns if stat else None,
        }
    archive_root = _archive_paths(output_dir, stem, [*json_paths, manifest_path])
    upgraded_records = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cannot upgrade corrupt transcript JSON: {path}") from exc
        upgraded = build_v2_payload(
            payload,
            episode_stem=stem,
            source_fingerprint=source_fingerprint,
            manifest=manifest,
            upgrade_method=method,
            output_dir=output_dir,
        )
        _atomic_json(path, upgraded)
        upgraded_records.append(_artifact_record(path))
    manifest = dict(manifest)
    manifest.update(
        {
            "contract_version": EPISODE_CONTRACT_V2,
            "episode_id": stem,
            "episode_uid": stable_episode_uid(str(manifest.get("source_file") or audio_path), source_fingerprint),
            "contract_upgrade": {
                "method": method,
                "archive_path": str(archive_root),
                "upgraded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            "outputs": upgraded_records + [
                item for item in manifest.get("outputs") or []
                if isinstance(item, Mapping) and not str(item.get("filename") or "").endswith("_speaker_transcript.json")
            ],
        }
    )
    _atomic_json(manifest_path, manifest)
    return {
        "status": "v2_complete",
        "episode_id": stem,
        "upgrade_method": method,
        "archive_path": str(archive_root),
        "upgraded_files": [str(path) for path in json_paths],
    }
