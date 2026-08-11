"""Dependency-light cross-episode speaker review and safe-write helpers."""

from __future__ import annotations

import hashlib
import json
import datetime as dt
import shutil
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


def _normalized_vector(raw: object) -> Optional[List[float]]:
    if not isinstance(raw, list) or not raw:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values] if norm else None


def _vector_similarity(left: List[float], right: List[float]) -> float:
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


def collect_speaker_evidence(output_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cleaned_path in sorted(output_dir.glob("*_cleaned_speaker_transcript.json")):
        episode_id = cleaned_path.name[: -len("_cleaned_speaker_transcript.json")]
        cleaned = _load_json(cleaned_path)
        contract_evidence = cleaned.get("speaker_identity_evidence")
        if not isinstance(contract_evidence, list):
            metadata = cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else {}
            contract_evidence = metadata.get("speaker_identity_evidence") if isinstance(metadata.get("speaker_identity_evidence"), list) else []
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
                    "identity_evidence": next(
                        (
                            item for item in contract_evidence
                            if isinstance(item, dict)
                            and str(item.get("local_speaker") or "") == str(segment.get("original_speaker") or segment.get("speaker") or "")
                        ),
                        {},
                    ),
                }
            )
    return rows


def group_recurring_unknown_speakers(rows: List[Dict[str, object]], min_episode_count: int = 2) -> List[Dict[str, object]]:
    evidence_by_id: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if not _is_unknown(row.get("speaker")):
            continue
        evidence = row.get("identity_evidence")
        if not isinstance(evidence, dict) or not evidence.get("evidence_id") or not evidence.get("embedding"):
            continue
        evidence_by_id[str(evidence["evidence_id"])] = evidence
    families: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for evidence in evidence_by_id.values():
        families[str(evidence.get("embedding_family") or "unknown")].append(evidence)
    candidates: List[Dict[str, object]] = []
    for family, family_evidence in sorted(families.items()):
        clusters: List[List[Dict[str, object]]] = []
        for evidence in sorted(family_evidence, key=lambda item: str(item.get("evidence_id") or "")):
            vector = _normalized_vector(evidence.get("embedding"))
            if vector is None:
                continue
            evidence = {**evidence, "_vector": vector}
            compatible = []
            for index, cluster in enumerate(clusters):
                similarities = [
                    _vector_similarity(evidence["_vector"], member["_vector"])
                    for member in cluster
                ]
                if similarities and min(similarities) >= 0.72:
                    compatible.append((min(similarities), index))
            if compatible:
                clusters[max(compatible)[1]].append(evidence)
            else:
                clusters.append([evidence])
        for cluster in clusters:
            episodes = sorted({str(item.get("episode_id") or "") for item in cluster if item.get("episode_id")})
            if len(episodes) < min_episode_count:
                continue
            evidence_ids = sorted(str(item.get("evidence_id") or "") for item in cluster)
            candidate_id = "speaker_candidate_" + hashlib.sha256(
                json.dumps({"family": family, "evidence_ids": evidence_ids}, sort_keys=True).encode("utf-8")
            ).hexdigest()
            duration = sum(float(item.get("duration_seconds") or 0.0) for item in cluster)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "embedding_family": family,
                    "episode_count": len(episodes),
                    "episodes": episodes,
                    "total_duration_seconds": round(duration, 3),
                    "quality_score": round(
                        sum(float(item.get("quality_score") or 0.0) for item in cluster) / len(cluster), 4
                    ),
                    "promotion_eligible": len(episodes) >= 3 or duration >= 600.0,
                    "evidence_ids": evidence_ids,
                    "evidence_clips": [
                        {
                            "evidence_id": item.get("evidence_id"),
                            "episode_id": item.get("episode_id"),
                            "audio_path": item.get("source_audio"),
                            "spans": item.get("spans") or [],
                            "local_speaker": item.get("local_speaker"),
                        }
                        for item in cluster
                    ],
                }
            )
    return sorted(candidates, key=lambda item: (-int(item["episode_count"]), str(item["candidate_id"])))


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
        "workflow_version": 2,
        "view": normalized_view,
        "row_count": len(rows),
        "rows": rows,
        "recurring_unknown_speakers": group_recurring_unknown_speakers(collect_speaker_evidence(output_dir)),
        "changed_count": sum(bool(row["changed"]) for row in collect_speaker_evidence(output_dir)),
        "identity_basis": "embedding_evidence_complete_link",
        "similarity_threshold": 0.72,
    }


def load_speaker_library(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"speaker_schema_version": 2, "speakers": [], "revision_history": []}
    payload = _load_json(path)
    speakers = []
    for index, raw in enumerate(payload.get("speakers") or []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("display_name") or raw.get("name") or f"Speaker {index + 1}")
        speaker_id = str(raw.get("speaker_id") or "speaker_" + hashlib.sha256(name.casefold().encode("utf-8")).hexdigest())
        roles = raw.get("roles")
        if not isinstance(roles, list):
            roles = ["host"] if raw.get("is_host") else ["guest"]
        speakers.append(
            {
                **raw,
                "speaker_id": speaker_id,
                "display_name": name,
                "name": name,
                "aliases": [str(value) for value in raw.get("aliases") or []],
                "roles": [str(value) for value in roles],
                "is_host": "host" in roles or bool(raw.get("is_host")),
                "status": str(raw.get("status") or "active"),
                "reference_evidence": list(raw.get("reference_evidence") or []),
                "promotion_history": list(raw.get("promotion_history") or []),
            }
        )
    return {
        **payload,
        "speaker_schema_version": 2,
        "speakers": speakers,
        "revision_history": list(payload.get("revision_history") or []),
    }


def save_speaker_library(path: Path, payload: Dict[str, object], *, action: str, reviewer_id: str) -> Dict[str, object]:
    current = load_speaker_library(path)
    backup = path.with_name(path.name + ".bak")
    if path.exists():
        shutil.copy2(path, backup)
    result = {**payload, "speaker_schema_version": 2}
    history = list(result.get("revision_history") or current.get("revision_history") or [])
    history.append(
        {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "action": action,
            "reviewer_id": str(reviewer_id).strip() or "anonymous",
        }
    )
    result["revision_history"] = history
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    return {"status": "approved", "path": str(path), "backup_path": str(backup) if backup.exists() else ""}


def promote_speaker_candidate(
    path: Path,
    candidate: Dict[str, object],
    *,
    display_name: str,
    roles: List[str],
    aliases: Optional[List[str]] = None,
    reviewer_id: str,
) -> Dict[str, object]:
    if not bool(candidate.get("promotion_eligible")):
        raise RuntimeError("Speaker candidate does not meet the promotion evidence threshold.")
    library = load_speaker_library(path)
    family = str(candidate.get("embedding_family") or "")
    evidence_ids = [str(value) for value in candidate.get("evidence_ids") or []]
    identity = {"display_name": display_name, "embedding_family": family, "evidence_ids": evidence_ids}
    speaker_id = "speaker_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    library["speakers"].append(
        {
            "speaker_id": speaker_id,
            "display_name": str(display_name).strip(),
            "name": str(display_name).strip(),
            "aliases": sorted({str(value).strip() for value in aliases or [] if str(value).strip()}),
            "roles": sorted({str(value).strip() for value in roles if str(value).strip()}),
            "is_host": "host" in roles,
            "status": "active",
            "files": [],
            "embedding_family": family,
            "reference_evidence": evidence_ids,
            "promotion_history": [
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "reviewer_id": reviewer_id,
                    "approved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            ],
        }
    )
    return {
        **save_speaker_library(path, library, action="candidate_promoted", reviewer_id=reviewer_id),
        "speaker_id": speaker_id,
    }


def merge_speaker_identities(path: Path, speaker_ids: List[str], *, reviewer_id: str) -> Dict[str, object]:
    library = load_speaker_library(path)
    selected = [item for item in library["speakers"] if item.get("speaker_id") in set(speaker_ids)]
    if len(selected) < 2:
        raise RuntimeError("At least two speaker identities are required for a merge.")
    families = {str(item.get("embedding_family") or "") for item in selected if item.get("embedding_family")}
    if len(families) > 1:
        raise RuntimeError("Embedding-family mixing is not allowed.")
    survivor = selected[0]
    for item in selected[1:]:
        survivor["aliases"] = sorted(set(survivor.get("aliases") or []) | {str(item.get("display_name") or "")} | set(item.get("aliases") or []))
        survivor["roles"] = sorted(set(survivor.get("roles") or []) | set(item.get("roles") or []))
        survivor["reference_evidence"] = sorted(set(survivor.get("reference_evidence") or []) | set(item.get("reference_evidence") or []))
        item["status"] = "merged"
        item["merged_into"] = survivor["speaker_id"]
    return {
        **save_speaker_library(path, library, action="identities_merged", reviewer_id=reviewer_id),
        "survivor_speaker_id": survivor["speaker_id"],
    }


def split_speaker_identity(path: Path, speaker_id: str, evidence_ids: List[str], *, display_name: str, reviewer_id: str) -> Dict[str, object]:
    library = load_speaker_library(path)
    source = next((item for item in library["speakers"] if item.get("speaker_id") == speaker_id), None)
    if source is None:
        raise RuntimeError(f"Speaker identity not found: {speaker_id}")
    moved = set(evidence_ids)
    if not moved or not moved.issubset(set(source.get("reference_evidence") or [])):
        raise RuntimeError("Split evidence must be a non-empty subset of the source identity evidence.")
    source["reference_evidence"] = [value for value in source.get("reference_evidence") or [] if value not in moved]
    new_id = "speaker_" + hashlib.sha256(
        json.dumps({"source": speaker_id, "evidence": sorted(moved)}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    library["speakers"].append(
        {
            "speaker_id": new_id,
            "display_name": display_name,
            "name": display_name,
            "aliases": [],
            "roles": ["guest"],
            "is_host": False,
            "status": "active",
            "files": [],
            "embedding_family": source.get("embedding_family"),
            "reference_evidence": sorted(moved),
            "promotion_history": [{"split_from": speaker_id, "reviewer_id": reviewer_id}],
        }
    )
    return {
        **save_speaker_library(path, library, action="identity_split", reviewer_id=reviewer_id),
        "speaker_id": new_id,
    }


def rollback_speaker_library(path: Path, *, reviewer_id: str) -> Dict[str, object]:
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        raise RuntimeError("No speaker-library backup is available for rollback.")
    current = path.with_name(path.name + ".rollback-current")
    if path.exists():
        shutil.copy2(path, current)
    shutil.copy2(backup, path)
    return {
        "status": "rolled_back",
        "path": str(path),
        "reviewer_id": str(reviewer_id).strip() or "anonymous",
        "replaced_revision_path": str(current) if current.exists() else "",
    }
