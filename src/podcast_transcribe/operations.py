"""Operator-facing campaign, downstream delivery, and retention operations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _identity(value: Mapping[str, Any], prefix: str) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def campaign_preflight(project_root: Path, output_dir: Path) -> dict[str, Any]:
    """Return deterministic capacity, risk, review, and delivery diagnostics."""
    from .workbench_core import discover_episode_bundles, evaluation_queues

    episodes = discover_episode_bundles(output_dir)
    queues = evaluation_queues(project_root, output_dir)
    rows = []
    total_bytes = 0
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or episode.get("id") or "")
        candidates = list(output_dir.glob(f"{episode_id}*")) if episode_id else []
        size = sum(path.stat().st_size for path in candidates if path.is_file())
        total_bytes += size
        duration = float(episode.get("duration_seconds") or episode.get("duration") or 0.0)
        issue_count = int(episode.get("issue_count") or episode.get("finding_count") or 0)
        reviewed = bool(episode.get("human_approved") or episode.get("reviewed"))
        risk = issue_count * 4 + min(int(duration // 1800), 4) * 2 + (0 if reviewed else 5)
        rows.append({
            "episode_id": episode_id,
            "duration_seconds": duration,
            "artifact_bytes": size,
            "issue_count": issue_count,
            "human_approved": reviewed,
            "risk_score": risk,
            "risk_reasons": [
                reason for condition, reason in (
                    (not reviewed, "human review incomplete"),
                    (issue_count > 0, f"{issue_count} review findings"),
                    (duration >= 3600, "long episode"),
                ) if condition
            ],
        })
    rows.sort(key=lambda item: (-item["risk_score"], item["episode_id"]))
    deliveries = downstream_delivery_status(output_dir)
    payload = {
        "contract_version": "transcription-operations-v1",
        "episode_count": len(rows),
        "artifact_bytes": total_bytes,
        "estimated_additional_bytes": max(total_bytes, 1),
        "risk_order": rows,
        "evaluation_queues": queues,
        "downstream": deliveries,
        "blockers": [
            *([] if rows else ["no episode bundles found"]),
            *([] if deliveries["failed_count"] == 0 else ["downstream delivery failures require retry"]),
        ],
    }
    payload["preflight_id"] = _identity(payload, "transcription_preflight")
    return payload


def downstream_delivery_status(output_dir: Path) -> dict[str, Any]:
    root = output_dir / "_downstream_corrections"
    events = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                events.append({
                    "correction_set_id": value.get("correction_set_id"),
                    "status": value.get("status", "downstream_pending"),
                    "missing_consumers": value.get("missing_consumers", []),
                    "delivery_errors": value.get("delivery_errors", []),
                    "retry_endpoint": f"/api/operations/downstream/{value.get('correction_set_id')}/retry",
                })
            except (OSError, ValueError) as exc:
                events.append({"status": "downstream_failed", "error": str(exc), "file": path.name})
    return {
        "events": events,
        "pending_count": sum(item.get("status") == "downstream_pending" for item in events),
        "failed_count": sum(item.get("status") == "downstream_failed" for item in events),
    }


def retry_downstream_delivery(project_root: Path, output_dir: Path, correction_set_id: str) -> dict[str, Any]:
    from .workbench_core import load_project_config

    pending_path = output_dir / "_downstream_corrections" / f"{correction_set_id}.json"
    if not pending_path.is_file():
        raise RuntimeError(f"Downstream correction event was not found: {correction_set_id}")
    event = json.loads(pending_path.read_text(encoding="utf-8"))
    config = load_project_config(project_root)
    delivered, missing, errors = [], [], []
    for key in ("podcast_rag_project_dir", "ragscope_project_dir"):
        raw = str(config.get(key) or "").strip()
        if not raw:
            missing.append(key)
            continue
        root = Path(raw)
        if not root.is_absolute():
            root = (project_root / root).resolve()
        if not root.exists():
            missing.append(key)
            continue
        target = root / "state" / "transcription_corrections" / f"{correction_set_id}.json"
        try:
            _atomic_json(target, {key: value for key, value in event.items() if key not in {"delivered_paths", "delivery_errors", "missing_consumers", "status"}})
            delivered.append(str(target))
        except OSError as exc:
            errors.append({"consumer": key, "error": str(exc)})
    event.update({
        "status": "downstream_failed" if errors else "delivered" if delivered and not missing else "downstream_pending",
        "delivered_paths": delivered,
        "missing_consumers": missing,
        "delivery_errors": errors,
        "last_attempt_epoch_ms": int(time.time() * 1000),
    })
    _atomic_json(pending_path, event)
    return event


def apply_retention(output_dir: Path, policy: Mapping[str, Any], *, dry_run: bool = True) -> dict[str, Any]:
    """Preview or apply explicit retention to non-authoritative runtime artifacts."""
    allowed = {
        "semantic_scans": output_dir / "_semantic_scans",
        "temporary_audio": output_dir / "_review_audio",
        "logs": output_dir / "logs",
    }
    categories = set(map(str, policy.get("categories") or []))
    unknown = categories.difference(allowed)
    if unknown:
        raise RuntimeError(f"Unsupported retention categories: {', '.join(sorted(unknown))}")
    older_than_days = max(0, int(policy.get("older_than_days", 30)))
    cutoff = time.time() - older_than_days * 86400
    candidates = []
    for category in sorted(categories):
        root = allowed[category]
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if older_than_days == 0 or path.stat().st_mtime < cutoff:
                candidates.append({"category": category, "relative_path": path.relative_to(output_dir).as_posix(), "bytes": path.stat().st_size})
                if not dry_run:
                    path.unlink(missing_ok=True)
    return {"dry_run": dry_run, "older_than_days": older_than_days, "candidates": candidates, "deleted_count": 0 if dry_run else len(candidates)}
