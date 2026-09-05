"""Processing-space registry and partition-aware path resolution.

The transcription pipeline historically accepted one source and output folder.
This module adds a small, local SQLite registry so operators can manage several
independent processing spaces without hand-editing configuration files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PARTITION_REGISTRY_RELATIVE_PATH = Path("config") / "partitions.sqlite3"
PARTITION_SCHEMA_VERSION = 1
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma", ".webm"}
CONTEXT_TYPES = {"podcast", "meeting", "custom"}
WORKFLOW_PROFILES = {"podcast", "anonymous_meeting"}
PARTITION_STATUSES = {"discovered", "ready", "processing", "completed", "failed", "quarantined", "missing"}
SENSITIVE_CONFIG_KEY_FRAGMENTS = {"token", "secret", "password", "api_key", "apikey"}
PARTITION_RESERVED_OVERRIDE_KEYS = {
    "project_root",
    "partition",
    "partition_id",
    "input_dir",
    "output_dir",
    "default_source_dir",
    "default_output_dir",
    "state_dir",
    "corrections_dir",
    "known_speakers_dir",
    "speaker_reference_dir",
    "workflow_profile",
}


class PartitionError(RuntimeError):
    """Raised when a partition cannot be created, resolved, or safely used."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(_canonical_path(value)))


def _is_same_or_child(candidate: Path, parent: Path) -> bool:
    candidate_key = _path_key(candidate)
    parent_key = _path_key(parent)
    if candidate_key == parent_key:
        return True
    try:
        _canonical_path(candidate).relative_to(_canonical_path(parent))
        return True
    except ValueError:
        return False


def slugify(value: str) -> str:
    """Return a stable, Windows-safe slug suitable for a managed folder."""

    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "processing-space"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _validate_config_values(values: Mapping[str, Any]) -> None:
    for key in values:
        folded = str(key).strip().lower()
        if folded in PARTITION_RESERVED_OVERRIDE_KEYS:
            raise PartitionError(
                f"Partition-managed field cannot be overridden in processing-space settings: {key}"
            )
        if any(fragment in folded for fragment in SENSITIVE_CONFIG_KEY_FRAGMENTS):
            raise PartitionError(f"Secrets are not allowed in processing-space settings: {key}")


def load_project_config(project_root: Path) -> Dict[str, Any]:
    config_path = _canonical_path(project_root) / "podcast_transcribe_config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PartitionError(f"Invalid project configuration: {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PartitionError(f"Project configuration must contain an object: {config_path}")
    return payload


def _row_to_record(row: sqlite3.Row) -> "PartitionRecord":
    return PartitionRecord(
        partition_id=str(row["partition_id"]),
        display_name=str(row["display_name"]),
        slug=str(row["slug"]),
        context_type=str(row["context_type"]),
        workflow_profile=str(row["workflow_profile"]),
        intake_dir=Path(row["intake_dir"]),
        output_dir=Path(row["output_dir"]),
        state_dir=Path(row["state_dir"]),
        speaker_reference_dir=Path(row["speaker_reference_dir"]) if row["speaker_reference_dir"] else None,
        corrections_dir=Path(row["corrections_dir"]) if row["corrections_dir"] else None,
        config_overrides=json.loads(row["config_overrides_json"] or "{}"),
        downstream_config=json.loads(row["downstream_config_json"] or "{}"),
        archived=bool(row["archived"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        config_fingerprint=str(row["config_fingerprint"]),
    )


@dataclass
class PartitionRecord:
    partition_id: str
    display_name: str
    slug: str
    context_type: str
    workflow_profile: str
    intake_dir: Path
    output_dir: Path
    state_dir: Path
    speaker_reference_dir: Optional[Path] = None
    corrections_dir: Optional[Path] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    downstream_config: Dict[str, Any] = field(default_factory=dict)
    archived: bool = False
    created_at: str = ""
    updated_at: str = ""
    config_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("intake_dir", "output_dir", "state_dir", "speaker_reference_dir", "corrections_dir"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


@dataclass
class PartitionContext:
    project_root: Path
    registry_path: Path
    record: PartitionRecord
    effective_config: Dict[str, Any]

    @property
    def partition_id(self) -> str:
        return self.record.partition_id

    @property
    def input_dir(self) -> Path:
        return self.record.intake_dir

    @property
    def output_dir(self) -> Path:
        return self.record.output_dir

    def metadata(self) -> Dict[str, Any]:
        return {
            "partition_id": self.record.partition_id,
            "corpus_id": self.record.partition_id,
            "partition_display_name": self.record.display_name,
            "partition_slug": self.record.slug,
            "context_type": self.record.context_type,
            "workflow_profile": self.record.workflow_profile,
            "partition_config_fingerprint": self.record.config_fingerprint,
        }


class PartitionRegistry:
    """Transactional registry for processing spaces and intake status."""

    def __init__(self, project_root: str | Path):
        self.project_root = _canonical_path(project_root)
        self.path = self.project_root / PARTITION_REGISTRY_RELATIVE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registry_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS partitions (
                    partition_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    context_type TEXT NOT NULL,
                    workflow_profile TEXT NOT NULL,
                    intake_dir TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    state_dir TEXT NOT NULL,
                    speaker_reference_dir TEXT,
                    corrections_dir TEXT,
                    config_overrides_json TEXT NOT NULL DEFAULT '{}',
                    downstream_config_json TEXT NOT NULL DEFAULT '{}',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS partition_files (
                    partition_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    mtime_ns INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    run_id TEXT,
                    stage TEXT,
                    output_valid INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    first_seen_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    PRIMARY KEY (partition_id, source_path),
                    FOREIGN KEY (partition_id) REFERENCES partitions(partition_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS partition_runs (
                    run_id TEXT PRIMARY KEY,
                    partition_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    error TEXT,
                    FOREIGN KEY (partition_id) REFERENCES partitions(partition_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS partition_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    partition_id TEXT,
                    action TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_partition_runs_active
                    ON partition_runs(partition_id) WHERE status = 'running';
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO registry_meta(key, value) VALUES('schema_version', ?)",
                (str(PARTITION_SCHEMA_VERSION),),
            )

    def _validate_record_paths(
        self,
        intake_dir: Path,
        output_dir: Path,
        state_dir: Path,
        speaker_reference_dir: Optional[Path],
        corrections_dir: Optional[Path],
        partition_id: Optional[str] = None,
    ) -> None:
        intake_dir = _canonical_path(intake_dir)
        for label, path in (("output", output_dir), ("state", state_dir), ("speaker reference", speaker_reference_dir), ("corrections", corrections_dir)):
            if path is not None and _is_same_or_child(path, intake_dir):
                raise PartitionError(f"{label} directory must not be inside the intake directory: {path}")
        if _path_key(output_dir) == _path_key(state_dir):
            raise PartitionError("Output and state directories must be different.")

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT partition_id, intake_dir FROM partitions WHERE archived = 0 AND partition_id != ?",
                (partition_id or "",),
            ).fetchall()
        for row in rows:
            existing = Path(row["intake_dir"])
            if _is_same_or_child(intake_dir, existing) or _is_same_or_child(existing, intake_dir):
                raise PartitionError(
                    f"Intake directory overlaps active partition {row['partition_id']}: {existing}"
                )

    def _audit(self, connection: sqlite3.Connection, partition_id: Optional[str], action: str, detail: Mapping[str, Any] | None = None) -> None:
        connection.execute(
            "INSERT INTO partition_audit(partition_id, action, detail_json, created_at) VALUES(?, ?, ?, ?)",
            (partition_id, action, _json(dict(detail or {})), _utc_now()),
        )

    def list(self, include_archived: bool = False) -> List[PartitionRecord]:
        with self._connect() as connection:
            query = "SELECT * FROM partitions"
            if not include_archived:
                query += " WHERE archived = 0"
            query += " ORDER BY archived, display_name COLLATE NOCASE"
            return [_row_to_record(row) for row in connection.execute(query).fetchall()]

    def get(self, partition_id: str, include_archived: bool = False) -> PartitionRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM partitions WHERE partition_id = ?", (partition_id,)).fetchone()
        if row is None or (bool(row["archived"]) and not include_archived):
            raise PartitionError(f"Processing space not found: {partition_id}")
        return _row_to_record(row)

    def create(
        self,
        display_name: str,
        *,
        context_type: str = "podcast",
        workflow_profile: Optional[str] = None,
        intake_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        state_dir: str | Path | None = None,
        speaker_reference_dir: str | Path | None = None,
        corrections_dir: str | Path | None = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        downstream_config: Optional[Mapping[str, Any]] = None,
        create_directories: bool = True,
    ) -> PartitionRecord:
        display_name = str(display_name or "").strip()
        if not display_name:
            raise PartitionError("Processing space name is required.")
        context_type = str(context_type or "podcast").strip().lower()
        if context_type not in CONTEXT_TYPES:
            raise PartitionError(f"Unknown context type: {context_type}")
        if workflow_profile is None:
            workflow_profile = "anonymous_meeting" if context_type == "meeting" else "podcast"
        workflow_profile = str(workflow_profile).strip().lower()
        if workflow_profile not in WORKFLOW_PROFILES:
            raise PartitionError(f"Unknown workflow profile: {workflow_profile}")

        base_slug = slugify(display_name)
        existing_slugs = {record.slug for record in self.list(include_archived=True)}
        slug = base_slug
        suffix = 2
        while slug in existing_slugs:
            slug = f"{base_slug}-{suffix}"
            suffix += 1
        partition_root = self.project_root / "partitions" / slug
        intake = _canonical_path(intake_dir or partition_root / "intake")
        output = _canonical_path(output_dir or partition_root / "output")
        state = _canonical_path(state_dir or partition_root / "state")
        speakers = _canonical_path(speaker_reference_dir) if speaker_reference_dir else (
            _canonical_path(partition_root / "speaker-references") if workflow_profile == "podcast" else None
        )
        corrections = _canonical_path(corrections_dir or partition_root / "corrections")
        _validate_config_values(config_overrides or {})
        self._validate_record_paths(intake, output, state, speakers, corrections)
        now = _utc_now()
        record = PartitionRecord(
            partition_id=f"partition_{uuid.uuid4().hex}",
            display_name=display_name,
            slug=slug,
            context_type=context_type,
            workflow_profile=workflow_profile,
            intake_dir=intake,
            output_dir=output,
            state_dir=state,
            speaker_reference_dir=speakers,
            corrections_dir=corrections,
            config_overrides=dict(config_overrides or {}),
            downstream_config=dict(downstream_config or {}),
            created_at=now,
            updated_at=now,
        )
        record.config_fingerprint = _fingerprint({
            "context_type": record.context_type,
            "workflow_profile": record.workflow_profile,
            "intake_dir": str(record.intake_dir),
            "output_dir": str(record.output_dir),
            "state_dir": str(record.state_dir),
            "speaker_reference_dir": str(record.speaker_reference_dir or ""),
            "corrections_dir": str(record.corrections_dir or ""),
            "config_overrides": record.config_overrides,
            "downstream_config": record.downstream_config,
        })
        if create_directories:
            for directory in (record.intake_dir, record.output_dir, record.state_dir, record.speaker_reference_dir, record.corrections_dir):
                if directory:
                    directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO partitions(
                    partition_id, display_name, slug, context_type, workflow_profile,
                    intake_dir, output_dir, state_dir, speaker_reference_dir, corrections_dir,
                    config_overrides_json, downstream_config_json, archived, created_at, updated_at,
                    config_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    record.partition_id, record.display_name, record.slug, record.context_type,
                    record.workflow_profile, str(record.intake_dir), str(record.output_dir), str(record.state_dir),
                    str(record.speaker_reference_dir) if record.speaker_reference_dir else None,
                    str(record.corrections_dir) if record.corrections_dir else None,
                    _json(record.config_overrides), _json(record.downstream_config), record.created_at,
                    record.updated_at, record.config_fingerprint,
                ),
            )
            self._audit(connection, record.partition_id, "created", record.to_dict())
        return record

    def update(self, partition_id: str, **changes: Any) -> PartitionRecord:
        current = self.get(partition_id, include_archived=True)
        allowed = {
            "display_name", "context_type", "workflow_profile", "intake_dir", "output_dir", "state_dir",
            "speaker_reference_dir", "corrections_dir", "config_overrides", "downstream_config", "archived",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise PartitionError(f"Unsupported processing-space fields: {', '.join(unknown)}")
        values = current.to_dict()
        values.update({key: value for key, value in changes.items() if value is not None})
        context_type = str(values["context_type"]).lower()
        workflow_profile = str(values["workflow_profile"]).lower()
        if context_type not in CONTEXT_TYPES or workflow_profile not in WORKFLOW_PROFILES:
            raise PartitionError("Invalid context type or workflow profile.")
        intake = _canonical_path(values["intake_dir"])
        output = _canonical_path(values["output_dir"])
        state = _canonical_path(values["state_dir"])
        speakers = _canonical_path(values["speaker_reference_dir"]) if values.get("speaker_reference_dir") else None
        corrections = _canonical_path(values["corrections_dir"]) if values.get("corrections_dir") else None
        _validate_config_values(values.get("config_overrides") or {})
        self._validate_record_paths(intake, output, state, speakers, corrections, partition_id=partition_id)
        for directory in (intake, output, state, speakers, corrections):
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
        updated = PartitionRecord(
            partition_id=current.partition_id,
            display_name=str(values["display_name"]),
            slug=current.slug,
            context_type=context_type,
            workflow_profile=workflow_profile,
            intake_dir=intake,
            output_dir=output,
            state_dir=state,
            speaker_reference_dir=speakers,
            corrections_dir=corrections,
            config_overrides=dict(values.get("config_overrides") or {}),
            downstream_config=dict(values.get("downstream_config") or {}),
            archived=bool(values.get("archived")),
            created_at=current.created_at,
            updated_at=_utc_now(),
        )
        updated.config_fingerprint = _fingerprint({
            "context_type": updated.context_type,
            "workflow_profile": updated.workflow_profile,
            "intake_dir": str(updated.intake_dir),
            "output_dir": str(updated.output_dir),
            "state_dir": str(updated.state_dir),
            "speaker_reference_dir": str(updated.speaker_reference_dir or ""),
            "corrections_dir": str(updated.corrections_dir or ""),
            "config_overrides": updated.config_overrides,
            "downstream_config": updated.downstream_config,
        })
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE partitions SET display_name=?, context_type=?, workflow_profile=?, intake_dir=?, output_dir=?,
                    state_dir=?, speaker_reference_dir=?, corrections_dir=?, config_overrides_json=?,
                    downstream_config_json=?, archived=?, updated_at=?, config_fingerprint=?
                WHERE partition_id=?
                """,
                (
                    updated.display_name, updated.context_type, updated.workflow_profile, str(updated.intake_dir),
                    str(updated.output_dir), str(updated.state_dir), str(updated.speaker_reference_dir or ""),
                    str(updated.corrections_dir or ""), _json(updated.config_overrides), _json(updated.downstream_config),
                    int(updated.archived), updated.updated_at, updated.config_fingerprint, updated.partition_id,
                ),
            )
            self._audit(connection, partition_id, "updated", changes)
        return updated

    def backup(self) -> Path:
        backup_path = self.path.with_name(f"{self.path.stem}.backup-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}{self.path.suffix}")
        # Copy through SQLite's backup API so committed data in WAL mode is
        # included even when the sidecar files have not been checkpointed.
        with self._connect() as source:
            with sqlite3.connect(backup_path) as destination:
                source.backup(destination)
        shutil.copystat(self.path, backup_path, follow_symlinks=True)
        return backup_path

    def scan(self, partition_id: str, *, include_archived: bool = False) -> Dict[str, Any]:
        record = self.get(partition_id, include_archived=include_archived)
        record.intake_dir.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        observed: set[str] = set()
        with self._connect() as connection:
            for path in sorted(record.intake_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                    continue
                path = _canonical_path(path)
                key = str(path)
                observed.add(key)
                stat = path.stat()
                row = connection.execute(
                    "SELECT * FROM partition_files WHERE partition_id=? AND source_path=?",
                    (partition_id, key),
                ).fetchone()
                changed = row is None or int(row["size_bytes"]) != int(stat.st_size) or int(row["mtime_ns"]) != int(stat.st_mtime_ns)
                status = "ready" if changed or row is None or row["status"] in {"failed", "missing", "quarantined"} else str(row["status"])
                if row is not None and row["status"] == "completed" and not changed:
                    status = "completed"
                connection.execute(
                    """
                    INSERT INTO partition_files(partition_id, source_path, source_name, size_bytes, mtime_ns, status, run_id, stage, output_valid, last_error, first_seen_at, updated_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?)
                    ON CONFLICT(partition_id, source_path) DO UPDATE SET source_name=excluded.source_name,
                        size_bytes=excluded.size_bytes, mtime_ns=excluded.mtime_ns, status=excluded.status,
                        updated_at=excluded.updated_at, completed_at=CASE WHEN excluded.status='completed' THEN partition_files.completed_at ELSE NULL END
                    """,
                    (
                        partition_id, key, path.name, int(stat.st_size), int(stat.st_mtime_ns), status,
                        int(status == "completed"), row["first_seen_at"] if row else now, now,
                        row["completed_at"] if row and status == "completed" else None,
                    ),
                )
            existing = connection.execute("SELECT source_path FROM partition_files WHERE partition_id=?", (partition_id,)).fetchall()
            for row in existing:
                if str(row["source_path"]) not in observed:
                    connection.execute(
                        "UPDATE partition_files SET status='missing', updated_at=? WHERE partition_id=? AND source_path=? AND status != 'processing'",
                        (now, partition_id, row["source_path"]),
                    )
            rows = connection.execute(
                "SELECT * FROM partition_files WHERE partition_id=? ORDER BY source_name COLLATE NOCASE",
                (partition_id,),
            ).fetchall()
        counts: Dict[str, int] = {}
        files = []
        for row in rows:
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + 1
            files.append({
                "source_path": row["source_path"], "source_name": row["source_name"], "status": status,
                "size_bytes": int(row["size_bytes"]), "mtime_ns": int(row["mtime_ns"]),
                "stage": row["stage"] or "", "last_error": row["last_error"] or "",
                "completed_at": row["completed_at"] or "",
            })
        return {"partition": record.to_dict(), "files": files, "counts": counts, "scanned_at": now}

    def mark_file(self, partition_id: str, source_path: str | Path, status: str, *, run_id: str = "", stage: str = "", error: str = "", output_valid: bool = False) -> None:
        if status not in PARTITION_STATUSES:
            raise PartitionError(f"Unknown intake status: {status}")
        path = _canonical_path(source_path)
        now = _utc_now()
        stat = path.stat() if path.exists() else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO partition_files(partition_id, source_path, source_name, size_bytes, mtime_ns, status, run_id, stage, output_valid, last_error, first_seen_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(partition_id, source_path) DO UPDATE SET size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns, status=excluded.status, run_id=excluded.run_id, stage=excluded.stage,
                    output_valid=excluded.output_valid, last_error=excluded.last_error, updated_at=excluded.updated_at,
                    completed_at=excluded.completed_at
                """,
                (
                    partition_id, str(path), path.name, int(stat.st_size) if stat else 0,
                    int(stat.st_mtime_ns) if stat else 0, status, run_id or None, stage or None,
                    int(output_valid), error or None, now, now, now if status == "completed" else None,
                ),
            )

    def start_run(self, partition_id: str) -> str:
        run_id = f"run_{uuid.uuid4().hex}"
        with self._connect() as connection:
            active = connection.execute(
                "SELECT run_id FROM partition_runs WHERE partition_id=? AND status='running' LIMIT 1",
                (partition_id,),
            ).fetchone()
            if active is not None:
                raise PartitionError(
                    f"Processing space {partition_id} already has an active run: {active['run_id']}"
                )
            try:
                connection.execute(
                    "INSERT INTO partition_runs(run_id, partition_id, status, started_at) VALUES (?, ?, 'running', ?)",
                    (run_id, partition_id, _utc_now()),
                )
            except sqlite3.IntegrityError as exc:
                raise PartitionError(f"Processing space {partition_id} already has an active run.") from exc
            self._audit(connection, partition_id, "run_started", {"run_id": run_id})
        return run_id

    def finish_run(self, run_id: str, status: str = "completed", error: str = "") -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT partition_id FROM partition_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise PartitionError(f"Run not found: {run_id}")
            connection.execute(
                "UPDATE partition_runs SET status=?, ended_at=?, error=? WHERE run_id=?",
                (status, _utc_now(), error or None, run_id),
            )
            self._audit(connection, row["partition_id"], "run_finished", {"run_id": run_id, "status": status, "error": error})

    def summary(self, partition_id: str, *, include_archived: bool = False) -> Dict[str, Any]:
        return self.scan(partition_id, include_archived=include_archived)


def resolve_partition_context(project_root: str | Path, partition_id: str, global_config: Optional[Mapping[str, Any]] = None) -> PartitionContext:
    project = _canonical_path(project_root)
    registry = PartitionRegistry(project)
    record = registry.get(partition_id)
    effective = dict(global_config or load_project_config(project))
    effective.update(record.config_overrides)
    effective.update(
        {
            "default_source_dir": str(record.intake_dir),
            "default_output_dir": str(record.output_dir),
            "workflow_profile": record.workflow_profile,
            "corrections_dir": str(record.corrections_dir) if record.corrections_dir else "",
            "known_speakers_dir": str(record.speaker_reference_dir) if record.speaker_reference_dir else "",
            "partition_id": record.partition_id,
            "partition_display_name": record.display_name,
            "partition_slug": record.slug,
            "context_type": record.context_type,
            "partition_config_fingerprint": record.config_fingerprint,
            "partition_downstream_config": record.downstream_config,
        }
    )
    return PartitionContext(project_root=project, registry_path=registry.path, record=record, effective_config=effective)


def ensure_legacy_partition(project_root: str | Path, *, display_name: str = "Legacy default") -> PartitionRecord:
    """Adopt the existing configured source/output pair without moving data."""

    project = _canonical_path(project_root)
    registry = PartitionRegistry(project)
    config = load_project_config(project)
    source = config.get("default_source_dir") or project / "source"
    source_path = _canonical_path(source if Path(str(source)).is_absolute() else project / str(source))
    output = config.get("default_output_dir") or source_path.parent / "output"
    output_path = _canonical_path(output if Path(str(output)).is_absolute() else project / str(output))
    existing = [
        record
        for record in registry.list(include_archived=True)
        if record.slug == "legacy-default"
        or (_path_key(record.intake_dir) == _path_key(source_path) and _path_key(record.output_dir) == _path_key(output_path))
    ]
    if existing:
        return existing[0]
    known = config.get("known_speakers_dir")
    corrections = config.get("corrections_dir")
    record = registry.create(
        display_name,
        context_type="podcast",
        workflow_profile=str(config.get("workflow_profile") or "podcast"),
        intake_dir=source_path,
        output_dir=output_path,
        state_dir=output_path / "_partition_state",
        speaker_reference_dir=known if known else None,
        corrections_dir=corrections if corrections else output_path / "_partition_corrections",
        config_overrides={},
        create_directories=True,
    )
    return record


def partition_context_from_args(args: Any) -> Optional[PartitionContext]:
    partition_id = str(getattr(args, "partition", "") or "").strip()
    if not partition_id:
        return None
    project_root = str(getattr(args, "project_root", "") or "").strip()
    project = _canonical_path(project_root) if project_root else Path(__file__).resolve().parents[2]
    return resolve_partition_context(project, partition_id)


def apply_partition_to_args(args: Any) -> Optional[PartitionContext]:
    context = partition_context_from_args(args)
    if context is None:
        return None
    path_keys = {
        "provider_cache_dir",
        "host_reference",
        "host_profile_json",
        "preferred_terms_file",
        "replacement_map_json",
        "review_debug_dir",
        "evaluation_pack_path",
        "gold_set_dir",
    }
    for key, value in context.effective_config.items():
        if not hasattr(args, key) or value in (None, ""):
            continue
        option = "--" + key.replace("_", "-")
        if option in sys.argv:
            continue
        resolved_value = value
        if key in path_keys and isinstance(value, str) and value and not Path(value).is_absolute():
            resolved_value = str(context.project_root / value)
        setattr(args, key, resolved_value)
    args.project_root = str(context.project_root)
    args.input_dir = str(context.input_dir)
    args.output_dir = str(context.output_dir)
    args.workflow_profile = context.record.workflow_profile
    if context.record.corrections_dir:
        args.corrections_dir = str(context.record.corrections_dir)
    if context.record.speaker_reference_dir:
        args.known_speakers_dir = str(context.record.speaker_reference_dir)
    for key, value in context.record.config_overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.partition_context = context
    return context


def partition_manager(project_root: str | Path) -> None:
    """Small terminal manager used by the launcher and recovery workflows."""

    registry = PartitionRegistry(project_root)

    def select_record(records: List[PartitionRecord]) -> Optional[PartitionRecord]:
        selected = input("Partition number or ID: ").strip()
        return next(
            (item for index, item in enumerate(records, start=1) if str(index) == selected or item.partition_id == selected),
            None,
        )

    while True:
        print("\nProcessing spaces")
        records = registry.list(include_archived=True)
        if records:
            for index, record in enumerate(records, start=1):
                state = "archived" if record.archived else "active"
                print(f"  {index}. {record.display_name} [{state}] ({record.partition_id})")
        else:
            print("  No processing spaces have been created.")
        print("  C. Create processing space")
        print("  A. Adopt legacy source/output folders")
        print("  E. Edit a processing space")
        print("  R. Reactivate an archived space")
        print("  V. Validate a processing space")
        print("  S. Scan a processing space")
        print("  P. Process a processing space")
        print("  B. Backup registry")
        print("  Q. Return")
        choice = input("Select an option: ").strip().upper()
        if choice == "Q":
            return
        if choice == "B":
            print(f"Registry backup: {registry.backup()}")
            continue
        if choice == "C":
            name = input("Name: ").strip()
            context_type = input("Context (podcast, meeting, custom) [podcast]: ").strip().lower() or "podcast"
            intake = input("Intake folder [managed default]: ").strip() or None
            output = input("Output folder [managed default]: ").strip() or None
            record = registry.create(name, context_type=context_type, intake_dir=intake, output_dir=output)
            print(f"Created {record.display_name}: {record.partition_id}")
            continue
        if choice == "A":
            name = input("Name [Legacy default]: ").strip() or "Legacy default"
            record = ensure_legacy_partition(project_root, display_name=name)
            print(f"Adopted folders as {record.display_name}: {record.partition_id}")
            continue
        if choice == "S":
            record = select_record(records)
            if record is None:
                print("Unknown processing space.")
                continue
            result = registry.scan(record.partition_id)
            print(json.dumps({"partition": record.display_name, "counts": result["counts"], "files": result["files"]}, indent=2))
            continue
        if choice == "P":
            record = select_record(records)
            if record is None or record.archived:
                print("Unknown or archived processing space.")
                continue
            wrapper = Path(__file__).resolve().parents[2] / "podcast_transcribe_host.py"
            command = [sys.executable, str(wrapper), "--partition", record.partition_id, "--project-root", str(project_root)]
            print(f"Starting processing space: {record.display_name}")
            result = subprocess.run(command)
            print(f"Processing space exited with code {result.returncode}.")
            continue
        if choice == "V":
            record = select_record(records)
            if record is None:
                print("Unknown processing space.")
                continue
            result = registry.summary(record.partition_id)
            missing = [str(path) for path in (record.intake_dir, record.output_dir, record.state_dir) if not path.exists()]
            print(json.dumps({"valid": not missing and not record.archived, "missing_paths": missing, "counts": result["counts"]}, indent=2))
            continue
        if choice == "R":
            record = select_record(records)
            if record is None:
                print("Unknown processing space.")
                continue
            updated = registry.update(record.partition_id, archived=False)
            print(f"Reactivated {updated.display_name}.")
            continue
        if choice == "E":
            record = select_record(records)
            if record is None:
                print("Unknown processing space.")
                continue
            updated = registry.update(
                record.partition_id,
                display_name=input(f"Name [{record.display_name}]: ").strip() or record.display_name,
                context_type=input(f"Context [{record.context_type}]: ").strip().lower() or record.context_type,
                workflow_profile=input(f"Workflow [{record.workflow_profile}]: ").strip().lower() or record.workflow_profile,
                intake_dir=input(f"Intake [{record.intake_dir}]: ").strip() or str(record.intake_dir),
                output_dir=input(f"Output [{record.output_dir}]: ").strip() or str(record.output_dir),
                state_dir=input(f"State [{record.state_dir}]: ").strip() or str(record.state_dir),
                speaker_reference_dir=input(f"Speaker references [{record.speaker_reference_dir or ''}]: ").strip() or str(record.speaker_reference_dir or ""),
                corrections_dir=input(f"Corrections [{record.corrections_dir or ''}]: ").strip() or str(record.corrections_dir or ""),
            )
            print(f"Updated {updated.display_name}.")
            continue
        print("Unrecognized option.")
