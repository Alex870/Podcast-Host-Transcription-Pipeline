"""M6 transcription resilience commands; no installation or packaging behavior."""

from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile, zipfile
from pathlib import Path
from typing import Any, Iterable
from .m6_preflight import build_preflight, write_report
from .operations import apply_retention, campaign_preflight

BACKUP_CONTRACT = "transcription-state-backup-1.0"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _id(value: Any, prefix: str) -> str:
    return (
        prefix
        + "_"
        + hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
    )


def _files(project: Path, output: Path) -> Iterable[tuple[str, Path]]:
    seen = set()
    for pattern in ("*.json", "*.toml", "*.yaml", "*.yml"):
        for path in project.glob(pattern):
            if path.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                yield "project/" + path.name, path
    for path in output.rglob("*") if output.exists() else []:
        if (
            path.is_file()
            and path.suffix.casefold() in {".json", ".jsonl", ".csv"}
            and not any(
                part in {"logs", "_review_audio", "_semantic_scans"}
                for part in path.relative_to(output).parts
            )
        ):
            yield "output/" + path.relative_to(output).as_posix(), path


def create_backup(project: Path, output: Path, destination: Path) -> dict[str, Any]:
    rows = []
    sources = list(_files(project.resolve(), output.resolve()))
    for name, path in sources:
        rows.append({"path": name, "sha256": _sha(path), "bytes": path.stat().st_size})
    manifest = {
        "contract_version": BACKUP_CONTRACT,
        "component": "podcast-host-transcription-pipeline",
        "entries": rows,
        "privacy": "local_explicit_backup_contains_operator_state",
    }
    manifest["backup_id"] = _id(manifest, "tx_backup")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        for name, path in sources:
            archive.write(path, name)
    return {**manifest, "path": str(destination), "file_count": len(rows)}


def inspect_backup(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        names = set(archive.namelist())
        if manifest.get("contract_version") != BACKUP_CONTRACT:
            raise ValueError(f"unsupported backup contract in {path}")
        errors = []
        for row in manifest.get("entries") or []:
            name = str(row["path"])
            if name not in names:
                errors.append(f"missing:{name}")
                continue
            if hashlib.sha256(archive.read(name)).hexdigest() != row["sha256"]:
                errors.append(f"checksum:{name}")
        return {
            "valid": not errors,
            "errors": errors,
            "backup_id": manifest.get("backup_id"),
            "entry_count": len(manifest.get("entries") or []),
        }


def restore_backup(
    path: Path, project: Path, output: Path, *, approved_backup_id: str
) -> dict[str, Any]:
    check = inspect_backup(path)
    if not check["valid"]:
        raise ValueError("backup validation failed: " + ", ".join(check["errors"]))
    if check["backup_id"] != approved_backup_id:
        raise PermissionError("approved backup identity does not match")
    restored = []
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for row in manifest["entries"]:
            name = str(row["path"])
            scope, relative = name.split("/", 1)
            root = project.resolve() if scope == "project" else output.resolve()
            target = (root / relative).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe backup member: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            restored.append(name)
    return {
        "backup_id": approved_backup_id,
        "restored": restored,
        "restored_count": len(restored),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="M6 transcription diagnostics and state recovery; never downloads models."
    )
    c = p.add_subparsers(dest="command", required=True)
    pre = c.add_parser("preflight")
    pre.add_argument("--workspace", required=True)
    pre.add_argument("--output")
    back = c.add_parser("backup")
    back.add_argument("--project", required=True)
    back.add_argument("--output-dir", required=True)
    back.add_argument("--destination", required=True)
    verify = c.add_parser("inspect-backup")
    verify.add_argument("path")
    restore = c.add_parser("restore")
    restore.add_argument("path")
    restore.add_argument("--project", required=True)
    restore.add_argument("--output-dir", required=True)
    restore.add_argument("--approve", required=True)
    capacity = c.add_parser("capacity")
    capacity.add_argument("--project", required=True)
    capacity.add_argument("--output-dir", required=True)
    retention = c.add_parser("retention")
    retention.add_argument("--output-dir", required=True)
    retention.add_argument("--categories", nargs="+", required=True)
    retention.add_argument("--older-than-days", type=int, default=30)
    retention.add_argument("--apply", action="store_true")
    a = p.parse_args(argv)
    if a.command == "preflight":
        value = build_preflight(
            "podcast-host-transcription-pipeline",
            Path(a.workspace),
            required_tools=("ffmpeg",),
            optional_tools=("node",),
            optional_modules=("torch", "faster_whisper", "pyannote.audio"),
        )
        write_report(Path(a.output), value) if a.output else None
    elif a.command == "backup":
        value = create_backup(Path(a.project), Path(a.output_dir), Path(a.destination))
    elif a.command == "inspect-backup":
        value = inspect_backup(Path(a.path))
    elif a.command == "restore":
        value = restore_backup(
            Path(a.path),
            Path(a.project),
            Path(a.output_dir),
            approved_backup_id=a.approve,
        )
    elif a.command == "capacity":
        value = campaign_preflight(Path(a.project), Path(a.output_dir))
    else:
        value = apply_retention(
            Path(a.output_dir),
            {"categories": a.categories, "older_than_days": a.older_than_days},
            dry_run=not a.apply,
        )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
