"""CLI for correction-manifest inspection, preview, and atomic application."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .ecosystem_contracts import apply_preview, inspect_file, preview_corrections


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_atomic(path: Path, value: dict) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="podcast-transcribe-corrections")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("manifest")
    preview = commands.add_parser("preview")
    preview.add_argument("transcript")
    preview.add_argument("corrections")
    preview.add_argument("--reviewer", required=True)
    preview.add_argument("--output", required=True)
    apply = commands.add_parser("apply")
    apply.add_argument("preview")
    apply.add_argument("transcript")
    apply.add_argument("--approve", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--manifest-output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_file(args.manifest), sort_keys=True))
        return 0
    if args.command == "preview":
        result = preview_corrections(
            _read(args.transcript), _read(args.corrections), reviewer=args.reviewer,
            producer={"name": "podcast-host-transcription-pipeline", "contract_version": "2"},
        )
        _write_atomic(Path(args.output), result)
        print(result["preview_id"])
        return 0
    preview_value = _read(args.preview)
    result, manifest = apply_preview(preview_value, _read(args.transcript), approved_preview_id=args.approve)
    _write_atomic(Path(args.output), result)
    _write_atomic(Path(args.manifest_output), manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
