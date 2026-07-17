"""Versioned correction contracts used at the ecosystem boundary.

This module deliberately has no dependencies on the transcription runtime.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


CORRECTION_CONTRACT = "correction-manifest-v1"
MUTABLE_IDENTITY_KEYS = {"notes", "display_label", "display_labels", "ui_state"}


class ContractError(ValueError):
    """Raised when an ecosystem contract is invalid or stale."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_id(value: Any, *, prefix: str) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_json(value)).hexdigest()}"


def transcript_hash(transcript: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(transcript)).hexdigest()


def correction_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in manifest.items()
        if key not in MUTABLE_IDENTITY_KEYS and key != "correction_set_id"
    }


def build_correction_manifest(
    transcript: Mapping[str, Any],
    corrections: Iterable[Mapping[str, Any]],
    *,
    reviewer: str,
    producer: Mapping[str, str],
    result_transcript: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = [deepcopy(dict(item)) for item in corrections if item.get("adjudication_state") == "accepted"]
    result = result_transcript if result_transcript is not None else apply_corrections(transcript, accepted)
    manifest: dict[str, Any] = {
        "contract_version": CORRECTION_CONTRACT,
        "producer": dict(producer),
        "source_transcript_hash": transcript_hash(transcript),
        "result_transcript_hash": transcript_hash(result),
        "reviewer_pseudonym": reviewer,
        "accepted_corrections": accepted,
    }
    manifest["correction_set_id"] = canonical_id(correction_identity_payload(manifest), prefix="correction")
    validate_correction_manifest(manifest, transcript)
    return manifest


def _segments(transcript: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = transcript.get("segments")
    if not isinstance(value, list):
        raise ContractError("transcript must contain a segments list")
    return value


def _find_segment(transcript: Mapping[str, Any], source_span_id: str) -> dict[str, Any]:
    for segment in _segments(transcript):
        if str(segment.get("source_span_id", segment.get("id", ""))) == source_span_id:
            return segment
    raise ContractError(f"unknown source_span_id: {source_span_id}")


def validate_correction_manifest(manifest: Mapping[str, Any], transcript: Mapping[str, Any]) -> None:
    if manifest.get("contract_version") != CORRECTION_CONTRACT:
        raise ContractError("unsupported correction contract")
    if manifest.get("source_transcript_hash") != transcript_hash(transcript):
        raise ContractError("stale source transcript hash")
    expected_id = canonical_id(correction_identity_payload(manifest), prefix="correction")
    if manifest.get("correction_set_id") != expected_id:
        raise ContractError("correction-set identity mismatch")
    for correction in manifest.get("accepted_corrections", []):
        required = {"source_span_id", "field", "before", "after", "reason_code", "adjudication_state"}
        missing = required - correction.keys()
        if missing:
            raise ContractError(f"correction missing fields: {sorted(missing)}")
        if correction["adjudication_state"] != "accepted":
            raise ContractError("authoritative correction set may only contain accepted corrections")
        segment = _find_segment(transcript, str(correction["source_span_id"]))
        if segment.get(str(correction["field"])) != correction["before"]:
            raise ContractError(f"before value mismatch for {correction['source_span_id']}")


def apply_corrections(transcript: Mapping[str, Any], corrections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result = deepcopy(dict(transcript))
    for correction in corrections:
        segment = _find_segment(result, str(correction["source_span_id"]))
        field = str(correction["field"])
        if segment.get(field) != correction["before"]:
            raise ContractError(f"before value mismatch for {correction['source_span_id']}")
        segment[field] = correction["after"]
    return result


def preview_corrections(
    transcript: Mapping[str, Any], corrections: Iterable[Mapping[str, Any]], *, reviewer: str, producer: Mapping[str, str]
) -> dict[str, Any]:
    correction_list = [dict(item) for item in corrections]
    manifest = build_correction_manifest(transcript, correction_list, reviewer=reviewer, producer=producer)
    return {
        "preview_id": canonical_id({"manifest": manifest, "targets": ["reviewed-transcript"]}, prefix="preview"),
        "manifest": manifest,
        "target_files": ["reviewed-transcript"],
        "downstream_impact_required": True,
    }


def apply_preview(
    preview: Mapping[str, Any], transcript: Mapping[str, Any], *, approved_preview_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if preview.get("preview_id") != approved_preview_id:
        raise ContractError("human approval does not match preview identity")
    manifest = dict(preview["manifest"])
    validate_correction_manifest(manifest, transcript)
    result = apply_corrections(transcript, manifest["accepted_corrections"])
    if transcript_hash(result) != manifest["result_transcript_hash"]:
        raise ContractError("applied output does not match preview")
    return result, manifest


def inspect_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"contract_version": payload.get("contract_version"), "correction_set_id": payload.get("correction_set_id")}
