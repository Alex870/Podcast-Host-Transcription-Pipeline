"""Versioned correction contracts used at the ecosystem boundary.

This module deliberately has no dependencies on the transcription runtime.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


CORRECTION_CONTRACT_V1 = "correction-manifest-v1"
CORRECTION_CONTRACT = "correction-manifest-v2"
SUPPORTED_CORRECTION_CONTRACTS = {CORRECTION_CONTRACT_V1, CORRECTION_CONTRACT}
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


def _normalize_correction(
    item: Mapping[str, Any],
    *,
    ordinal: int,
    reviewer: str = "",
) -> dict[str, Any]:
    source_span_id = str(item.get("source_span_id") or item.get("segment_id") or "")
    field = str(item.get("field") or "text")
    status = str(item.get("status") or item.get("adjudication_state") or "approved")
    if status == "accepted":
        status = "approved"
    anchor = item.get("source_anchor") if isinstance(item.get("source_anchor"), Mapping) else {}
    normalized = {
        **deepcopy(dict(item)),
        "correction_kind": str(item.get("correction_kind") or item.get("reason_code") or "text"),
        "scope": str(item.get("scope") or "episode_segment"),
        "source_span_id": source_span_id,
        "source_anchor": {
            "source_span_id": source_span_id,
            "field": field,
            **deepcopy(dict(anchor)),
        },
        "field": field,
        "before": item.get("before"),
        "after": item.get("after"),
        "before_value_guard": item.get("before"),
        "reason_code": str(item.get("reason_code") or "operator_correction"),
        "status": status,
        "adjudication_state": "accepted" if status == "approved" else status,
        "supersedes": [str(value) for value in item.get("supersedes") or []],
        "superseded_by": [str(value) for value in item.get("superseded_by") or []],
        "provenance": deepcopy(dict(item.get("provenance") or {"reviewer": reviewer})),
        "ordinal": ordinal,
    }
    identity = {
        key: normalized[key]
        for key in (
            "correction_kind",
            "scope",
            "source_span_id",
            "source_anchor",
            "field",
            "before",
            "after",
            "reason_code",
            "supersedes",
        )
    }
    normalized["correction_id"] = str(item.get("correction_id") or canonical_id(identity, prefix="corr"))
    return normalized


def normalize_correction_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    version = str(manifest.get("contract_version") or "")
    if version not in SUPPORTED_CORRECTION_CONTRACTS:
        raise ContractError(f"unsupported correction contract: {version or 'missing'}")
    if version == CORRECTION_CONTRACT:
        result = deepcopy(dict(manifest))
        raw = result.get("corrections") or result.get("accepted_corrections") or []
    else:
        result = {
            key: deepcopy(value)
            for key, value in manifest.items()
            if key not in {"contract_version", "accepted_corrections", "correction_set_id"}
        }
        result["contract_version"] = CORRECTION_CONTRACT
        result["source_contract_version"] = CORRECTION_CONTRACT_V1
        raw = manifest.get("accepted_corrections") or []
    corrections = [
        _normalize_correction(item, ordinal=index, reviewer=str(result.get("reviewer_pseudonym") or ""))
        for index, item in enumerate(raw)
        if isinstance(item, Mapping)
    ]
    result["corrections"] = corrections
    result["accepted_corrections"] = [
        deepcopy(item) for item in corrections if str(item.get("status") or "") == "approved"
    ]
    result["correction_set_id"] = canonical_id(correction_identity_payload(result), prefix="correction")
    return result


def build_correction_manifest(
    transcript: Mapping[str, Any],
    corrections: Iterable[Mapping[str, Any]],
    *,
    reviewer: str,
    producer: Mapping[str, str],
    result_transcript: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = [
        _normalize_correction(item, ordinal=index, reviewer=reviewer)
        for index, item in enumerate(corrections)
        if str(item.get("status") or item.get("adjudication_state") or "accepted") in {"approved", "accepted"}
    ]
    accepted = [item for item in normalized if item["status"] == "approved"]
    result = result_transcript if result_transcript is not None else apply_corrections(transcript, accepted)
    manifest: dict[str, Any] = {
        "contract_version": CORRECTION_CONTRACT,
        "producer": dict(producer),
        "source_transcript_hash": transcript_hash(transcript),
        "result_transcript_hash": transcript_hash(result),
        "reviewer_pseudonym": reviewer,
        "corrections": normalized,
        "accepted_corrections": accepted,
    }
    manifest["correction_set_id"] = canonical_id(correction_identity_payload(manifest), prefix="correction")
    validate_correction_manifest(manifest, transcript)
    return manifest


def build_correction_manifest_history(
    transcript: Mapping[str, Any],
    corrections: Iterable[Mapping[str, Any]],
    *,
    reviewer: str,
    producer: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = [
        _normalize_correction(item, ordinal=index, reviewer=reviewer)
        for index, item in enumerate(corrections)
        if isinstance(item, Mapping)
    ]
    approved = [item for item in normalized if item["status"] == "approved"]
    result = apply_corrections(transcript, approved)
    manifest: dict[str, Any] = {
        "contract_version": CORRECTION_CONTRACT,
        "producer": dict(producer),
        "source_transcript_hash": transcript_hash(transcript),
        "result_transcript_hash": transcript_hash(result),
        "reviewer_pseudonym": reviewer,
        "corrections": normalized,
        "accepted_corrections": deepcopy(approved),
    }
    manifest["correction_set_id"] = canonical_id(correction_identity_payload(manifest), prefix="correction")
    validate_correction_manifest(manifest, transcript)
    return manifest, result


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
    original_version = str(manifest.get("contract_version") or "")
    if original_version not in SUPPORTED_CORRECTION_CONTRACTS:
        raise ContractError("unsupported correction contract")
    if original_version == CORRECTION_CONTRACT_V1:
        if manifest.get("source_transcript_hash") != transcript_hash(transcript):
            raise ContractError("stale source transcript hash")
        for correction in manifest.get("accepted_corrections", []):
            segment = _find_segment(transcript, str(correction.get("source_span_id") or ""))
            if segment.get(str(correction.get("field") or "")) != correction.get("before"):
                raise ContractError(f"before value mismatch for {correction.get('source_span_id')}")
        return
    if manifest.get("source_transcript_hash") != transcript_hash(transcript):
        raise ContractError("stale source transcript hash")
    expected_id = canonical_id(correction_identity_payload(manifest), prefix="correction")
    if manifest.get("correction_set_id") != expected_id:
        raise ContractError("correction-set identity mismatch")
    for correction in manifest.get("corrections", manifest.get("accepted_corrections", [])):
        required = {
            "correction_id", "correction_kind", "scope", "source_span_id", "source_anchor",
            "field", "before", "after", "before_value_guard", "reason_code", "status",
            "supersedes", "superseded_by", "provenance",
        }
        missing = required - correction.keys()
        if missing:
            raise ContractError(f"correction missing fields: {sorted(missing)}")
        if correction["status"] not in {"draft", "approved", "disabled", "superseded", "rejected"}:
            raise ContractError("invalid correction status")
        if correction["status"] != "approved":
            continue
        segment = _find_segment(transcript, str(correction["source_span_id"]))
        if segment.get(str(correction["field"])) != correction["before"]:
            raise ContractError(f"before value mismatch for {correction['source_span_id']}")


def apply_corrections(transcript: Mapping[str, Any], corrections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result = deepcopy(dict(transcript))
    for correction in corrections:
        status = str(correction.get("status") or correction.get("adjudication_state") or "approved")
        if status not in {"approved", "accepted"}:
            continue
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
    normalized = normalize_correction_manifest(payload)
    return {
        "contract_version": payload.get("contract_version"),
        "normalized_contract_version": normalized.get("contract_version"),
        "correction_set_id": normalized.get("correction_set_id"),
        "correction_count": len(normalized.get("corrections") or []),
    }
