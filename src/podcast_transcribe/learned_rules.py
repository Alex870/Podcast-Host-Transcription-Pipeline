"""Helpers for project-local learned review rules used by the workbench."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


REVIEW_RULE_LIBRARY_FILENAME = "review_rule_library.json"
LEARNED_RULE_LIBRARY_VERSION = 1
LEARNED_RULE_ALLOWED_STATUSES = {"draft", "approved", "disabled", "superseded"}
LEARNED_RULE_ALLOWED_FAMILIES = {
    "cleanup_preference",
    "glossary_naming_preference",
    "speaker_label_preference",
    "style_phrasing_preference",
    "do_not_change_constraint",
}
LEARNED_RULE_ALLOWED_STAGES = {
    "transcript_cleanup_review",
    "glossary_correction_review",
    "speaker_consistency_review",
    "episode_qa_review",
}


def review_rule_library_path(project_root: Path) -> Path:
    return project_root / REVIEW_RULE_LIBRARY_FILENAME


def load_review_rule_library(project_root: Path) -> Dict[str, object]:
    path = review_rule_library_path(project_root)
    if not path.exists():
        return {"library_version": LEARNED_RULE_LIBRARY_VERSION, "rules": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {"library_version": LEARNED_RULE_LIBRARY_VERSION, "rules": []}
    if not isinstance(payload, dict):
        return {"library_version": LEARNED_RULE_LIBRARY_VERSION, "rules": []}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        rules = []
    normalized = [normalize_learned_rule(rule) for rule in rules if isinstance(rule, dict)]
    return {
        "library_version": int(payload.get("library_version") or LEARNED_RULE_LIBRARY_VERSION),
        "rules": normalized,
    }


def save_review_rule_library(project_root: Path, payload: Dict[str, object]):
    path = review_rule_library_path(project_root)
    rules = payload.get("rules") if isinstance(payload.get("rules"), list) else []
    normalized = [normalize_learned_rule(rule) for rule in rules if isinstance(rule, dict)]
    path.write_text(
        json.dumps(
            {
                "library_version": LEARNED_RULE_LIBRARY_VERSION,
                "rules": normalized,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def normalize_learned_rule(rule: Dict[str, object]) -> Dict[str, object]:
    now_ms = int(time.time() * 1000)
    status = str(rule.get("status") or "draft").strip().lower()
    if status not in LEARNED_RULE_ALLOWED_STATUSES:
        status = "draft"
    rule_family = str(rule.get("rule_family") or "style_phrasing_preference").strip()
    if rule_family not in LEARNED_RULE_ALLOWED_FAMILIES:
        rule_family = "style_phrasing_preference"
    stage_target = str(rule.get("stage_target") or "transcript_cleanup_review").strip()
    if stage_target not in LEARNED_RULE_ALLOWED_STAGES:
        stage_target = "transcript_cleanup_review"
    source_examples = rule.get("source_examples") if isinstance(rule.get("source_examples"), list) else []
    validation = rule.get("validation") if isinstance(rule.get("validation"), dict) else {}
    provenance = rule.get("provenance") if isinstance(rule.get("provenance"), dict) else {}
    instruction_payload = rule.get("instruction_payload") if isinstance(rule.get("instruction_payload"), dict) else {}
    audit = rule.get("audit") if isinstance(rule.get("audit"), dict) else {}
    return {
        "rule_id": str(rule.get("rule_id") or f"rule_{now_ms}"),
        "status": status,
        "activation_status": str(rule.get("activation_status") or ("approved" if status == "approved" else "pending_approval")),
        "rule_family": rule_family,
        "stage_target": stage_target,
        "summary": str(rule.get("summary") or "").strip(),
        "explanation": str(rule.get("explanation") or "").strip(),
        "instruction_payload": {
            "directive": str(instruction_payload.get("directive") or "").strip(),
            "avoid": [str(item).strip() for item in (instruction_payload.get("avoid") or []) if str(item).strip()],
            "positive_examples": [
                item for item in (instruction_payload.get("positive_examples") or []) if isinstance(item, dict)
            ],
            "negative_examples": [
                item for item in (instruction_payload.get("negative_examples") or []) if isinstance(item, dict)
            ],
        },
        "confidence": float(rule.get("confidence") or 0.0),
        "ambiguity_notes": [str(item).strip() for item in (rule.get("ambiguity_notes") or []) if str(item).strip()],
        "validation": validation,
        "source_examples": [item for item in source_examples if isinstance(item, dict)],
        "activation_scope": str(rule.get("activation_scope") or "project_review_layer"),
        "supersedes_rule_id": str(rule.get("supersedes_rule_id") or "").strip(),
        "superseded_by_rule_id": str(rule.get("superseded_by_rule_id") or "").strip(),
        "provenance": {
            "created_at_epoch_ms": int(provenance.get("created_at_epoch_ms") or now_ms),
            "updated_at_epoch_ms": int(provenance.get("updated_at_epoch_ms") or now_ms),
            "backend": str(provenance.get("backend") or "").strip(),
            "review_model_name": str(provenance.get("review_model_name") or "").strip(),
            "validation_evidence": provenance.get("validation_evidence") if isinstance(provenance.get("validation_evidence"), dict) else {},
        },
        "audit": {
            "approvals": audit.get("approvals") if isinstance(audit.get("approvals"), list) else [],
            "reruns": audit.get("reruns") if isinstance(audit.get("reruns"), list) else [],
            "backfills": audit.get("backfills") if isinstance(audit.get("backfills"), list) else [],
        },
    }


def list_review_rules(project_root: Path, include_inactive: bool = True) -> List[Dict[str, object]]:
    rules = load_review_rule_library(project_root)["rules"]
    if include_inactive:
        return list(rules)
    return [rule for rule in rules if str(rule.get("status") or "") == "approved"]


def approved_review_rules(project_root: Path) -> List[Dict[str, object]]:
    return [rule for rule in list_review_rules(project_root) if str(rule.get("status") or "") == "approved"]


def upsert_review_rule(project_root: Path, rule: Dict[str, object]) -> Dict[str, object]:
    payload = load_review_rule_library(project_root)
    normalized = normalize_learned_rule(rule)
    rules = payload["rules"]
    updated = False
    for index, existing in enumerate(rules):
        if str(existing.get("rule_id") or "") == normalized["rule_id"]:
            rules[index] = normalized
            updated = True
            break
    if not updated:
        rules.append(normalized)
    save_review_rule_library(project_root, payload)
    return normalized


def get_review_rule(project_root: Path, rule_id: str) -> Optional[Dict[str, object]]:
    for rule in list_review_rules(project_root):
        if str(rule.get("rule_id") or "") == str(rule_id):
            return rule
    return None


def rule_prompt_payload(rule: Dict[str, object]) -> Dict[str, object]:
    instruction = rule.get("instruction_payload") if isinstance(rule.get("instruction_payload"), dict) else {}
    return {
        "rule_id": str(rule.get("rule_id") or ""),
        "rule_family": str(rule.get("rule_family") or ""),
        "stage_target": str(rule.get("stage_target") or ""),
        "summary": str(rule.get("summary") or ""),
        "directive": str(instruction.get("directive") or ""),
        "avoid": [str(item).strip() for item in (instruction.get("avoid") or []) if str(item).strip()],
    }


def active_rules_for_stage(rules: List[Dict[str, object]], stage_name: str) -> List[Dict[str, object]]:
    return [
        rule_prompt_payload(rule)
        for rule in rules
        if str(rule.get("status") or "") == "approved" and str(rule.get("stage_target") or "") == stage_name
    ]

