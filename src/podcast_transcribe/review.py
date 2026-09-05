"""Optional staged LLM review helpers for additive transcript post-processing."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from hashlib import sha1
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from podcast_transcribe.config import resolve_review_runtime_config
from podcast_transcribe.learned_rules import active_rules_for_stage
from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.state import ARTIFACT_DIRNAME, atomic_write_text, audio_file_fingerprint


REVIEW_PIPELINE_VERSION = 2
LOCAL_REVIEW_CONTEXT_LIMIT = 6000
LONG_CONTEXT_MIN_BUDGET = 48000
EPISODE_QA_OVERLAP_SEGMENTS = 3
LOCAL_REVIEW_MAX_OUTPUT_TOKENS = 1400
CHUNKED_REVIEW_MAX_OUTPUT_TOKENS = 2400
LONG_REVIEW_MAX_OUTPUT_TOKENS = 4000
CALIBRATION_SAFETY_RATIO = 0.8
ADAPT_UPWARD_AFTER_SUCCESSES = 10
ADAPT_UPWARD_GROWTH_FACTOR = 1.05
ADAPT_UPWARD_MAX_STEP = 512
ADAPT_COOLDOWN_SUCCESSES = 10
ADAPT_DOWNWARD_COOLDOWN_SUCCESSES = 10
LONG_CONTEXT_ADAPT_UPWARD_AFTER_SUCCESSES = 20
LONG_CONTEXT_ADAPT_UPWARD_GROWTH_FACTOR = 1.03
LONG_CONTEXT_ADAPT_UPWARD_MAX_STEP = 256
LONG_CONTEXT_ADAPT_COOLDOWN_SUCCESSES = 20
UPWARD_FAILURE_LOCK_THRESHOLD = 2
SYNTHETIC_SEGMENT_ID_FACTOR = 10000

REVIEW_STAGE_FAMILIES = {
    "transcript_cleanup_review": "local_text_review",
    "glossary_correction_review": "local_text_review",
    "speaker_consistency_review": "local_speaker_review",
    "episode_qa_review": "long_context_review",
}

STAGE_DEFINITIONS = [
    {
        "name": "transcript_cleanup_review",
        "label": "cleanup",
        "edit_scope": "text_only",
        "mode": "local_batch",
        "description": (
            "Conservative transcript cleanup only. Remove obvious local transcription noise, "
            "small restarts, or local mistakes without rewriting style."
        ),
    },
    {
        "name": "glossary_correction_review",
        "label": "glossary",
        "edit_scope": "text_only",
        "mode": "local_batch",
        "description": (
            "Preferred-term, capitalization, and recurring naming consistency only. "
            "Do not make broad editorial changes."
        ),
    },
    {
        "name": "speaker_consistency_review",
        "label": "speaker_consistency",
        "edit_scope": "speaker_labels_primary",
        "mode": "local_batch",
        "description": (
            "Correct likely speaker-label drift. Speaker changes are allowed. "
            "Text changes are allowed only when needed to preserve obvious speaker-boundary coherence."
        ),
    },
    {
        "name": "episode_qa_review",
        "label": "episode_qa",
        "edit_scope": "cross_segment_consistency",
        "mode": "long_context",
        "description": (
            "Conservative episode-level QA using broader context. Fix only cross-segment consistency issues "
            "or obvious misrecognitions that need larger context. Do not broadly rewrite style."
        ),
    },
]

ReviewProgressCallback = Callable[[Dict[str, object]], None]


def _review_progress_path(debug_context: Optional[Dict[str, object]]) -> Optional[Path]:
    if not isinstance(debug_context, dict):
        return None
    output_dir = str(debug_context.get("output_dir") or "").strip()
    audio_path = str(debug_context.get("audio_path") or "").strip()
    if not output_dir or not audio_path:
        return None
    return Path(output_dir) / ARTIFACT_DIRNAME / Path(audio_path).stem / "review_progress.json"


def _review_progress_fingerprint(
    backend_capabilities: Dict[str, object],
    flags: Dict[str, bool],
    review_input_source: str,
) -> Dict[str, object]:
    return {
        "pipeline_version": REVIEW_PIPELINE_VERSION,
        "runtime_profile": str(backend_capabilities.get("runtime_profile") or ""),
        "backend_name": str(backend_capabilities.get("backend_name") or ""),
        "review_base_url": str(backend_capabilities.get("review_base_url") or ""),
        "review_model_name": str(backend_capabilities.get("review_model_name") or ""),
        "review_reasoning_effort": str(backend_capabilities.get("review_reasoning_effort") or ""),
        "review_batch_token_limit": int(backend_capabilities.get("review_batch_token_limit") or 0),
        "review_candidate_filter": bool(backend_capabilities.get("review_candidate_filter")),
        "stage_flags": dict(flags),
        "review_input_source": review_input_source,
    }


def _segment_from_progress_payload(payload: Dict[str, object]) -> SegmentItem:
    words = [
        WordItem(
            start=word.get("start"),
            end=word.get("end"),
            word=str(word.get("word") or ""),
            speaker=word.get("speaker"),
        )
        for word in payload.get("words") or []
        if isinstance(word, dict)
    ]
    return SegmentItem(
        id=int(payload["id"]),
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload.get("text") or ""),
        speaker=payload.get("speaker"),
        avg_logprob=payload.get("avg_logprob"),
        no_speech_prob=payload.get("no_speech_prob"),
        words=words,
        original_text=payload.get("original_text"),
        cleanup_applied=bool(payload.get("cleanup_applied")),
        cleanup_level=str(payload.get("cleanup_level") or ""),
        manual_correction_applied=bool(payload.get("manual_correction_applied")),
        original_speaker=payload.get("original_speaker"),
        llm_reviewed_text=payload.get("llm_reviewed_text"),
        review_runtime_profile=payload.get("review_runtime_profile"),
        review_backend=payload.get("review_backend"),
        review_model_name=payload.get("review_model_name"),
        review_stage_flags=payload.get("review_stage_flags"),
    )


def _load_review_progress(
    debug_context: Optional[Dict[str, object]],
    backend_capabilities: Dict[str, object],
    flags: Dict[str, bool],
    review_input_source: str,
) -> Dict[str, object]:
    path = _review_progress_path(debug_context)
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        audio_path = Path(str((debug_context or {}).get("audio_path") or ""))
        if payload.get("source_fingerprint") != audio_file_fingerprint(audio_path):
            return {}
        if payload.get("runtime_fingerprint") != _review_progress_fingerprint(backend_capabilities, flags, review_input_source):
            return {}
        segments = [
            _segment_from_progress_payload(item)
            for item in payload.get("segments") or []
            if isinstance(item, dict) and item.get("id") is not None
        ]
        return {**payload, "segments": segments} if segments else {}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}


def _save_review_progress(
    debug_context: Optional[Dict[str, object]],
    backend_capabilities: Dict[str, object],
    flags: Dict[str, bool],
    review_input_source: str,
    segments: List[SegmentItem],
    completed_stages: List[str],
    stage_results: Dict[str, Dict[str, object]],
    episode_notes: List[str],
):
    path = _review_progress_path(debug_context)
    if path is None:
        return
    audio_path = Path(str((debug_context or {}).get("audio_path") or ""))
    try:
        source_fingerprint = audio_file_fingerprint(audio_path)
    except OSError:
        return
    payload = {
        "progress_version": 1,
        "source_fingerprint": source_fingerprint,
        "runtime_fingerprint": _review_progress_fingerprint(backend_capabilities, flags, review_input_source),
        "completed_stages": list(completed_stages),
        "stage_results": stage_results,
        "episode_notes": list(episode_notes),
        "segments": [asdict(segment) for segment in segments],
    }
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _review_stage_family(stage_name: str) -> str:
    return REVIEW_STAGE_FAMILIES.get(stage_name, "local_text_review")


def _family_default_budget(backend_capabilities: Dict[str, object], family_name: str) -> int:
    max_context_budget = int(backend_capabilities.get("max_context_budget") or 0)
    batch_limit = int(backend_capabilities.get("review_batch_token_limit") or 12000)
    batch_limit = batch_limit if backend_capabilities.get("review_batch_token_limit_configured") else max_context_budget
    if family_name == "long_context_review":
        return max(4096, min(batch_limit, max_context_budget - 2048))
    return min(LOCAL_REVIEW_CONTEXT_LIMIT, batch_limit, max(2048, max_context_budget - 1024))


def _family_hard_ceiling(backend_capabilities: Dict[str, object], family_name: str) -> int:
    max_context_budget = int(backend_capabilities.get("max_context_budget") or 0)
    batch_limit = int(backend_capabilities.get("review_batch_token_limit") or 12000)
    batch_limit = batch_limit if backend_capabilities.get("review_batch_token_limit_configured") else max_context_budget
    if family_name == "long_context_review":
        return max(4096, min(batch_limit, max_context_budget - 2048))
    return max(2048, min(LOCAL_REVIEW_CONTEXT_LIMIT, batch_limit, max_context_budget - 1024))


def _segment_review_reasons(segment: SegmentItem) -> List[str]:
    """Return conservative reasons for sending a segment to the LLM reviewer."""

    reasons: List[str] = []
    if bool(getattr(segment, "cleanup_applied", False)):
        reasons.append("deterministic_cleanup")
    if bool(getattr(segment, "manual_correction_applied", False)):
        reasons.append("manual_correction")
    original_text = getattr(segment, "original_text", None)
    if original_text not in (None, "") and str(original_text) != str(segment.text):
        reasons.append("text_changed")
    original_speaker = getattr(segment, "original_speaker", None)
    if original_speaker not in (None, "") and str(original_speaker) != str(segment.speaker):
        reasons.append("speaker_changed")
    if not str(getattr(segment, "text", "") or "").strip():
        reasons.append("empty_text")

    # Missing confidence is treated as uncertain. That keeps legacy cleaned
    # JSON safe while allowing high-confidence, unchanged segments to skip.
    try:
        avg_logprob = float(segment.avg_logprob) if segment.avg_logprob is not None else None
    except (TypeError, ValueError):
        avg_logprob = None
    try:
        no_speech_prob = float(segment.no_speech_prob) if segment.no_speech_prob is not None else None
    except (TypeError, ValueError):
        no_speech_prob = None
    if avg_logprob is None or avg_logprob < -1.0:
        reasons.append("low_or_missing_avg_logprob")
    if no_speech_prob is None or no_speech_prob > 0.6:
        reasons.append("high_or_missing_no_speech_prob")
    return reasons


def _review_scope(
    segments: List[SegmentItem],
    candidate_ids: set[str],
    stage_definition: Dict[str, object],
) -> Tuple[List[SegmentItem], set[str]]:
    """Return editable candidates plus small context windows for episode QA."""

    if stage_definition["mode"] != "long_context":
        return [segment for segment in segments if str(segment.id) in candidate_ids], candidate_ids

    if not candidate_ids:
        return [], set()
    scope_indexes = set()
    for index, segment in enumerate(segments):
        if str(segment.id) not in candidate_ids:
            continue
        for context_index in range(max(0, index - EPISODE_QA_OVERLAP_SEGMENTS), min(len(segments), index + EPISODE_QA_OVERLAP_SEGMENTS + 1)):
            scope_indexes.add(context_index)
    scope = [segments[index] for index in sorted(scope_indexes)]
    return scope, candidate_ids


def _family_floor_budget(family_name: str) -> int:
    return 4096 if family_name == "long_context_review" else 512


def _family_growth_policy(family_name: str) -> Dict[str, object]:
    if family_name == "long_context_review":
        return {
            "growth_enabled": True,
            "success_threshold": LONG_CONTEXT_ADAPT_UPWARD_AFTER_SUCCESSES,
            "growth_factor": LONG_CONTEXT_ADAPT_UPWARD_GROWTH_FACTOR,
            "max_step": LONG_CONTEXT_ADAPT_UPWARD_MAX_STEP,
            "cooldown_successes": LONG_CONTEXT_ADAPT_COOLDOWN_SUCCESSES,
        }
    return {
        "growth_enabled": True,
        "success_threshold": ADAPT_UPWARD_AFTER_SUCCESSES,
        "growth_factor": ADAPT_UPWARD_GROWTH_FACTOR,
        "max_step": ADAPT_UPWARD_MAX_STEP,
        "cooldown_successes": ADAPT_COOLDOWN_SUCCESSES,
    }


def _read_json_url(url: str, timeout_seconds: int = 5) -> Dict[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def discover_backend_identity(backend_capabilities: Dict[str, object]) -> Dict[str, object]:
    healthcheck_url = str(backend_capabilities.get("healthcheck_url") or "").strip()
    backend_name = str(backend_capabilities.get("backend_name") or "").strip().lower()
    requested_model = str(backend_capabilities.get("review_model_name") or "").strip()
    if not healthcheck_url or backend_name not in {"vllm", "lm_studio"}:
        return {}

    try:
        payload = _read_json_url(healthcheck_url)
    except Exception:
        return {}

    model_entries = payload.get("data")
    if not isinstance(model_entries, list):
        return {}

    model_ids: List[str] = []
    matched_entry: Dict[str, object] = {}
    for entry in model_entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or "").strip()
        if not model_id:
            continue
        model_ids.append(model_id)
        if requested_model and model_id == requested_model:
            matched_entry = entry

    if not model_ids:
        return {}

    model_ids = sorted(set(model_ids))
    primary_model_id = requested_model if requested_model in model_ids else (model_ids[0] if len(model_ids) == 1 else "")
    if not matched_entry and primary_model_id:
        matched_entry = next(
            (entry for entry in model_entries if isinstance(entry, dict) and str(entry.get("id") or "").strip() == primary_model_id),
            {},
        )

    reported_max_context = matched_entry.get("max_model_len") if isinstance(matched_entry, dict) else None
    try:
        reported_max_context = int(reported_max_context) if reported_max_context not in ("", None) else None
    except (TypeError, ValueError):
        reported_max_context = None

    digest = sha1("|".join(model_ids).encode("utf-8")).hexdigest()[:12]
    return {
        "backend_identity_source": f"{backend_name}_models_endpoint",
        "backend_identity_model_id": primary_model_id,
        "backend_identity_model_ids": model_ids,
        "backend_identity_digest": digest,
        "backend_identity_model_count": len(model_ids),
        "backend_identity_reported_max_context": reported_max_context,
    }


def enrich_backend_capabilities_with_identity(backend_capabilities: Dict[str, object]) -> Dict[str, object]:
    discovered = discover_backend_identity(backend_capabilities)
    return {**backend_capabilities, **discovered}


class ReviewCalibrationSession:
    def __init__(self, backend_capabilities: Dict[str, object], persisted_state: Optional[Dict[str, object]] = None):
        self.backend_capabilities = backend_capabilities
        self.persisted_state = persisted_state if isinstance(persisted_state, dict) else {}
        self.runtime_fingerprint = {
            "runtime_profile": str(backend_capabilities.get("runtime_profile") or ""),
            "backend_name": str(backend_capabilities.get("backend_name") or ""),
            "review_model_name": str(backend_capabilities.get("review_model_name") or ""),
            "review_base_url": str(backend_capabilities.get("review_base_url") or ""),
            "review_reasoning_effort": str(backend_capabilities.get("review_reasoning_effort") or ""),
            "review_batch_token_limit": int(backend_capabilities.get("review_batch_token_limit") or 0),
            "review_candidate_filter": bool(backend_capabilities.get("review_candidate_filter")),
            "backend_identity_source": str(backend_capabilities.get("backend_identity_source") or ""),
            "backend_identity_model_id": str(backend_capabilities.get("backend_identity_model_id") or ""),
            "backend_identity_digest": str(backend_capabilities.get("backend_identity_digest") or ""),
            "backend_identity_reported_max_context": backend_capabilities.get("backend_identity_reported_max_context"),
        }
        saved_fingerprint = self.persisted_state.get("runtime_fingerprint")
        # The /v1/models identity fields are opportunistic: a temporary health
        # check failure must not invalidate otherwise reusable calibration.
        fingerprint_keys = tuple(
            key
            for key in self.runtime_fingerprint
            if not key.startswith("backend_identity_")
        )

        def fingerprint_value(payload: object, key: str):
            if not isinstance(payload, dict) or key not in payload:
                return {
                    "review_reasoning_effort": "none",
                    "review_batch_token_limit": 0,
                    "review_candidate_filter": False,
                }.get(key)
            return payload.get(key)

        fingerprint_matches = bool(
            isinstance(saved_fingerprint, dict)
            and all(
                fingerprint_value(saved_fingerprint, key) == fingerprint_value(self.runtime_fingerprint, key)
                for key in fingerprint_keys
            )
        )
        families_payload = self.persisted_state.get("families") if fingerprint_matches else {}
        self._session_events: List[Dict[str, object]] = []
        self.warm_start_used = bool(fingerprint_matches and isinstance(families_payload, dict) and families_payload)
        # Calibration is a runtime-level property. If the persisted budgets were
        # calibrated for the same backend/model/prompt settings, reuse them and
        # do not probe the live model again on every episode.
        self.calibrated = bool(
            fingerprint_matches
            and self.persisted_state.get("calibrated") is True
            and isinstance(families_payload, dict)
            and families_payload
        )
        self.calibrated_this_run = False
        self.families: Dict[str, Dict[str, object]] = {}
        for family_name in {"local_text_review", "local_speaker_review", "long_context_review"}:
            default_budget = int(_family_default_budget(backend_capabilities, family_name))
            hard_ceiling = int(_family_hard_ceiling(backend_capabilities, family_name))
            floor_budget = int(_family_floor_budget(family_name))
            saved = families_payload.get(family_name) if isinstance(families_payload, dict) else {}
            if not isinstance(saved, dict):
                saved = {}
            current_budget = int(saved.get("current_budget") or default_budget)
            current_budget = max(floor_budget, min(hard_ceiling, current_budget))
            growth_policy = _family_growth_policy(family_name)
            self.families[family_name] = {
                "family_name": family_name,
                "current_budget": current_budget,
                "initial_budget": current_budget,
                "warm_start_budget": current_budget,
                "calibrated_budget": 0,
                "hard_ceiling": hard_ceiling,
                "floor_budget": floor_budget,
                "calibration_source": "warm_start_hint" if self.warm_start_used else "fresh_default_hint",
                "probe_attempts": 0,
                "stable_success_count": int(saved.get("stable_success_count") or 0) if self.warm_start_used else 0,
                "cooldown_remaining": int(saved.get("cooldown_remaining") or 0) if self.warm_start_used else 0,
                "upward_locked": bool(saved.get("upward_locked")),
                "growth_enabled": bool(growth_policy["growth_enabled"]) and (
                    family_name != "long_context_review" or bool(backend_capabilities.get("long_context_available"))
                ),
                "growth_policy": growth_policy,
                "warm_start_used": self.warm_start_used,
                "recent_events": list(saved.get("recent_events") or [])[-8:],
                "last_increase_budget": int(saved.get("last_increase_budget") or 0),
                "post_increase_failure_count": int(saved.get("post_increase_failure_count") or 0),
            }

    def serialize(self) -> Dict[str, object]:
        return {
            "runtime_fingerprint": self.runtime_fingerprint,
            "calibrated": bool(self.calibrated or self.calibrated_this_run),
            "families": {
                family_name: {
                    key: value
                    for key, value in family_state.items()
                    if key
                    in {
                        "current_budget",
                        "initial_budget",
                        "warm_start_budget",
                        "calibrated_budget",
                        "hard_ceiling",
                        "floor_budget",
                        "calibration_source",
                        "probe_attempts",
                        "stable_success_count",
                        "cooldown_remaining",
                        "upward_locked",
                        "recent_events",
                        "last_increase_budget",
                        "post_increase_failure_count",
                    }
                }
                for family_name, family_state in self.families.items()
            },
        }

    def metadata_snapshot(self) -> Dict[str, object]:
        return {
            "calibrated": self.calibrated,
            "calibrated_this_run": self.calibrated_this_run,
            "warm_start_used": self.warm_start_used,
            "runtime_fingerprint": self.runtime_fingerprint,
            "families": {
                family_name: {
                    "current_budget": int(family_state["current_budget"]),
                    "initial_budget": int(family_state["initial_budget"]),
                    "warm_start_budget": int(family_state["warm_start_budget"]),
                    "calibrated_budget": int(family_state["calibrated_budget"]),
                    "hard_ceiling": int(family_state["hard_ceiling"]),
                    "floor_budget": int(family_state["floor_budget"]),
                    "calibration_source": str(family_state["calibration_source"]),
                    "probe_attempts": int(family_state["probe_attempts"]),
                    "stable_success_count": int(family_state["stable_success_count"]),
                    "cooldown_remaining": int(family_state["cooldown_remaining"]),
                    "upward_locked": bool(family_state["upward_locked"]),
                    "growth_enabled": bool(family_state["growth_enabled"]),
                    "warm_start_used": bool(family_state["warm_start_used"]),
                    "recent_events": list(family_state["recent_events"]),
                }
                for family_name, family_state in self.families.items()
            },
        }

    def budget_for_stage(self, stage_name: str) -> int:
        family_name = _review_stage_family(stage_name)
        return int(self.families[family_name]["current_budget"])

    def calibrate_for_run(
        self,
        enabled_stage_names: List[str],
        segments: List[SegmentItem],
        progress_callback: Optional[ReviewProgressCallback] = None,
        debug_context: Optional[Dict[str, object]] = None,
    ):
        calibration_started = time.perf_counter()
        if self.calibrated or not self.backend_capabilities.get("backend_ready"):
            return
        families_to_calibrate: List[str] = []
        for stage_name in enabled_stage_names:
            family_name = _review_stage_family(stage_name)
            if family_name not in families_to_calibrate:
                families_to_calibrate.append(family_name)
        if not families_to_calibrate:
            self.calibrated = True
            self.calibrated_this_run = True
            if progress_callback:
                progress_callback(
                    {
                        "event": "calibration_complete",
                        "summary": "Review calibration complete (no enabled stage families).",
                        "elapsed_seconds": time.perf_counter() - calibration_started,
                    }
                )
            return
        for family_name in families_to_calibrate:
            self._calibrate_family(family_name, segments, debug_context=debug_context)
        self.calibrated = True
        self.calibrated_this_run = True
        if progress_callback:
            progress_callback(
                {
                    "event": "calibration_complete",
                    "summary": self._console_summary(families_to_calibrate),
                    "elapsed_seconds": time.perf_counter() - calibration_started,
                }
            )

    def _console_summary(self, families: List[str]) -> str:
        labels = {
            "local_text_review": "local text",
            "local_speaker_review": "speaker",
            "long_context_review": "long-context",
        }
        details = []
        source = "fallback defaults"
        for family_name in families:
            state = self.families[family_name]
            details.append(f"{labels[family_name]}={int(state['current_budget'])}")
            if "real_text" in str(state["calibration_source"]):
                source = "real-text probe"
        calibration_mode = "warm-start" if self.warm_start_used else "fresh"
        return f"Review calibration ({calibration_mode}): {', '.join(details)} ({source})"

    def _calibrate_family(
        self,
        family_name: str,
        segments: List[SegmentItem],
        debug_context: Optional[Dict[str, object]] = None,
    ):
        family_state = self.families[family_name]
        if family_name == "long_context_review" and not self.backend_capabilities.get("long_context_available"):
            family_state["calibration_source"] = "fallback_default"
            family_state["growth_enabled"] = False
            return
        sample_segments = list(segments)
        if not sample_segments:
            family_state["calibration_source"] = "warm_start_fallback_default" if self.warm_start_used else "fresh_fallback_default"
            family_state["calibrated_budget"] = int(family_state["current_budget"])
            return
        stage_definition = next(
            stage for stage in STAGE_DEFINITIONS if _review_stage_family(stage["name"]) == family_name
        )
        stage_mode = "chunked" if family_name == "long_context_review" else "local_batch"
        sample_total = sum(_estimated_segment_tokens(segment) for segment in sample_segments)
        high = max(
            family_state["floor_budget"],
            min(int(family_state["hard_ceiling"]), int(sample_total) if sample_total > 0 else int(family_state["hard_ceiling"])),
        )
        best_success = None

        def try_probe(target: int) -> bool:
            windows = _split_segments_by_token_budget(sample_segments, target, overlap_segments=0)
            probe_segments = windows[0] if windows else sample_segments[:1]
            actual_estimate = sum(_estimated_segment_tokens(segment) for segment in probe_segments)
            family_state["probe_attempts"] = int(family_state["probe_attempts"]) + 1
            try:
                payload = _execute_stage_backend_request(
                    probe_segments,
                    self.backend_capabilities,
                    stage_definition,
                    stage_mode,
                    debug_context={
                        **(debug_context or {}),
                        "window_index": f"calibration_{family_name}",
                        "window_total": family_state["probe_attempts"],
                    },
                )
                if not isinstance(payload.get("reviewed_segments"), list) or payload.get("corrected_segment_count") is None:
                    raise RuntimeError("Calibration probe returned incomplete review JSON keys.")
                nonlocal_best["value"] = actual_estimate
                return True
            except Exception:
                return False

        nonlocal_best = {"value": None}
        warm_start_target = max(
            family_state["floor_budget"],
            min(int(family_state["hard_ceiling"]), int(family_state["warm_start_budget"])),
        )
        if try_probe(warm_start_target):
            best_success = nonlocal_best["value"]
            low = warm_start_target + 1
            high_bound = high
        else:
            low = int(family_state["floor_budget"])
            high_bound = warm_start_target - 1

        while low <= high_bound:
            target = (low + high_bound) // 2
            if try_probe(target):
                best_success = nonlocal_best["value"]
                low = target + 1
            else:
                high_bound = target - 1
        if best_success is None:
            family_state["calibration_source"] = "warm_start_fallback_default" if self.warm_start_used else "fresh_fallback_default"
            family_state["current_budget"] = int(_family_default_budget(self.backend_capabilities, family_name))
            family_state["calibrated_budget"] = int(family_state["current_budget"])
            return
        calibrated_budget = max(
            family_state["floor_budget"],
            min(
                family_state["hard_ceiling"],
                int(best_success * CALIBRATION_SAFETY_RATIO),
            ),
        )
        family_state["current_budget"] = calibrated_budget
        family_state["calibrated_budget"] = calibrated_budget
        family_state["calibration_source"] = "warm_start_real_text" if self.warm_start_used else "fresh_real_text"

    def note_truncation(
        self,
        family_name: str,
        failing_estimate: int,
        progress_callback: Optional[ReviewProgressCallback] = None,
    ):
        family_state = self.families[family_name]
        old_budget = int(family_state["current_budget"])
        new_budget = max(
            int(family_state["floor_budget"]),
            min(old_budget, int(failing_estimate * CALIBRATION_SAFETY_RATIO)),
        )
        family_state["current_budget"] = new_budget
        family_state["stable_success_count"] = 0
        family_state["cooldown_remaining"] = max(int(family_state["cooldown_remaining"]), ADAPT_DOWNWARD_COOLDOWN_SUCCESSES)
        if int(family_state["last_increase_budget"]) > 0:
            family_state["post_increase_failure_count"] = int(family_state["post_increase_failure_count"]) + 1
            if int(family_state["post_increase_failure_count"]) >= UPWARD_FAILURE_LOCK_THRESHOLD:
                family_state["upward_locked"] = True
        if new_budget < old_budget:
            event = {"event": "shrink", "from": old_budget, "to": new_budget, "estimate": failing_estimate}
            family_state["recent_events"] = (family_state["recent_events"] + [event])[-8:]
            self._session_events.append({"family": family_name, **event})
        if progress_callback and new_budget < old_budget:
            progress_callback(
                {
                    "event": "budget_reduced",
                    "family_name": family_name,
                    "old_budget": old_budget,
                    "new_budget": new_budget,
                }
            )

    def note_success(
        self,
        family_name: str,
        progress_callback: Optional[ReviewProgressCallback] = None,
        had_split: bool = False,
    ):
        family_state = self.families[family_name]
        if had_split:
            family_state["stable_success_count"] = 0
            return
        growth_policy = family_state["growth_policy"]
        family_state["stable_success_count"] = int(family_state["stable_success_count"]) + 1
        if int(family_state["cooldown_remaining"]) > 0:
            family_state["cooldown_remaining"] = int(family_state["cooldown_remaining"]) - 1
        if (
            not family_state["growth_enabled"]
            or not self.backend_capabilities.get("review_auto_adapt_upward")
            or family_state["upward_locked"]
            or int(family_state["cooldown_remaining"]) > 0
            or int(family_state["stable_success_count"]) < int(growth_policy["success_threshold"])
        ):
            return
        old_budget = int(family_state["current_budget"])
        proposed_budget = min(
            int(family_state["hard_ceiling"]),
            old_budget + min(int(growth_policy["max_step"]), max(1, int(old_budget * (float(growth_policy["growth_factor"]) - 1.0)))),
        )
        if proposed_budget <= old_budget:
            return
        family_state["current_budget"] = proposed_budget
        family_state["cooldown_remaining"] = int(growth_policy["cooldown_successes"])
        family_state["stable_success_count"] = 0
        family_state["last_increase_budget"] = proposed_budget
        family_state["post_increase_failure_count"] = 0
        event = {"event": "grow", "from": old_budget, "to": proposed_budget}
        family_state["recent_events"] = (family_state["recent_events"] + [event])[-8:]
        self._session_events.append({"family": family_name, **event})
        if progress_callback:
            progress_callback(
                {
                    "event": "budget_increased",
                    "family_name": family_name,
                    "old_budget": old_budget,
                    "new_budget": proposed_budget,
                }
            )

def build_review_stage_flags(runtime_review_config: Dict[str, object]) -> Dict[str, bool]:
    return {
        "transcript_cleanup_review": bool(runtime_review_config.get("transcript_cleanup_review")),
        "glossary_correction_review": bool(runtime_review_config.get("glossary_correction_review")),
        "speaker_consistency_review": bool(runtime_review_config.get("speaker_consistency_review")),
        "episode_qa_review": bool(runtime_review_config.get("episode_qa_review")),
    }


def resolve_backend_capabilities(runtime_review_config: Optional[Dict[str, object]]) -> Dict[str, object]:
    resolved = resolve_review_runtime_config(runtime_review_config or {})
    backend_name = str(resolved.get("effective_backend") or "none")
    max_context_budget = int(resolved.get("max_context_budget") or 0)
    if backend_name == "vllm":
        long_context_available = (
            bool(resolved.get("transcript_qa_available"))
            and bool(resolved.get("episode_wide_correction_available"))
            and max_context_budget >= LONG_CONTEXT_MIN_BUDGET
        )
    elif backend_name == "lm_studio":
        long_context_available = (
            bool(resolved.get("transcript_qa_available"))
            and bool(resolved.get("episode_wide_correction_available"))
            and bool(resolved.get("structured_output_support"))
            and max_context_budget >= 64000
        )
    else:
        long_context_available = False

    base_url = str(resolved.get("review_base_url") or "").rstrip("/")
    return {
        **resolved,
        "preferred_terms": list(resolved.get("preferred_terms") or []),
        "backend_name": backend_name,
        "healthcheck_url": f"{base_url}/v1/models" if base_url else "",
        "max_context_budget": max_context_budget,
        "long_context_available": long_context_available,
        "backend_identity_source": "",
        "backend_identity_model_id": "",
        "backend_identity_model_ids": [],
        "backend_identity_digest": "",
        "backend_identity_model_count": 0,
        "backend_identity_reported_max_context": None,
    }


def _segment_prompt_payload(segments: List[SegmentItem]) -> List[Dict[str, object]]:
    return [
        {
            "id": segment.id,
            "speaker": segment.speaker,
            "text": segment.text,
        }
        for segment in segments
    ]


def _preferred_terms_payload(backend_capabilities: Dict[str, object]) -> List[str]:
    return [str(term).strip() for term in (backend_capabilities.get("preferred_terms") or []) if str(term).strip()]


def _segment_contains_protected_term(text: str, protected_term: str) -> bool:
    return protected_term in (text or "")


def _normalize_failure_reason(message: str) -> str:
    text = (message or "").strip().lower()
    if "truncated" in text or "length" in text:
        return "truncation"
    if "invalid json" in text:
        return "invalid_json"
    if "invalid_response_at_minimum_size" in text:
        return "invalid_response_at_minimum_size"
    if "backend_unavailable" in text or "connection failed" in text:
        return "backend_unavailable"
    if text.startswith("review backend http "):
        return "http_error"
    if "empty response" in text:
        return "empty_response"
    return "other"


def _estimated_segment_tokens(segment: SegmentItem) -> int:
    text = str(segment.text or "")
    # Be conservative here: review prompts wrap each segment in JSON and the model often
    # echoes long text spans back in structured output, so prompt-side underestimation can
    # easily produce oversized windows and truncated completions.
    return max(32, len(text) // 2 + 48)


def _split_segments_by_token_budget(
    segments: List[SegmentItem],
    token_budget: int,
    overlap_segments: int = 0,
) -> List[List[SegmentItem]]:
    if not segments:
        return []
    windows: List[List[SegmentItem]] = []
    index = 0
    budget = max(256, token_budget)
    while index < len(segments):
        running = 0
        end = index
        while end < len(segments):
            estimate = _estimated_segment_tokens(segments[end])
            if end > index and running + estimate > budget:
                break
            running += estimate
            end += 1
        if end == index:
            end += 1
        windows.append(segments[index:end])
        if end >= len(segments):
            break
        index = max(index + 1, end - overlap_segments)
    return windows


def _split_text_into_review_chunks(text: str, token_budget: int) -> List[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []

    max_words = max(48, token_budget // 4)
    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentence_parts:
        sentence_parts = [normalized]

    units: List[str] = []
    for part in sentence_parts:
        words = part.split()
        if len(words) <= max_words:
            units.append(part)
            continue
        for index in range(0, len(words), max_words):
            units.append(" ".join(words[index : index + max_words]).strip())

    chunks: List[str] = []
    current_words: List[str] = []
    current_count = 0
    for unit in units:
        unit_words = unit.split()
        unit_count = len(unit_words)
        if current_words and current_count + unit_count > max_words:
            chunks.append(" ".join(current_words).strip())
            current_words = unit_words
            current_count = unit_count
            continue
        current_words.extend(unit_words)
        current_count += unit_count
    if current_words:
        chunks.append(" ".join(current_words).strip())
    return [chunk for chunk in chunks if chunk]


def _synthetic_segment_id(original_id: int, piece_index: int) -> int:
    return int(original_id) * SYNTHETIC_SEGMENT_ID_FACTOR + piece_index + 1


def _split_single_segment_for_review(segment: SegmentItem, token_budget: int) -> List[SegmentItem]:
    pieces = _split_text_into_review_chunks(segment.text, token_budget)
    if len(pieces) <= 1:
        return []

    total_chars = max(1, sum(len(piece) for piece in pieces))
    duration = max(0.0, float(segment.end) - float(segment.start))
    cursor = float(segment.start)
    synthetic_segments: List[SegmentItem] = []
    for piece_index, piece in enumerate(pieces):
        if piece_index == len(pieces) - 1:
            piece_end = float(segment.end)
        else:
            share = len(piece) / total_chars
            piece_end = min(float(segment.end), cursor + duration * share)
        synthetic_segments.append(
            SegmentItem(
                id=_synthetic_segment_id(int(segment.id), piece_index),
                start=cursor,
                end=piece_end,
                text=piece,
                speaker=segment.speaker,
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                words=[],
                original_text=piece,
                cleanup_applied=segment.cleanup_applied,
                cleanup_level=segment.cleanup_level,
                manual_correction_applied=segment.manual_correction_applied,
                original_speaker=segment.original_speaker,
                llm_reviewed_text=None,
                review_runtime_profile=segment.review_runtime_profile,
                review_backend=segment.review_backend,
                review_model_name=segment.review_model_name,
                review_stage_flags=deepcopy(segment.review_stage_flags),
            )
        )
        cursor = piece_end
    return synthetic_segments


def _merge_synthetic_segment_updates(
    original_segment: SegmentItem,
    synthetic_segments: List[SegmentItem],
    reviewed_items: List[Dict[str, object]],
    stage_definition: Dict[str, object],
) -> Optional[Dict[str, object]]:
    updates_by_id = {
        str(item.get("id")): item
        for item in reviewed_items
        if isinstance(item, dict) and item.get("id") is not None
    }
    text_parts: List[str] = []
    changed = False
    speaker_candidates: List[str] = []

    for synthetic_segment in synthetic_segments:
        update = updates_by_id.get(str(synthetic_segment.id), {})
        piece_text = str(update.get("text") or synthetic_segment.text).strip()
        if piece_text != str(synthetic_segment.text).strip():
            changed = True
        text_parts.append(piece_text)
        if update.get("speaker") is not None:
            speaker_candidates.append(str(update.get("speaker") or "").strip())

    merged_text = " ".join(part for part in text_parts if part).strip()
    merged_update: Dict[str, object] = {"id": original_segment.id}
    if merged_text and merged_text != str(original_segment.text or "").strip():
        merged_update["text"] = merged_text
        changed = True

    if stage_definition.get("edit_scope") != "text_only" and speaker_candidates:
        unique_speakers = {speaker for speaker in speaker_candidates if speaker}
        if len(unique_speakers) == 1:
            merged_speaker = next(iter(unique_speakers))
            if merged_speaker != str(original_segment.speaker or "").strip():
                merged_update["speaker"] = merged_speaker
                changed = True

    return merged_update if changed else None


def _normalize_backend_response_text(response: Dict[str, object]) -> str:
    def _coerce_parts(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return ""

    choices = response.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    content = _coerce_parts(message.get("content"))
    if content.strip():
        return content

    reasoning = _coerce_parts(message.get("reasoning"))
    if reasoning.strip():
        return reasoning

    return _coerce_parts(first.get("text"))


def _strip_json_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _extract_first_json_object(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    start = stripped.find("{")
    if start < 0:
        return stripped
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped


def _response_finish_reason(response: Dict[str, object]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    return str(first.get("finish_reason") or first.get("stop_reason") or "").strip().lower()


def _coerce_debug_flag(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def review_debug_directory(
    runtime_review_config: Optional[Dict[str, object]],
    debug_context: Optional[Dict[str, object]],
) -> Optional[Path]:
    resolved = resolve_review_runtime_config(runtime_review_config or {})
    config_debug_enabled = bool(resolved.get("review_debug"))
    env_debug_enabled = _coerce_debug_flag(os.getenv("PODCAST_TRANSCRIBE_REVIEW_DEBUG"))
    if not config_debug_enabled and not env_debug_enabled:
        return None

    explicit_dir = str(resolved.get("review_debug_dir") or os.getenv("PODCAST_TRANSCRIBE_REVIEW_DEBUG_DIR") or "").strip()
    if explicit_dir:
        return Path(explicit_dir)

    context = debug_context or {}
    output_dir = context.get("output_dir")
    audio_path = context.get("audio_path")
    if output_dir and audio_path:
        audio_stem = Path(str(audio_path)).stem
        return Path(str(output_dir)) / ARTIFACT_DIRNAME / audio_stem / "review_debug"
    return Path.cwd() / "_review_debug"


def _write_review_debug_artifact(
    runtime_review_config: Optional[Dict[str, object]],
    debug_context: Optional[Dict[str, object]],
    stage_definition: Dict[str, object],
    stage_mode: str,
    artifact_payload: Dict[str, object],
):
    debug_dir = review_debug_directory(runtime_review_config, debug_context)
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    context = debug_context or {}
    window_suffix = ""
    if context.get("window_index") is not None and context.get("window_total") is not None:
        window_suffix = f"_w{context['window_index']}of{context['window_total']}"
    file_name = (
        f"{int(time.time() * 1000)}_{stage_definition['name']}_{stage_mode}{window_suffix}.json"
    )
    payload = {
        "created_at_epoch_ms": int(time.time() * 1000),
        "audio_path": str(context.get("audio_path") or ""),
        "review_input_source": str(context.get("review_input_source") or ""),
        "stage_name": stage_definition["name"],
        "stage_label": stage_definition["label"],
        "stage_mode": stage_mode,
        "window_index": context.get("window_index"),
        "window_total": context.get("window_total"),
        **artifact_payload,
    }
    (debug_dir / file_name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _openai_compatible_chat_completion(
    backend_capabilities: Dict[str, object],
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> str:
    base_url = str(backend_capabilities.get("review_base_url") or "").rstrip("/")
    model_name = str(backend_capabilities.get("review_model_name") or "").strip()
    if not base_url or not model_name:
        raise RuntimeError("Review backend is not fully configured.")

    endpoint = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"} if backend_capabilities.get("structured_output_support") else None,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if backend_capabilities.get("backend_name") == "vllm" and "qwen" in model_name.lower():
        reasoning_effort = str(backend_capabilities.get("review_reasoning_effort") or "none").strip().lower()
        if reasoning_effort == "none":
            # Qwen3.8 defaults to thinking. This hard per-request switch is
            # stronger and more reliable than relying on a /no_think prompt.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_effort": reasoning_effort,
            }
    if payload["response_format"] is None:
        payload.pop("response_format")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Review backend HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Review backend connection failed: {exc.reason}") from exc


def _should_force_no_think(runtime_review_config: Dict[str, object]) -> bool:
    backend_name = str(runtime_review_config.get("backend_name") or "").strip().lower()
    model_name = str(runtime_review_config.get("review_model_name") or "").strip().lower()
    effort = str(runtime_review_config.get("review_reasoning_effort") or "none").strip().lower()
    return backend_name == "vllm" and "qwen" in model_name and effort == "none"


def _normalize_episode_notes(payload: Dict[str, object]) -> List[str]:
    notes = payload.get("episode_notes")
    if isinstance(notes, str):
        text = notes.strip()
        return [text] if text else []
    if isinstance(notes, list):
        return [str(note).strip() for note in notes if str(note).strip()]
    return []


def _is_transport_or_backend_failure(message: str) -> bool:
    text = message.strip().lower()
    return (
        "connection failed" in text
        or text.startswith("review backend http ")
        or "backend_unavailable" in text
    )


def _minimum_size_failure_reason(stage_mode: str) -> str:
    if stage_mode == "chunked":
        return "long_context_backend_failure_at_minimum_chunk"
    return "invalid_response_at_minimum_size"


def _build_stage_system_prompt(
    stage_definition: Dict[str, object],
    preferred_terms: Optional[List[str]] = None,
    learned_rules: Optional[List[Dict[str, object]]] = None,
) -> str:
    label = stage_definition["label"]
    description = stage_definition["description"]
    edit_scope = stage_definition["edit_scope"]
    prompt = (
        f"You are performing the '{label}' stage of podcast transcript review. "
        f"{description} Preserve segment ids and order. "
        "Do not invent facts. Make the smallest safe changes needed. "
        f"Allowed edit scope: {edit_scope}. "
        "Return strict JSON with keys reviewed_segments, corrected_segment_count, and episode_notes. "
        "IMPORTANT: reviewed_segments must contain ONLY segments that actually changed. "
        "Do not repeat unchanged segments. "
        "Do not echo or restate the transcript window. Return compact patch-style output only. "
        "If many segments could be changed, return only the highest-confidence changes for this pass. "
        "If nothing changed, return reviewed_segments as an empty list and corrected_segment_count as 0. "
        "Each returned reviewed segment must include id, text, and may include speaker only if the stage allows speaker changes."
    )
    protected_terms = [str(term).strip() for term in (preferred_terms or []) if str(term).strip()]
    if protected_terms:
        prompt += (
            " Preferred glossary terms are reserved spellings. "
            "If a segment already contains one of these exact spellings, preserve it exactly and do not normalize it away. "
            "Glossary corrections may move text toward these spellings, but never away from them. "
            f"Reserved preferred terms: {', '.join(protected_terms[:40])}."
        )
    active_rule_summaries = [
        str(rule.get("summary") or "").strip()
        for rule in (learned_rules or [])
        if str(rule.get("summary") or "").strip()
    ]
    if active_rule_summaries:
        prompt += (
            " Approved project-specific learned review rules also apply to this stage. "
            "Treat them as narrow local guidance and prefer the smallest change that satisfies them. "
            f"Active learned rules: {' | '.join(active_rule_summaries[:8])}."
        )
    return prompt


def _execute_stage_backend_request(
    segments: List[SegmentItem],
    backend_capabilities: Dict[str, object],
    stage_definition: Dict[str, object],
    stage_mode: str,
    preferred_terms: Optional[List[str]] = None,
    learned_rules: Optional[List[Dict[str, object]]] = None,
    debug_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    budget = int(backend_capabilities.get("max_context_budget") or 16000)
    if stage_mode == "local_batch":
        max_output_tokens = max(
            512,
            min(LOCAL_REVIEW_MAX_OUTPUT_TOKENS, budget // 12 if budget > 0 else 1200),
        )
    elif stage_mode == "chunked":
        max_output_tokens = max(
            768,
            min(CHUNKED_REVIEW_MAX_OUTPUT_TOKENS, budget // 10 if budget > 0 else 1800),
        )
    else:
        max_output_tokens = max(
            1024,
            min(LONG_REVIEW_MAX_OUTPUT_TOKENS, budget // 8 if budget > 0 else 2400),
        )
    request_payload = {
        "runtime_profile": backend_capabilities.get("runtime_profile"),
        "backend": backend_capabilities.get("backend_name"),
        "stage_name": stage_definition["name"],
        "stage_mode": stage_mode,
        "edit_scope": stage_definition["edit_scope"],
        "return_only_changed_segments": True,
        "changed_segments_only": True,
        "max_changed_segments_hint": 8 if stage_mode == "local_batch" else 16,
        "preferred_terms": [str(term).strip() for term in (preferred_terms or []) if str(term).strip()],
        "preferred_terms_are_reserved": True,
        "learned_rules": list(learned_rules or []),
        "segments": _segment_prompt_payload(segments),
    }
    user_prompt = json.dumps(request_payload, ensure_ascii=True)
    system_prompt = _build_stage_system_prompt(
        stage_definition,
        preferred_terms=preferred_terms,
        learned_rules=learned_rules,
    )
    if _should_force_no_think(backend_capabilities) and not system_prompt.lstrip().startswith("/no_think"):
        system_prompt = f"/no_think {system_prompt}"
    raw_response_text = _openai_compatible_chat_completion(
        backend_capabilities,
        system_prompt,
        user_prompt,
        max_output_tokens=max_output_tokens,
    )
    try:
        response = json.loads(raw_response_text)
    except json.JSONDecodeError as exc:
        _write_review_debug_artifact(
            backend_capabilities,
            debug_context,
            stage_definition,
            stage_mode,
            {
                "status": "invalid_http_json",
                "request_payload": request_payload,
                "raw_http_body": raw_response_text,
                "error": str(exc),
            },
        )
        raise RuntimeError(f"Review backend returned invalid HTTP JSON: {exc}") from exc
    if not isinstance(response, dict):
        _write_review_debug_artifact(
            backend_capabilities,
            debug_context,
            stage_definition,
            stage_mode,
            {
                "status": "non_object_http_json",
                "request_payload": request_payload,
                "raw_http_body": raw_response_text,
                "parsed_response": response,
            },
        )
        raise RuntimeError("Review backend returned a non-object HTTP JSON payload.")
    content = _normalize_backend_response_text(response)
    finish_reason = _response_finish_reason(response)
    if not content.strip():
        _write_review_debug_artifact(
            backend_capabilities,
            debug_context,
            stage_definition,
            stage_mode,
            {
                "status": "empty_normalized_content",
                "request_payload": request_payload,
                "raw_http_body": raw_response_text,
                "parsed_response": response,
                "normalized_content": content,
                "finish_reason": finish_reason,
            },
        )
        if finish_reason == "length":
            raise RuntimeError("Review backend response was truncated before any usable content was returned.")
        raise RuntimeError("Review backend returned an empty response.")
    content = _extract_first_json_object(_strip_json_code_fences(content))
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        _write_review_debug_artifact(
            backend_capabilities,
            debug_context,
            stage_definition,
            stage_mode,
            {
                "status": "invalid_stage_json",
                "request_payload": request_payload,
                "raw_http_body": raw_response_text,
                "parsed_response": response,
                "normalized_content": content,
                "error": str(exc),
                "finish_reason": finish_reason,
            },
        )
        if finish_reason == "length":
            raise RuntimeError(
                "Review backend response was truncated before the JSON payload completed."
            ) from exc
        raise RuntimeError(f"Review backend returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        _write_review_debug_artifact(
            backend_capabilities,
            debug_context,
            stage_definition,
            stage_mode,
            {
                "status": "non_object_stage_json",
                "request_payload": request_payload,
                "raw_http_body": raw_response_text,
                "parsed_response": response,
                "normalized_content": content,
                "stage_payload": payload,
                "finish_reason": finish_reason,
            },
        )
        raise RuntimeError("Review backend returned a non-object JSON payload.")
    _write_review_debug_artifact(
        backend_capabilities,
        debug_context,
        stage_definition,
        stage_mode,
        {
            "status": "ok",
            "request_payload": request_payload,
            "raw_http_body": raw_response_text,
            "parsed_response": response,
            "normalized_content": content,
            "stage_payload": payload,
            "finish_reason": finish_reason,
        },
    )
    return payload


def _call_stage_backend_request(
    window: List[SegmentItem],
    backend_capabilities: Dict[str, object],
    stage_definition: Dict[str, object],
    stage_mode: str,
    preferred_terms: Optional[List[str]] = None,
    learned_rules: Optional[List[Dict[str, object]]] = None,
    debug_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    kwargs = {
        "preferred_terms": preferred_terms,
        "debug_context": debug_context,
    }
    if learned_rules:
        kwargs["learned_rules"] = learned_rules
    return _execute_stage_backend_request(
        window,
        backend_capabilities,
        stage_definition,
        stage_mode,
        **kwargs,
    )


def _collect_window_updates(
    windows: List[List[SegmentItem]],
    backend_capabilities: Dict[str, object],
    stage_definition: Dict[str, object],
    stage_mode: str,
    preferred_terms: Optional[List[str]] = None,
    learned_rules: Optional[List[Dict[str, object]]] = None,
    calibration_session: Optional[ReviewCalibrationSession] = None,
    progress_callback: Optional[ReviewProgressCallback] = None,
    debug_context: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, Dict[str, object]], List[str], Dict[str, int]]:
    candidates_by_id: Dict[str, List[Dict[str, object]]] = {}
    notes: List[str] = []
    family_name = _review_stage_family(stage_definition["name"])
    split_event_count = 0
    conflicting_update_count = 0
    returned_item_count = 0

    def collect_with_backoff(
        window: List[SegmentItem],
        window_context: Dict[str, object],
    ) -> Tuple[List[Dict[str, object]], List[str], bool]:
        nonlocal split_event_count
        try:
            payload = _call_stage_backend_request(
                window,
                backend_capabilities,
                stage_definition,
                stage_mode,
                preferred_terms=preferred_terms,
                learned_rules=learned_rules,
                debug_context=window_context,
            )
        except RuntimeError as exc:
            message = str(exc)
            if _is_transport_or_backend_failure(message):
                raise
            if len(window) > 1:
                if calibration_session:
                    calibration_session.note_truncation(
                        family_name,
                        sum(_estimated_segment_tokens(segment) for segment in window),
                        progress_callback=progress_callback,
                    )
                midpoint = max(1, len(window) // 2)
                left = window[:midpoint]
                right = window[midpoint:]
                split_event_count += 1
                split_notes = [
                    (
                        f"{stage_definition['name']} window {window_context.get('window_index')}/"
                        f"{window_context.get('window_total')} was truncated; retrying in "
                        f"{len(left)}+{len(right)} segment sub-windows."
                    )
                ]
                if progress_callback:
                    progress_callback(
                        {
                            "event": "stage_window_split",
                            "stage_name": stage_definition["name"],
                            "stage_label": stage_definition["label"],
                            "mode": stage_mode,
                            "split_sizes": [len(left), len(right)],
                        }
                    )
                left_updates, left_notes, _ = collect_with_backoff(
                    left,
                    {
                        **window_context,
                        "window_index": f"{window_context.get('window_index')}a",
                    },
                )
                right_updates, right_notes, _ = collect_with_backoff(
                    right,
                    {
                        **window_context,
                        "window_index": f"{window_context.get('window_index')}b",
                    },
                )
                return left_updates + right_updates, split_notes + left_notes + right_notes, True
            if stage_mode == "local_batch":
                if calibration_session:
                    calibration_session.note_truncation(
                        family_name,
                        sum(_estimated_segment_tokens(segment) for segment in window),
                        progress_callback=progress_callback,
                    )
                synthetic_split_depth = int(window_context.get("synthetic_split_depth") or 0)
                synthetic_budget = max(
                    128,
                    min(
                        int(_family_floor_budget(family_name)),
                        max(128, sum(_estimated_segment_tokens(segment) for segment in window) // 2),
                    ),
                )
                synthetic_segments = _split_single_segment_for_review(window[0], synthetic_budget)
                if synthetic_split_depth < 3 and len(synthetic_segments) > 1:
                    split_event_count += 1
                    split_notes = [
                        (
                            f"{stage_definition['name']} window {window_context.get('window_index')}/"
                            f"{window_context.get('window_total')} hit minimum segment size; retrying as "
                            f"{len(synthetic_segments)} synthetic text chunks."
                        )
                    ]
                    if progress_callback:
                        progress_callback(
                            {
                                "event": "stage_window_split",
                                "stage_name": stage_definition["name"],
                                "stage_label": stage_definition["label"],
                                "mode": stage_mode,
                                "split_sizes": [len(synthetic_segments)],
                            }
                        )
                    synthetic_updates, synthetic_notes, _ = collect_with_backoff(
                        synthetic_segments,
                        {
                            **window_context,
                            "window_index": f"{window_context.get('window_index')}s",
                            "window_total": len(synthetic_segments),
                            "synthetic_split_depth": synthetic_split_depth + 1,
                        },
                    )
                    merged_update = _merge_synthetic_segment_updates(
                        window[0],
                        synthetic_segments,
                        synthetic_updates,
                        stage_definition,
                    )
                    if merged_update is None:
                        return [], split_notes + synthetic_notes, True
                    return [merged_update], split_notes + synthetic_notes, True
            raise RuntimeError(_minimum_size_failure_reason(stage_mode)) from exc

        window_notes = _normalize_episode_notes(payload)
        reviewed_items = [
            item
            for item in payload.get("reviewed_segments") or []
            if isinstance(item, dict) and item.get("id") is not None
        ]
        return reviewed_items, window_notes, False

    for window_index, window in enumerate(windows, start=1):
        if progress_callback:
            progress_callback(
                {
                    "event": "stage_window_progress",
                    "stage_name": stage_definition["name"],
                    "stage_label": stage_definition["label"],
                    "mode": stage_mode,
                    "current": window_index,
                    "total": len(windows),
                }
            )
        reviewed_items, window_notes, had_split = collect_with_backoff(
            window,
            {
                **(debug_context or {}),
                "window_index": window_index,
                "window_total": len(windows),
            },
        )
        if progress_callback:
            progress_callback(
                {
                    "event": "stage_response_success",
                    "stage_name": stage_definition["name"],
                    "stage_label": stage_definition["label"],
                    "mode": stage_mode,
                    "current": window_index,
                    "total": len(windows),
                    "had_split": had_split,
                }
            )
        if calibration_session:
            calibration_session.note_success(
                family_name,
                progress_callback=progress_callback,
                had_split=had_split,
            )
        for note in window_notes:
            notes.append(str(note))
        for item in reviewed_items:
            returned_item_count += 1
            candidates_by_id.setdefault(str(item["id"]), []).append(item)

    resolved_updates: Dict[str, Dict[str, object]] = {}
    for segment_id, candidates in candidates_by_id.items():
        unique_signatures = {
            (
                str(candidate.get("text") or ""),
                str(candidate.get("speaker") or ""),
            )
            for candidate in candidates
        }
        if len(unique_signatures) == 1:
            resolved_updates[segment_id] = candidates[0]
        else:
            conflicting_update_count += 1
            notes.append(
                f"Conflicting {stage_definition['name']} updates were ignored for segment {segment_id} during chunk reconciliation."
            )
    return resolved_updates, notes, {
        "window_count": len(windows),
        "split_event_count": split_event_count,
        "conflicting_update_count": conflicting_update_count,
        "returned_item_count": returned_item_count,
    }


def _apply_stage_updates(
    segments: List[SegmentItem],
    updates_by_id: Dict[str, Dict[str, object]],
    stage_definition: Dict[str, object],
    preferred_terms: Optional[List[str]] = None,
    editable_segment_ids: Optional[set[str]] = None,
) -> Tuple[List[SegmentItem], Dict[str, int], List[str]]:
    corrected_segment_count = 0
    updated_segments: List[SegmentItem] = []
    stage_notes: List[str] = []
    protected_term_violation_count = 0
    returned_change_count = 0
    applied_change_count = 0
    overridden_change_count = 0
    allow_text_edits = True
    allow_speaker_edits = stage_definition["name"] in {"speaker_consistency_review", "episode_qa_review"}
    protected_terms = [str(term).strip() for term in (preferred_terms or []) if str(term).strip()]

    for segment in segments:
        reviewed = deepcopy(segment)
        is_editable = editable_segment_ids is None or str(segment.id) in editable_segment_ids
        update = updates_by_id.get(str(segment.id), {}) if is_editable else {}
        intended_change = False
        if allow_text_edits and "text" in update:
            intended_text = str(update.get("text") or segment.text).strip() or segment.text
            if intended_text != segment.text:
                intended_change = True
        if allow_speaker_edits and "speaker" in update:
            intended_speaker = str(update.get("speaker") or segment.speaker or "").strip()
            if intended_speaker and intended_speaker != segment.speaker:
                intended_change = True
        if intended_change:
            returned_change_count += 1
        changed = False
        if allow_text_edits:
            reviewed_text = str(update.get("text") or segment.text).strip() or segment.text
            for protected_term in protected_terms:
                if _segment_contains_protected_term(segment.text, protected_term) and not _segment_contains_protected_term(
                    reviewed_text,
                    protected_term,
                ):
                    reviewed_text = segment.text
                    protected_term_violation_count += 1
                    stage_notes.append(
                        f"Protected preferred term preserved in segment {segment.id}: {protected_term}"
                    )
                    break
            if reviewed_text != segment.text:
                changed = True
                reviewed.text = reviewed_text
                reviewed.llm_reviewed_text = reviewed_text
        if allow_speaker_edits:
            reviewed_speaker = str(update.get("speaker") or segment.speaker or "").strip()
            if reviewed_speaker and reviewed_speaker != segment.speaker:
                reviewed.original_speaker = segment.speaker
                reviewed.speaker = reviewed_speaker
                changed = True
        if changed:
            corrected_segment_count += 1
            applied_change_count += 1
        elif intended_change:
            overridden_change_count += 1
        updated_segments.append(reviewed)
    return updated_segments, {
        "corrected_segment_count": corrected_segment_count,
        "returned_change_count": returned_change_count,
        "applied_change_count": applied_change_count,
        "overridden_change_count": overridden_change_count,
        "protected_term_violation_count": protected_term_violation_count,
    }, stage_notes


def _disabled_stage_result(stage_definition: Dict[str, object]) -> Dict[str, object]:
    return {
        "attempted": False,
        "status": "disabled",
        "skip_reason": "",
        "corrected_segment_count": 0,
        "protected_term_violation_count": 0,
        "returned_change_count": 0,
        "applied_change_count": 0,
        "overridden_change_count": 0,
        "window_count": 0,
        "split_event_count": 0,
        "conflicting_update_count": 0,
        "budget_used": 0,
        "mode": "disabled",
        "edit_scope": stage_definition["edit_scope"],
        "candidate_count": 0,
        "context_segment_count": 0,
        "skipped_segment_count": 0,
    }


def _skipped_stage_result(stage_definition: Dict[str, object], reason: str, attempted: bool = False) -> Dict[str, object]:
    return {
        "attempted": attempted,
        "status": "skipped",
        "skip_reason": reason,
        "corrected_segment_count": 0,
        "protected_term_violation_count": 0,
        "returned_change_count": 0,
        "applied_change_count": 0,
        "overridden_change_count": 0,
        "window_count": 0,
        "split_event_count": 0,
        "conflicting_update_count": 0,
        "budget_used": 0,
        "mode": "skipped",
        "edit_scope": stage_definition["edit_scope"],
        "candidate_count": 0,
        "context_segment_count": 0,
        "skipped_segment_count": 0,
    }


def _run_review_stage(
    segments: List[SegmentItem],
    backend_capabilities: Dict[str, object],
    stage_definition: Dict[str, object],
    preferred_terms: Optional[List[str]] = None,
    learned_rules: Optional[List[Dict[str, object]]] = None,
    calibration_session: Optional[ReviewCalibrationSession] = None,
    progress_callback: Optional[ReviewProgressCallback] = None,
    debug_context: Optional[Dict[str, object]] = None,
    editable_segment_ids: Optional[set[str]] = None,
) -> Tuple[List[SegmentItem], Dict[str, object], List[str], str]:
    stage_started = time.perf_counter()
    family_name = _review_stage_family(stage_definition["name"])
    stage_budget = (
        calibration_session.budget_for_stage(stage_definition["name"])
        if calibration_session
        else _family_default_budget(backend_capabilities, family_name)
    )
    stage_learned_rules = active_rules_for_stage(learned_rules or [], stage_definition["name"])
    if progress_callback:
        progress_callback(
            {
                "event": "stage_started",
                "stage_name": stage_definition["name"],
                "stage_label": stage_definition["label"],
                "stage_mode": stage_definition["mode"],
            }
        )
    if stage_definition["mode"] == "long_context":
        if not backend_capabilities.get("long_context_available"):
            if progress_callback:
                progress_callback(
                    {
                        "event": "stage_skipped",
                        "stage_name": stage_definition["name"],
                        "stage_label": stage_definition["label"],
                        "reason": "long_context_unavailable",
                    }
                )
                progress_callback(
                    {
                        "event": "stage_finished",
                        "stage_name": stage_definition["name"],
                        "stage_label": stage_definition["label"],
                        "status": "skipped",
                        "mode": "skipped",
                        "elapsed_seconds": time.perf_counter() - stage_started,
                    }
                )
            return (
                segments,
                _skipped_stage_result(stage_definition, "long_context_unavailable"),
                [],
                "skipped",
            )
        budget = (
            stage_budget
        )
        full_episode_cost = sum(_estimated_segment_tokens(segment) for segment in segments)
        if full_episode_cost <= budget:
            try:
                payload = _call_stage_backend_request(
                    segments,
                    backend_capabilities,
                    stage_definition,
                    "full_episode",
                    preferred_terms=preferred_terms,
                    learned_rules=stage_learned_rules,
                    debug_context=debug_context,
                )
                updates_by_id = {
                    str(item.get("id")): item
                    for item in payload.get("reviewed_segments") or []
                    if isinstance(item, dict) and item.get("id") is not None
                }
                if progress_callback:
                    progress_callback(
                        {
                            "event": "stage_response_success",
                            "stage_name": stage_definition["name"],
                            "stage_label": stage_definition["label"],
                            "mode": "full_episode",
                            "current": 1,
                            "total": 1,
                            "had_split": False,
                        }
                    )
                updated_segments, change_stats, guard_notes = _apply_stage_updates(
                    segments,
                    updates_by_id,
                    stage_definition,
                    preferred_terms=preferred_terms,
                    editable_segment_ids=editable_segment_ids,
                )
                result = (
                    updated_segments,
                    {
                        "attempted": True,
                        "status": "completed",
                        "skip_reason": "",
                        "corrected_segment_count": int(payload.get("corrected_segment_count") or change_stats["corrected_segment_count"]),
                        "edit_scope": stage_definition["edit_scope"],
                        "protected_term_violation_count": change_stats["protected_term_violation_count"],
                        "returned_change_count": change_stats["returned_change_count"],
                        "applied_change_count": change_stats["applied_change_count"],
                        "overridden_change_count": change_stats["overridden_change_count"],
                        "window_count": 1,
                        "split_event_count": 0,
                        "conflicting_update_count": 0,
                        "budget_used": budget,
                        "mode": "full_episode",
                    },
                    _normalize_episode_notes(payload) + guard_notes,
                    "full_episode",
                )
                if progress_callback:
                    progress_callback(
                        {
                            "event": "stage_finished",
                            "stage_name": stage_definition["name"],
                            "stage_label": stage_definition["label"],
                            "status": "completed",
                            "mode": "full_episode",
                            "elapsed_seconds": time.perf_counter() - stage_started,
                        }
                    )
                return result
            except RuntimeError as exc:
                if _is_transport_or_backend_failure(str(exc)):
                    raise
                if calibration_session:
                    calibration_session.note_truncation(
                        family_name,
                        full_episode_cost,
                        progress_callback=progress_callback,
                    )

        windows = _split_segments_by_token_budget(segments, budget, overlap_segments=EPISODE_QA_OVERLAP_SEGMENTS)
        updates_by_id, notes, window_stats = _collect_window_updates(
            windows,
            backend_capabilities,
            stage_definition,
            "chunked",
            preferred_terms=preferred_terms,
            learned_rules=stage_learned_rules,
            calibration_session=calibration_session,
            progress_callback=progress_callback,
            debug_context=debug_context,
        )
        updated_segments, change_stats, guard_notes = _apply_stage_updates(
            segments,
            updates_by_id,
            stage_definition,
            preferred_terms=preferred_terms,
            editable_segment_ids=editable_segment_ids,
        )
        result = (
            updated_segments,
            {
                "attempted": True,
                "status": "completed",
                "skip_reason": "",
                "corrected_segment_count": change_stats["corrected_segment_count"],
                "edit_scope": stage_definition["edit_scope"],
                "protected_term_violation_count": change_stats["protected_term_violation_count"],
                "returned_change_count": change_stats["returned_change_count"],
                "applied_change_count": change_stats["applied_change_count"],
                "overridden_change_count": change_stats["overridden_change_count"],
                "window_count": window_stats["window_count"],
                "split_event_count": window_stats["split_event_count"],
                "conflicting_update_count": window_stats["conflicting_update_count"],
                "budget_used": budget,
                "mode": "chunked",
            },
            notes + guard_notes,
            "chunked",
        )
        if progress_callback:
            progress_callback(
                {
                    "event": "stage_finished",
                    "stage_name": stage_definition["name"],
                    "stage_label": stage_definition["label"],
                    "status": "completed",
                    "mode": "chunked",
                    "elapsed_seconds": time.perf_counter() - stage_started,
                }
            )
        return result

    local_budget = stage_budget
    windows = _split_segments_by_token_budget(segments, local_budget, overlap_segments=0)
    updates_by_id, notes, window_stats = _collect_window_updates(
        windows,
        backend_capabilities,
        stage_definition,
        "local_batch",
        preferred_terms=preferred_terms,
        learned_rules=stage_learned_rules,
        calibration_session=calibration_session,
        progress_callback=progress_callback,
        debug_context=debug_context,
    )
    updated_segments, change_stats, guard_notes = _apply_stage_updates(
        segments,
        updates_by_id,
        stage_definition,
        preferred_terms=preferred_terms,
        editable_segment_ids=editable_segment_ids,
    )
    result = (
        updated_segments,
        {
            "attempted": True,
            "status": "completed",
            "skip_reason": "",
            "corrected_segment_count": change_stats["corrected_segment_count"],
            "edit_scope": stage_definition["edit_scope"],
            "protected_term_violation_count": change_stats["protected_term_violation_count"],
            "returned_change_count": change_stats["returned_change_count"],
            "applied_change_count": change_stats["applied_change_count"],
            "overridden_change_count": change_stats["overridden_change_count"],
            "window_count": window_stats["window_count"],
            "split_event_count": window_stats["split_event_count"],
            "conflicting_update_count": window_stats["conflicting_update_count"],
            "budget_used": local_budget,
            "mode": "local_batch",
        },
        notes + guard_notes,
        "local_batch",
    )
    if progress_callback:
        progress_callback(
            {
                "event": "stage_finished",
                "stage_name": stage_definition["name"],
                "stage_label": stage_definition["label"],
                "status": "completed",
                "mode": "local_batch",
                "elapsed_seconds": time.perf_counter() - stage_started,
            }
        )
    return result


def _prepare_review_segments(
    segments: List[SegmentItem],
    backend_capabilities: Dict[str, object],
    flags: Dict[str, bool],
) -> List[SegmentItem]:
    prepared: List[SegmentItem] = []
    for segment in segments:
        reviewed = deepcopy(segment)
        reviewed.original_text = segment.text
        reviewed.llm_reviewed_text = segment.text
        reviewed.review_runtime_profile = str(backend_capabilities.get("runtime_profile") or "")
        reviewed.review_backend = str(backend_capabilities.get("backend_name") or "")
        reviewed.review_model_name = str(backend_capabilities.get("review_model_name") or "")
        reviewed.review_stage_flags = flags
        prepared.append(reviewed)
    return prepared


def _build_review_intelligence_summary(
    stage_results: Dict[str, Dict[str, object]],
    enabled_stages: List[str],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]], Dict[str, object]]:
    stage_value: Dict[str, Dict[str, object]] = {}
    unique_stage_count = 0
    no_op_stage_count = 0
    protected_term_intervention_count = 0
    total_returned_change_count = 0
    total_applied_change_count = 0
    total_overridden_change_count = 0
    for stage_name in enabled_stages:
        result = stage_results.get(stage_name) or {}
        corrected_segment_count = int(result.get("corrected_segment_count") or 0)
        applied_change_count = int(result.get("applied_change_count") or 0)
        returned_change_count = int(result.get("returned_change_count") or 0)
        overridden_change_count = int(result.get("overridden_change_count") or 0)
        intervention_count = int(result.get("protected_term_violation_count") or 0)
        produced_unique_changes = corrected_segment_count > 0
        is_no_op = str(result.get("status") or "") == "completed" and corrected_segment_count == 0
        if produced_unique_changes:
            unique_stage_count += 1
        if is_no_op:
            no_op_stage_count += 1
        protected_term_intervention_count += intervention_count
        total_returned_change_count += returned_change_count
        total_applied_change_count += applied_change_count
        total_overridden_change_count += overridden_change_count
        stage_value[stage_name] = {
            "status": str(result.get("status") or ""),
            "mode": str(result.get("mode") or ""),
            "produced_unique_changes": produced_unique_changes,
            "no_op": is_no_op,
            "corrected_segment_count": corrected_segment_count,
            "returned_change_count": returned_change_count,
            "applied_change_count": applied_change_count,
            "overridden_change_count": overridden_change_count,
            "protected_term_intervention_count": intervention_count,
            "window_count": int(result.get("window_count") or 0),
            "split_event_count": int(result.get("split_event_count") or 0),
            "conflicting_update_count": int(result.get("conflicting_update_count") or 0),
            "budget_used": int(result.get("budget_used") or 0),
            "edit_scope": str(result.get("edit_scope") or ""),
            "skip_reason": str(result.get("skip_reason") or ""),
        }

    change_summary = {
        "material_change": total_applied_change_count > 0,
        "enabled_stage_count": len(enabled_stages),
        "unique_stage_count": unique_stage_count,
        "no_op_stage_count": no_op_stage_count,
        "returned_change_count": total_returned_change_count,
        "applied_change_count": total_applied_change_count,
        "overridden_change_count": total_overridden_change_count,
        "protected_term_intervention_count": protected_term_intervention_count,
        "episode_qa_added_value": bool(
            (stage_value.get("episode_qa_review") or {}).get("produced_unique_changes")
        ),
    }
    guard_interventions = {
        "protected_term_preservations": protected_term_intervention_count,
        "stages_with_interventions": [
            stage_name
            for stage_name, value in stage_value.items()
            if int(value.get("protected_term_intervention_count") or 0) > 0
        ],
    }
    return change_summary, stage_value, guard_interventions


def review_segments(
    segments: List[SegmentItem],
    runtime_review_config: Optional[Dict[str, object]],
    review_input_source: str = "inline_cleaned_segments",
    calibration_session: Optional[ReviewCalibrationSession] = None,
    progress_callback: Optional[ReviewProgressCallback] = None,
    debug_context: Optional[Dict[str, object]] = None,
    learned_rules: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Return additive reviewed segments and metadata, or a skipped/fallback result."""

    backend_capabilities = resolve_backend_capabilities(runtime_review_config or {})
    preferred_terms = _preferred_terms_payload(backend_capabilities)
    flags = build_review_stage_flags(backend_capabilities)
    stage_results = {stage["name"]: _disabled_stage_result(stage) for stage in STAGE_DEFINITIONS}
    enabled_stages = [stage["name"] for stage in STAGE_DEFINITIONS if flags.get(stage["name"], False)]
    active_rule_ids = [
        str(rule.get("rule_id") or "")
        for rule in (learned_rules or [])
        if str(rule.get("status") or "") == "approved" and str(rule.get("rule_id") or "")
    ]
    metadata = {
        "review_pipeline_version": REVIEW_PIPELINE_VERSION,
        "review_runtime_profile": backend_capabilities["runtime_profile"],
        "review_backend": backend_capabilities["backend_name"],
        "review_model_name": backend_capabilities["review_model_name"],
        "review_stage_flags": flags,
        "review_stage_results": stage_results,
        "review_enabled_stages": enabled_stages,
        "review_completed_stages": [],
        "review_skipped_stages": [],
        "review_status": "skipped",
        "review_skip_reason": "",
        "reviewed_segment_count": 0,
        "corrected_segment_count": 0,
        "episode_notes": [],
        "episode_qa_mode": "disabled",
        "review_input_source": review_input_source,
        "review_calibration": {},
        "preferred_terms": preferred_terms,
        "protected_term_violation_count": 0,
        "review_change_summary": {},
        "review_stage_value": {},
        "review_guard_interventions": {},
        "review_candidate_filter": bool(backend_capabilities.get("review_candidate_filter")),
        "review_candidate_count": 0,
        "review_context_segment_count": 0,
        "review_skipped_segment_count": 0,
        "review_candidate_reason_counts": {},
        "active_learned_rule_ids": active_rule_ids,
        "learned_rule_stage_summary": {},
        "contributing_learned_rule_ids": [],
    }

    if not backend_capabilities["any_review_enabled"]:
        metadata["review_skip_reason"] = "review_disabled"
        (
            metadata["review_change_summary"],
            metadata["review_stage_value"],
            metadata["review_guard_interventions"],
        ) = _build_review_intelligence_summary(metadata["review_stage_results"], enabled_stages)
        return {
            "attempted": False,
            "skipped": True,
            "skip_reason": "review_disabled",
            "segments": [],
            "metadata": metadata,
        }

    if not backend_capabilities["backend_ready"]:
        for stage in STAGE_DEFINITIONS:
            if flags.get(stage["name"], False):
                metadata["review_stage_results"][stage["name"]] = _skipped_stage_result(stage, "backend_unavailable")
                metadata["review_skipped_stages"].append(stage["name"])
                if progress_callback:
                    progress_callback(
                        {
                            "event": "stage_skipped",
                            "stage_name": stage["name"],
                            "stage_label": stage["label"],
                            "reason": "backend_unavailable",
                        }
                    )
        metadata["review_skip_reason"] = "backend_unavailable"
        (
            metadata["review_change_summary"],
            metadata["review_stage_value"],
            metadata["review_guard_interventions"],
        ) = _build_review_intelligence_summary(metadata["review_stage_results"], enabled_stages)
        return {
            "attempted": False,
            "skipped": True,
            "skip_reason": "backend_unavailable",
            "segments": [],
            "metadata": metadata,
        }

    candidate_reason_counts: Dict[str, int] = {}
    candidate_ids: set[str] = set()
    if backend_capabilities.get("review_candidate_filter"):
        for segment in segments:
            reasons = _segment_review_reasons(segment)
            if not reasons:
                continue
            candidate_ids.add(str(segment.id))
            for reason in reasons:
                candidate_reason_counts[reason] = candidate_reason_counts.get(reason, 0) + 1
    else:
        candidate_ids = {str(segment.id) for segment in segments}
    metadata["review_candidate_count"] = len(candidate_ids)
    metadata["review_skipped_segment_count"] = max(0, len(segments) - len(candidate_ids))
    metadata["review_candidate_reason_counts"] = candidate_reason_counts

    if calibration_session is not None and backend_capabilities.get("review_auto_calibrate"):
        calibration_session.calibrate_for_run(
            enabled_stages,
            [segment for segment in segments if str(segment.id) in candidate_ids],
            progress_callback=progress_callback,
            debug_context=debug_context,
        )
        metadata["review_calibration"] = calibration_session.metadata_snapshot()

    working_segments = _prepare_review_segments(segments, backend_capabilities, flags)
    progress_state = _load_review_progress(
        debug_context,
        backend_capabilities,
        flags,
        review_input_source,
    )
    restored_segments = progress_state.get("segments") if isinstance(progress_state, dict) else None
    if isinstance(restored_segments, list):
        restored_by_id = {str(segment.id): segment for segment in restored_segments}
        if restored_by_id and all(str(segment.id) in restored_by_id for segment in working_segments):
            working_segments = [restored_by_id.get(str(segment.id), segment) for segment in working_segments]
            metadata["review_resume_source"] = "review_progress_checkpoint"
        else:
            progress_state = {}
    else:
        progress_state = {}
    restored_completed_stages = [
        str(stage_name)
        for stage_name in progress_state.get("completed_stages") or []
        if isinstance(stage_name, str) and stage_name in enabled_stages
    ]
    restored_stage_results = progress_state.get("stage_results") if isinstance(progress_state.get("stage_results"), dict) else {}
    restored_episode_notes = [str(note) for note in progress_state.get("episode_notes") or [] if str(note).strip()]
    metadata["review_resumed_completed_stages"] = list(restored_completed_stages)
    attempted_any = False
    total_corrected = 0
    episode_notes: List[str] = []
    total_enabled = len(enabled_stages)
    completed_count = 0

    def record_completed_stage(stage_name: str, stage_result: Dict[str, object]):
        nonlocal attempted_any, total_corrected, completed_count
        metadata["review_completed_stages"].append(stage_name)
        attempted_any = True
        total_corrected += int(stage_result.get("corrected_segment_count") or 0)
        metadata["protected_term_violation_count"] += int(stage_result.get("protected_term_violation_count") or 0)
        completed_count += 1

    for stage_index, stage_definition in enumerate(STAGE_DEFINITIONS, start=1):
        stage_name = stage_definition["name"]
        if not flags.get(stage_name, False):
            continue
        if stage_name in restored_completed_stages:
            stage_result = restored_stage_results.get(stage_name)
            if not isinstance(stage_result, dict):
                continue
            metadata["review_stage_results"][stage_name] = stage_result
            record_completed_stage(stage_name, stage_result)
            if stage_name == "episode_qa_review":
                metadata["episode_qa_mode"] = str(stage_result.get("mode") or "")
            continue
        if progress_callback:
            progress_callback(
                {
                    "event": "stage_index",
                    "stage_name": stage_name,
                    "stage_label": stage_definition["label"],
                    "current": stage_index - sum(1 for stage in STAGE_DEFINITIONS[: stage_index - 1] if not flags.get(stage["name"], False)),
                    "total": total_enabled,
                }
            )
        try:
            stage_scope, editable_ids = _review_scope(working_segments, candidate_ids, stage_definition)
            if not stage_scope:
                stage_result = {
                    **_skipped_stage_result(stage_definition, "no_changed_or_uncertain_segments"),
                    "attempted": False,
                    "status": "completed",
                    "skip_reason": "no_changed_or_uncertain_segments",
                    "budget_used": 0,
                    "mode": "candidate_filter",
                    "candidate_count": 0,
                    "context_segment_count": 0,
                    "skipped_segment_count": len(working_segments),
                }
                stage_notes = []
                stage_mode = "candidate_filter"
                metadata["review_context_segment_count"] = max(
                    int(metadata.get("review_context_segment_count") or 0),
                    0,
                )
            else:
                stage_result_input = _run_review_stage(
                    stage_scope,
                    backend_capabilities,
                    stage_definition,
                    preferred_terms=preferred_terms,
                    learned_rules=learned_rules,
                    calibration_session=calibration_session,
                    progress_callback=progress_callback,
                    debug_context=debug_context,
                    editable_segment_ids=editable_ids,
                )
                reviewed_scope, stage_result, stage_notes, stage_mode = stage_result_input
                reviewed_by_id = {str(segment.id): segment for segment in reviewed_scope}
                working_segments = [reviewed_by_id.get(str(segment.id), segment) for segment in working_segments]
                stage_result["candidate_count"] = len(candidate_ids)
                stage_result["context_segment_count"] = len(stage_scope)
                stage_result["skipped_segment_count"] = max(0, len(working_segments) - len(candidate_ids))
                metadata["review_context_segment_count"] = max(
                    int(metadata.get("review_context_segment_count") or 0),
                    len(stage_scope),
                )
        except Exception as exc:
            reason = str(exc)
            if _is_transport_or_backend_failure(reason):
                if "connection failed" in reason.lower():
                    reason = "backend_unavailable"
                elif reason.lower().startswith("review backend http "):
                    reason = "http_error"
            stage_result = _skipped_stage_result(stage_definition, reason, attempted=True)
            stage_notes = []
            stage_mode = "skipped"
            if progress_callback:
                progress_callback(
                    {
                        "event": "stage_skipped",
                        "stage_name": stage_name,
                        "stage_label": stage_definition["label"],
                        "reason": reason,
                    }
                )
                progress_callback(
                    {
                        "event": "stage_finished",
                        "stage_name": stage_name,
                        "stage_label": stage_definition["label"],
                        "status": "skipped",
                        "mode": "skipped",
                        "elapsed_seconds": 0.0,
                    }
                )
        metadata["review_stage_results"][stage_name] = stage_result
        if stage_result["status"] == "completed":
            record_completed_stage(stage_name, stage_result)
        elif stage_result["status"] == "skipped":
            metadata["review_skipped_stages"].append(stage_name)
        if stage_name == "episode_qa_review":
            metadata["episode_qa_mode"] = stage_mode
        stage_rule_ids = [
            str(rule.get("rule_id") or "")
            for rule in (learned_rules or [])
            if str(rule.get("status") or "") == "approved" and str(rule.get("stage_target") or "") == stage_name
        ]
        metadata["learned_rule_stage_summary"][stage_name] = {
            "active_rule_ids": stage_rule_ids,
            "materially_contributed": bool(stage_rule_ids and int(stage_result.get("corrected_segment_count") or 0) > 0),
        }
        if stage_notes:
            episode_notes.extend(f"{stage_name}: {note}" for note in stage_notes)
        if stage_result["status"] == "completed":
            _save_review_progress(
                debug_context,
                backend_capabilities,
                flags,
                review_input_source,
                working_segments,
                metadata["review_completed_stages"],
                metadata["review_stage_results"],
                episode_notes,
            )

    if attempted_any:
        metadata["review_status"] = "completed_with_stage_failures" if metadata["review_skipped_stages"] else "completed"
        metadata["reviewed_segment_count"] = len(working_segments)
        metadata["corrected_segment_count"] = total_corrected
        metadata["episode_notes"] = restored_episode_notes + episode_notes
        if metadata["review_skipped_stages"]:
            first_stage = metadata["review_skipped_stages"][0]
            metadata["review_skip_reason"] = metadata["review_stage_results"][first_stage]["skip_reason"]
        metadata["review_calibration"] = calibration_session.metadata_snapshot() if calibration_session else {}
        metadata["contributing_learned_rule_ids"] = [
            rule_id
            for stage_payload in metadata["learned_rule_stage_summary"].values()
            for rule_id in stage_payload.get("active_rule_ids") or []
            if stage_payload.get("materially_contributed")
        ]
        (
            metadata["review_change_summary"],
            metadata["review_stage_value"],
            metadata["review_guard_interventions"],
        ) = _build_review_intelligence_summary(metadata["review_stage_results"], enabled_stages)
        return {
            "attempted": True,
            "skipped": False,
            "skip_reason": metadata["review_skip_reason"],
            "segments": working_segments,
            "metadata": metadata,
        }

    metadata["review_status"] = "skipped"
    if metadata["review_skipped_stages"]:
        first_stage = metadata["review_skipped_stages"][0]
        metadata["review_skip_reason"] = metadata["review_stage_results"][first_stage]["skip_reason"]
    metadata["episode_notes"] = episode_notes
    metadata["review_calibration"] = calibration_session.metadata_snapshot() if calibration_session else {}
    (
        metadata["review_change_summary"],
        metadata["review_stage_value"],
        metadata["review_guard_interventions"],
    ) = _build_review_intelligence_summary(metadata["review_stage_results"], enabled_stages)
    return {
        "attempted": True,
        "skipped": True,
        "skip_reason": metadata["review_skip_reason"],
        "segments": [],
        "metadata": metadata,
    }
