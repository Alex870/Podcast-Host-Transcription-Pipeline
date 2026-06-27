import json
from pathlib import Path
from typing import Dict, List, Optional


RUNTIME_PROFILES = {"baseline_16gb", "high_context_5090", "custom"}
REVIEW_BACKENDS = {"none", "lm_studio", "vllm"}
DEFAULT_RUNTIME_PROFILE = "baseline_16gb"
DEFAULT_REVIEW_BACKEND = "none"


def _coerce_bool(value, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_string_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def resolve_review_runtime_config(raw_config: Optional[Dict[str, object]]) -> Dict[str, object]:
    payload = dict(raw_config or {})
    runtime_profile = str(payload.get("runtime_profile") or DEFAULT_RUNTIME_PROFILE).strip().lower()
    if runtime_profile not in RUNTIME_PROFILES:
        runtime_profile = DEFAULT_RUNTIME_PROFILE

    backend = str(payload.get("backend") or DEFAULT_REVIEW_BACKEND).strip().lower()
    if backend not in REVIEW_BACKENDS:
        backend = DEFAULT_REVIEW_BACKEND

    review_base_url = str(payload.get("review_base_url") or "").strip()
    review_model_name = str(payload.get("review_model_name") or "").strip()
    review_debug = _coerce_bool(payload.get("review_debug"), False) is True
    review_debug_dir = str(payload.get("review_debug_dir") or "").strip()
    review_auto_calibrate = _coerce_bool(payload.get("review_auto_calibrate"), None)
    review_auto_adapt_upward = _coerce_bool(payload.get("review_auto_adapt_upward"), None)
    preferred_terms = _coerce_string_list(payload.get("preferred_terms"))

    profile_defaults = {
        "baseline_16gb": {
            "max_context_budget": 16000,
            "structured_output_support": False,
            "transcript_qa_available": False,
            "episode_wide_correction_available": False,
            "default_enable_review": False,
            "default_enable_episode_qa": False,
        },
        "high_context_5090": {
            "max_context_budget": 131072,
            "structured_output_support": True,
            "transcript_qa_available": True,
            "episode_wide_correction_available": True,
            "default_enable_review": True,
            "default_enable_episode_qa": True,
        },
        "custom": {
            "max_context_budget": int(payload.get("review_context_budget") or 32768),
            "structured_output_support": _coerce_bool(payload.get("review_structured_output_support"), False) is True,
            "transcript_qa_available": _coerce_bool(payload.get("review_transcript_qa_available"), False) is True,
            "episode_wide_correction_available": _coerce_bool(payload.get("review_episode_wide_correction_available"), False) is True,
            "default_enable_review": False,
            "default_enable_episode_qa": False,
        },
    }[runtime_profile]

    transcript_cleanup_review = _coerce_bool(
        payload.get("transcript_cleanup_review"),
        profile_defaults["default_enable_review"],
    ) is True
    glossary_correction_review = _coerce_bool(
        payload.get("glossary_correction_review"),
        profile_defaults["default_enable_review"],
    ) is True
    speaker_consistency_review = _coerce_bool(
        payload.get("speaker_consistency_review"),
        profile_defaults["default_enable_review"],
    ) is True
    episode_qa_review = _coerce_bool(
        payload.get("episode_qa_review"),
        profile_defaults["default_enable_episode_qa"],
    ) is True

    any_review_enabled = any(
        [
            transcript_cleanup_review,
            glossary_correction_review,
            speaker_consistency_review,
            episode_qa_review,
        ]
    )

    effective_backend = backend
    if not any_review_enabled:
        effective_backend = "none"
    if backend == "none":
        transcript_cleanup_review = False
        glossary_correction_review = False
        speaker_consistency_review = False
        episode_qa_review = False
        any_review_enabled = False

    effective_review_auto_calibrate = any_review_enabled if review_auto_calibrate is None else review_auto_calibrate is True
    if not any_review_enabled:
        effective_review_auto_calibrate = False

    return {
        "runtime_profile": runtime_profile,
        "backend": backend,
        "effective_backend": effective_backend,
        "review_base_url": review_base_url,
        "review_model_name": review_model_name,
        "review_debug": review_debug,
        "review_debug_dir": review_debug_dir,
        "review_auto_calibrate": effective_review_auto_calibrate,
        "review_auto_adapt_upward": True if review_auto_adapt_upward is None else review_auto_adapt_upward is True,
        "preferred_terms": preferred_terms,
        "max_context_budget": int(profile_defaults["max_context_budget"]),
        "structured_output_support": bool(profile_defaults["structured_output_support"]),
        "transcript_qa_available": bool(profile_defaults["transcript_qa_available"]),
        "episode_wide_correction_available": bool(profile_defaults["episode_wide_correction_available"]),
        "transcript_cleanup_review": transcript_cleanup_review,
        "glossary_correction_review": glossary_correction_review,
        "speaker_consistency_review": speaker_consistency_review,
        "episode_qa_review": episode_qa_review,
        "any_review_enabled": any_review_enabled,
        "backend_ready": bool(review_base_url and review_model_name and effective_backend != "none"),
    }


def load_replacement_map(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}

    raw_text = file_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        lines = raw_text.splitlines()
        bad_line = lines[exc.lineno - 1] if 0 <= exc.lineno - 1 < len(lines) else ""
        pointer = " " * max(exc.colno - 1, 0) + "^"
        raise RuntimeError(
            f"Invalid JSON in replacement map file: {file_path}\n"
            f"JSON error at line {exc.lineno}, column {exc.colno}: {exc.msg}\n"
            f"{bad_line}\n"
            f"{pointer}\n"
            "Replacement maps must be strict JSON: use double quotes, no comments, and no trailing commas."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Replacement map file must contain a JSON object: {file_path}")

    normalized = {}
    for preferred, aliases in payload.items():
        if isinstance(aliases, list):
            normalized[preferred] = [alias for alias in aliases if isinstance(alias, str) and alias.strip()]
    return normalized
