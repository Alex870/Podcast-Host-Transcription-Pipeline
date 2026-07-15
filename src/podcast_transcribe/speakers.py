from collections import defaultdict
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def average_embeddings(embeddings):
    if not embeddings:
        return None
    merged = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(merged)
    if norm == 0:
        return None
    return merged / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def merge_profile(existing: Optional[np.ndarray], new_embedding: np.ndarray) -> np.ndarray:
    if existing is None:
        merged = new_embedding
    else:
        merged = (existing + new_embedding) / 2.0
    norm = np.linalg.norm(merged)
    if norm == 0:
        return new_embedding
    return merged / norm


def final_host_profile_update(
    existing_profile: Optional[np.ndarray],
    speaker_embeddings: Dict[str, np.ndarray],
    final_host_speaker: Optional[str],
    candidate_profile: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Merge the saved host profile with the final selected host speaker embedding."""

    if final_host_speaker and final_host_speaker in speaker_embeddings:
        return merge_profile(existing_profile, speaker_embeddings[final_host_speaker])
    return candidate_profile if final_host_speaker else existing_profile


def reference_sample_quality(
    duration_seconds: float,
    rms: Optional[float] = None,
    peak: Optional[float] = None,
    speech_ratio: Optional[float] = None,
) -> Dict[str, object]:
    warnings: List[str] = []
    score = 1.0

    if duration_seconds < 8:
        warnings.append("sample is very short")
        score -= 0.35
    elif duration_seconds < 20:
        warnings.append("sample is shorter than recommended")
        score -= 0.15
    if duration_seconds > 180:
        warnings.append("sample is longer than needed")
        score -= 0.05
    if rms is not None and rms < 0.005:
        warnings.append("sample is very quiet")
        score -= 0.2
    if peak is not None and peak > 0.98:
        warnings.append("sample may be clipped")
        score -= 0.2
    if speech_ratio is not None and speech_ratio < 0.55:
        warnings.append("sample appears to contain substantial silence or non-speech")
        score -= 0.2

    if score >= 0.8:
        rating = "good"
    elif score >= 0.55:
        rating = "usable"
    else:
        rating = "poor"

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "rating": rating,
        "warnings": warnings,
    }


def speaker_aggregate_stats(
    rows: Iterable[Dict[str, object]],
    speaker_field: str = "host_label",
) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "episode_count": 0,
            "total_duration_seconds": 0.0,
            "similarity_scores": [],
            "review_priority_scores": [],
        }
    )
    for row in rows:
        speaker = str(row.get(speaker_field) or "").strip()
        if not speaker:
            continue
        item = stats[speaker]
        item["episode_count"] += 1
        item["total_duration_seconds"] += float(row.get("host_duration_seconds") or 0.0)
        if row.get("top_host_similarity") not in ("", None):
            item["similarity_scores"].append(float(row["top_host_similarity"]))
        if row.get("review_priority_score") not in ("", None):
            item["review_priority_scores"].append(float(row["review_priority_score"]))

    result = {}
    for speaker, item in stats.items():
        scores = item.pop("similarity_scores")
        priorities = item.pop("review_priority_scores")
        item["average_similarity"] = round(sum(scores) / len(scores), 4) if scores else ""
        item["min_similarity"] = round(min(scores), 4) if scores else ""
        item["average_review_priority"] = round(sum(priorities) / len(priorities), 2) if priorities else ""
        item["total_duration_seconds"] = round(item["total_duration_seconds"], 2)
        result[speaker] = item
    return result


def detect_speaker_similarity_drift(
    current_scores: Dict[str, float],
    historical_scores: Dict[str, List[float]],
    drop_threshold: float = 0.12,
) -> List[Dict[str, object]]:
    """Flag speaker-match scores that drop sharply compared with prior episode history."""

    alerts = []
    for speaker, current in current_scores.items():
        history = historical_scores.get(speaker) or []
        if len(history) < 2:
            continue
        baseline = sum(history) / len(history)
        drop = baseline - current
        if drop >= drop_threshold:
            alerts.append(
                {
                    "speaker": speaker,
                    "current_similarity": round(current, 4),
                    "historical_average_similarity": round(baseline, 4),
                    "drop": round(drop, 4),
                    "review_reason": "speaker similarity dropped below historical pattern",
                }
            )
    return alerts


def promotion_candidates(
    rows: Iterable[Dict[str, object]],
    min_episode_count: int = 3,
    min_total_seconds: float = 600.0,
) -> List[Dict[str, object]]:
    stats = speaker_aggregate_stats(rows)
    candidates = []
    for speaker, item in stats.items():
        if speaker.upper().startswith("SPEAKER_") and (
            item["episode_count"] >= min_episode_count
            or item["total_duration_seconds"] >= min_total_seconds
        ):
            candidates.append(
                {
                    "speaker": speaker,
                    "episode_count": item["episode_count"],
                    "total_duration_seconds": item["total_duration_seconds"],
                    "recommendation": "review as recurring known speaker",
                }
            )
    return candidates


def _speaker_family(provider: Optional[Dict[str, object]]) -> str:
    if not isinstance(provider, dict):
        return "unknown"
    return f"{provider.get('provider', '')}:{provider.get('model', '')}"


def calibrate_speaker_thresholds(
    positive_pairs: Iterable[Dict[str, object]],
    negative_pairs: Iterable[Dict[str, object]],
    short_turn_pairs: Optional[Iterable[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Choose a deterministic threshold from positive/negative reference pairs."""

    positives = [float(item["similarity"]) for item in positive_pairs if isinstance(item, dict) and item.get("similarity") is not None]
    negatives = [float(item["similarity"]) for item in negative_pairs if isinstance(item, dict) and item.get("similarity") is not None]
    if not positives or not negatives:
        raise ValueError("Threshold calibration requires at least one positive and one negative pair.")
    candidates = sorted({round(value, 6) for value in positives + negatives})
    candidates = sorted(set(candidates + [0.0, 1.0]))
    best = None
    for threshold in candidates:
        tp = sum(value >= threshold for value in positives)
        fp = sum(value >= threshold for value in negatives)
        fn = len(positives) - tp
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score = (f1, precision, recall, -threshold)
        if best is None or score > best[0]:
            best = (score, threshold, tp, fp, fn, precision, recall, f1)
    assert best is not None
    short_turn = list(short_turn_pairs or [])
    short_values = [float(item["similarity"]) for item in short_turn if isinstance(item, dict) and item.get("similarity") is not None]
    _, threshold, tp, fp, fn, precision, recall, f1 = best
    return {
        "calibration_version": 1,
        "threshold": float(threshold),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "short_turn_count": len(short_values),
        "positive_min": min(positives),
        "negative_max": max(negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"true_positive": tp, "false_positive": fp, "false_negative": fn},
        "short_turn_pass_rate": sum(value >= threshold for value in short_values) / len(short_values) if short_values else None,
    }


def validate_profile_family(active_profile: Optional[Dict[str, object]], candidate_profile: Dict[str, object]) -> Dict[str, object]:
    """Validate provider family and vector dimensions before a profile can move."""

    candidate_provider = candidate_profile.get("embedding_provider")
    candidate_embedding = candidate_profile.get("embedding")
    errors = []
    if not isinstance(candidate_provider, dict) or not str(candidate_provider.get("provider") or ""):
        errors.append("candidate_provider_identity_missing")
    if not isinstance(candidate_embedding, list) or not candidate_embedding:
        errors.append("candidate_embedding_missing")
    candidate_family = _speaker_family(candidate_provider if isinstance(candidate_provider, dict) else None)
    if isinstance(active_profile, dict):
        active_family = _speaker_family(active_profile.get("embedding_provider"))
        if active_family != "unknown" and active_family != candidate_family:
            errors.append("embedding_family_mismatch")
        active_dimension = int(active_profile.get("embedding_dimension") or len(active_profile.get("embedding") or []))
        if active_dimension and len(candidate_embedding or []) != active_dimension:
            errors.append("embedding_dimension_mismatch")
    return {"valid": not errors, "errors": errors, "candidate_family": candidate_family}


def _profile_audit_path(profile_path: Path) -> Path:
    return profile_path.with_name(profile_path.name + ".audit.jsonl")


def _append_profile_audit(profile_path: Path, action: str, payload: Dict[str, object]) -> None:
    audit_path = _profile_audit_path(profile_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "action": action, **payload}
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def stage_speaker_profile_promotion(
    profile_path: Path,
    candidate_profile: Dict[str, object],
    evaluation_report: Dict[str, object],
) -> Path:
    """Write a pending candidate without changing the active profile."""

    active = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else None
    validation = validate_profile_family(active, candidate_profile)
    if not validation["valid"]:
        raise ValueError("Cannot stage incompatible speaker profile: " + ", ".join(validation["errors"]))
    if not bool(evaluation_report.get("passed")):
        raise ValueError("Cannot stage a profile whose calibration/promotion report failed.")
    pending_path = profile_path.with_name(profile_path.name + ".candidate.json")
    payload = {
        **candidate_profile,
        "promotion_status": "pending_review",
        "promotion_report": evaluation_report,
        "candidate_family": validation["candidate_family"],
        "candidate_fingerprint": hashlib.sha256(json.dumps(candidate_profile, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
    }
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _append_profile_audit(profile_path, "profile_candidate_staged", {"candidate_path": str(pending_path), "candidate_family": validation["candidate_family"]})
    return pending_path


def approve_speaker_profile_promotion(profile_path: Path, reviewer_id: str) -> Dict[str, object]:
    pending_path = profile_path.with_name(profile_path.name + ".candidate.json")
    if not pending_path.exists():
        raise FileNotFoundError(f"No staged speaker profile candidate: {pending_path}")
    candidate = json.loads(pending_path.read_text(encoding="utf-8"))
    validation = validate_profile_family(
        json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else None,
        candidate,
    )
    if not validation["valid"]:
        raise ValueError("Cannot approve incompatible speaker profile: " + ", ".join(validation["errors"]))
    if not bool((candidate.get("promotion_report") or {}).get("passed")):
        raise ValueError("Cannot approve a failed speaker profile promotion report.")
    backup_path = profile_path.with_name(profile_path.name + ".bak")
    if profile_path.exists():
        shutil.copy2(profile_path, backup_path)
    promoted = {key: value for key, value in candidate.items() if key not in {"promotion_status", "candidate_family", "candidate_fingerprint"}}
    promoted["promotion"] = {
        "status": "approved",
        "reviewer_id": str(reviewer_id).strip() or "anonymous",
        "approved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report_fingerprint": hashlib.sha256(json.dumps(candidate.get("promotion_report") or {}, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(promoted, indent=2, ensure_ascii=False), encoding="utf-8")
    pending_path.unlink()
    _append_profile_audit(profile_path, "profile_promotion_approved", {"reviewer_id": reviewer_id, "backup_path": str(backup_path) if backup_path.exists() else ""})
    return {"status": "approved", "profile_path": str(profile_path), "backup_path": str(backup_path) if backup_path.exists() else ""}


def rollback_speaker_profile_promotion(profile_path: Path, reviewer_id: str) -> Dict[str, object]:
    backup_path = profile_path.with_name(profile_path.name + ".bak")
    if not backup_path.exists():
        raise FileNotFoundError(f"No reversible speaker profile backup: {backup_path}")
    shutil.copy2(backup_path, profile_path)
    _append_profile_audit(profile_path, "profile_promotion_rolled_back", {"reviewer_id": reviewer_id, "backup_path": str(backup_path)})
    return {"status": "rolled_back", "profile_path": str(profile_path), "backup_path": str(backup_path)}
