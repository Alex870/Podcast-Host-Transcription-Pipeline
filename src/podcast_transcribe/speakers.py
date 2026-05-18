from collections import defaultdict
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

