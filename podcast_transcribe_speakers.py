from typing import Dict, Optional

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
    if final_host_speaker and final_host_speaker in speaker_embeddings:
        return merge_profile(existing_profile, speaker_embeddings[final_host_speaker])
    return candidate_profile if final_host_speaker else existing_profile
