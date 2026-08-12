"""Dependency-light transcript and speaker-attribution metrics."""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import permutations
from typing import Dict, Iterable, List, Sequence, Tuple


def normalized_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", str(text or "").lower())


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (0 if ref_item == hyp_item else 1),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference_text: str, hypothesis_text: str) -> Dict[str, object]:
    reference = normalized_words(reference_text)
    hypothesis = normalized_words(hypothesis_text)
    errors = edit_distance(reference, hypothesis)
    return {
        "errors": errors,
        "reference_words": len(reference),
        "hypothesis_words": len(hypothesis),
        "wer": errors / len(reference) if reference else (0.0 if not hypothesis else 1.0),
    }


def attributed_tokens(segments: Iterable[Dict[str, object]]) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "UNKNOWN").strip().upper()
        tokens.extend((speaker, word) for word in normalized_words(str(segment.get("text") or "")))
    return tokens


def speaker_attributed_wer(reference_segments: List[Dict[str, object]], hypothesis_segments: List[Dict[str, object]]) -> Dict[str, object]:
    reference = attributed_tokens(reference_segments)
    hypothesis = attributed_tokens(hypothesis_segments)
    errors = edit_distance(reference, hypothesis)
    return {
        "errors": errors,
        "reference_words": len(reference),
        "speaker_attributed_wer": errors / len(reference) if reference else (0.0 if not hypothesis else 1.0),
    }


def timestamp_error(reference_segments: List[Dict[str, object]], hypothesis_segments: List[Dict[str, object]]) -> Dict[str, object]:
    hypothesis_by_id = {str(segment.get("id")): segment for segment in hypothesis_segments}
    errors: List[float] = []
    for index, reference in enumerate(reference_segments):
        hypothesis = hypothesis_by_id.get(str(reference.get("id")))
        if hypothesis is None and index < len(hypothesis_segments):
            hypothesis = hypothesis_segments[index]
        if not isinstance(hypothesis, dict):
            continue
        for field in ("start", "end"):
            try:
                errors.append(abs(float(reference[field]) - float(hypothesis[field])))
            except (KeyError, TypeError, ValueError):
                continue
    ordered = sorted(errors)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)) if ordered else 0
    return {
        "boundary_count": len(errors),
        "mean_absolute_error_seconds": sum(errors) / len(errors) if errors else None,
        "p95_absolute_error_seconds": ordered[p95_index] if ordered else None,
    }


def host_classification(reference_segments: List[Dict[str, object]], hypothesis_segments: List[Dict[str, object]]) -> Dict[str, object]:
    hypothesis_by_id = {str(segment.get("id")): segment for segment in hypothesis_segments}
    true_positive = false_positive = false_negative = 0
    for index, reference in enumerate(reference_segments):
        hypothesis = hypothesis_by_id.get(str(reference.get("id")))
        if hypothesis is None and index < len(hypothesis_segments):
            hypothesis = hypothesis_segments[index]
        reference_host = str(reference.get("speaker") or "").upper() == "HOST"
        hypothesis_host = isinstance(hypothesis, dict) and str(hypothesis.get("speaker") or "").upper() == "HOST"
        true_positive += int(reference_host and hypothesis_host)
        false_positive += int(not reference_host and hypothesis_host)
        false_negative += int(reference_host and not hypothesis_host)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
    }


def diarization_error_rate(
    reference_turns: List[Dict[str, object]],
    hypothesis_turns: List[Dict[str, object]],
    frame_seconds: float = 0.25,
    collar_seconds: float = 0.0,
) -> Dict[str, object]:
    if not reference_turns:
        return {"scored_seconds": 0.0, "error_seconds": 0.0, "diarization_error_rate": None}
    end_time = max(float(turn.get("end") or 0.0) for turn in reference_turns)

    def active_speakers(turns: List[Dict[str, object]], timestamp: float) -> Tuple[str, ...]:
        matches = [
            str(turn.get("speaker") or "UNKNOWN").upper()
            for turn in turns
            if float(turn.get("start") or 0.0) <= timestamp < float(turn.get("end") or 0.0)
        ]
        return tuple(sorted(set(matches)))

    frame_count = max(1, math.ceil(end_time / frame_seconds))
    boundaries = [float(turn.get(field) or 0.0) for turn in reference_turns for field in ("start", "end")]
    frames = []
    for frame_index in range(frame_count):
        timestamp = (frame_index + 0.5) * frame_seconds
        if collar_seconds and any(abs(timestamp - boundary) <= collar_seconds for boundary in boundaries):
            continue
        reference_speakers = active_speakers(reference_turns, timestamp)
        if reference_speakers:
            frames.append((reference_speakers, active_speakers(hypothesis_turns, timestamp)))

    reference_labels = sorted({label for reference, _ in frames for label in reference})
    hypothesis_labels = sorted({label for _, hypothesis in frames for label in hypothesis})
    padded_reference = reference_labels + [f"__EXTRA_{index}" for index in range(max(0, len(hypothesis_labels) - len(reference_labels)))]
    candidates = permutations(padded_reference, len(hypothesis_labels)) if len(hypothesis_labels) <= 8 else [tuple(padded_reference[: len(hypothesis_labels)])]
    errors = len(frames)
    best_mapping: Dict[str, str] = {}
    for mapped in candidates:
        mapping = dict(zip(hypothesis_labels, mapped))
        candidate_errors = sum(
            int(tuple(sorted(mapping.get(label, label) for label in hypothesis)) != reference)
            for reference, hypothesis in frames
        )
        if candidate_errors < errors:
            errors = candidate_errors
            best_mapping = mapping
    scored = len(frames)
    mapped_frames = [
        (reference, tuple(sorted(best_mapping.get(label, label) for label in hypothesis)))
        for reference, hypothesis in frames
    ]
    miss_frames = sum(bool(reference) and not hypothesis for reference, hypothesis in mapped_frames)
    overlap_frames = sum((len(reference) > 1 or len(hypothesis) > 1) and reference != hypothesis for reference, hypothesis in mapped_frames)
    label_frames = sum(bool(hypothesis) and len(reference) == len(hypothesis) and reference != hypothesis for reference, hypothesis in mapped_frames)
    reference_boundaries = sorted({float(turn.get(field) or 0.0) for turn in reference_turns for field in ("start", "end")})
    hypothesis_boundaries = sorted({float(turn.get(field) or 0.0) for turn in hypothesis_turns for field in ("start", "end")})
    boundary_errors = [min(abs(value - candidate) for candidate in hypothesis_boundaries) for value in reference_boundaries] if hypothesis_boundaries else []
    return {
        "scored_seconds": scored * frame_seconds,
        "error_seconds": errors * frame_seconds,
        "diarization_error_rate": errors / scored if scored else None,
        "speaker_mapping": best_mapping,
        "speaker_count_error": abs(len(reference_labels) - len(hypothesis_labels)),
        "missed_speech_seconds": miss_frames * frame_seconds,
        "overlap_error_seconds": overlap_frames * frame_seconds,
        "label_mapping_error_seconds": label_frames * frame_seconds,
        "segmentation_boundary_mean_error_seconds": sum(boundary_errors) / len(boundary_errors) if boundary_errors else None,
        "collar_seconds": collar_seconds,
    }


def glossary_preservation(terms: Iterable[str], reference_text: str, hypothesis_text: str) -> Dict[str, object]:
    expected = [term for term in terms if str(term).strip() and str(term) in reference_text]
    preserved = [term for term in expected if str(term) in hypothesis_text]
    return {
        "expected_count": len(expected),
        "preserved_count": len(preserved),
        "missing_terms": sorted(set(expected) - set(preserved)),
        "preservation_rate": len(preserved) / len(expected) if expected else 1.0,
    }


def aggregate_metric_counts(items: List[Dict[str, object]], error_key: str, denominator_key: str, rate_key: str) -> Dict[str, object]:
    errors = sum(int(item.get(error_key) or 0) for item in items)
    denominator = sum(int(item.get(denominator_key) or 0) for item in items)
    return {error_key: errors, denominator_key: denominator, rate_key: errors / denominator if denominator else 0.0}
