"""Dedicated benchmark harness for additive tier-2 review models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from podcast_transcribe.contract import validate_transcript_payload
from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.review import (
    STAGE_DEFINITIONS,
    ReviewCalibrationSession,
    _estimated_segment_tokens,
    _execute_stage_backend_request,
    _review_stage_family,
    _split_segments_by_token_budget,
    enrich_backend_capabilities_with_identity,
    resolve_backend_capabilities,
    review_segments,
)


def default_fixture_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "benchmarks" / "review_fixtures"


def _load_cleaned_payload(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Benchmark fixture is not a JSON object: {path}")
    errors = validate_transcript_payload(payload)
    if errors:
        raise RuntimeError(f"Benchmark fixture failed transcript validation: {path} ({'; '.join(errors[:5])})")
    return payload


def _segment_items_from_cleaned_payload(payload: Dict[str, object]) -> List[SegmentItem]:
    rebuilt_segments: List[SegmentItem] = []
    for index, raw_segment in enumerate(payload.get("segments") or []):
        if not isinstance(raw_segment, dict):
            raise RuntimeError(f"Segment {index} in benchmark fixture is not an object.")
        words_payload = raw_segment.get("words") or []
        rebuilt_segments.append(
            SegmentItem(
                id=int(raw_segment["id"]),
                start=float(raw_segment["start"]),
                end=float(raw_segment["end"]),
                text=str(raw_segment["text"]),
                speaker=str(raw_segment.get("speaker") or ""),
                avg_logprob=raw_segment.get("avg_logprob"),
                no_speech_prob=raw_segment.get("no_speech_prob"),
                words=[
                    WordItem(
                        start=word.get("start") if isinstance(word, dict) else None,
                        end=word.get("end") if isinstance(word, dict) else None,
                        word=str(word.get("word") or "") if isinstance(word, dict) else "",
                        speaker=str(word.get("speaker") or raw_segment.get("speaker") or "") if isinstance(word, dict) else "",
                    )
                    for word in words_payload
                    if isinstance(word, dict)
                ],
                original_text=raw_segment.get("original_text"),
                cleanup_applied=bool(raw_segment.get("cleanup_applied")),
                cleanup_level=str(raw_segment.get("cleanup_level") or ""),
                manual_correction_applied=bool(raw_segment.get("manual_correction_applied")),
                original_speaker=raw_segment.get("original_speaker"),
            )
        )
    return rebuilt_segments


def load_benchmark_fixtures(fixture_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    root = fixture_dir or default_fixture_dir()
    fixtures: List[Dict[str, object]] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Benchmark fixture definition is not an object: {path}")
        cleaned_payload = payload.get("cleaned_payload")
        expected = payload.get("expected")
        if not isinstance(cleaned_payload, dict) or not isinstance(expected, dict):
            raise RuntimeError(f"Benchmark fixture is missing cleaned_payload or expected: {path}")
        transcript_errors = validate_transcript_payload(cleaned_payload)
        if transcript_errors:
            raise RuntimeError(f"Benchmark fixture cleaned_payload failed transcript validation: {path} ({'; '.join(transcript_errors[:5])})")
        fixtures.append(
            {
                "name": str(payload.get("name") or path.stem),
                "focus_stage": str(payload.get("focus_stage") or "transcript_cleanup_review"),
                "description": str(payload.get("description") or ""),
                "cleaned_payload": cleaned_payload,
                "expected": expected,
                "preferred_terms": [str(term).strip() for term in (payload.get("preferred_terms") or []) if str(term).strip()],
                "expected_unchanged_segment_ids": [int(value) for value in (payload.get("expected_unchanged_segment_ids") or [])],
                "max_allowed_changed_segment_count": (
                    int(payload.get("max_allowed_changed_segment_count"))
                    if payload.get("max_allowed_changed_segment_count") is not None
                    else None
                ),
                "benchmark_tags": [str(tag).strip() for tag in (payload.get("benchmark_tags") or []) if str(tag).strip()],
                "path": str(path),
            }
        )
    if not fixtures:
        raise RuntimeError(f"No benchmark fixtures found in {root}")
    return fixtures


class BenchmarkTelemetry:
    def __init__(self):
        self.started = time.perf_counter()
        self.first_success_seconds: Optional[float] = None
        self.responses_processed = 0
        self.window_events = 0
        self.chunk_events = 0
        self.adaptive_split_count = 0
        self.budget_reduction_count = 0
        self.budget_increase_count = 0
        self.stage_timings: Dict[str, float] = {}
        self.stage_statuses: Dict[str, str] = {}

    def callback(self, event: Dict[str, object]):
        event_type = str(event.get("event") or "")
        if event_type == "stage_window_progress":
            mode = str(event.get("mode") or "")
            if mode == "chunked":
                self.chunk_events += 1
            else:
                self.window_events += 1
        elif event_type == "stage_window_split":
            self.adaptive_split_count += 1
        elif event_type == "budget_reduced":
            self.budget_reduction_count += 1
        elif event_type == "budget_increased":
            self.budget_increase_count += 1
        elif event_type == "stage_response_success":
            self.responses_processed += 1
            if self.first_success_seconds is None:
                self.first_success_seconds = time.perf_counter() - self.started
        elif event_type == "stage_finished":
            stage_name = str(event.get("stage_name") or "")
            if stage_name:
                self.stage_timings[stage_name] = float(event.get("elapsed_seconds") or 0.0)
                self.stage_statuses[stage_name] = str(event.get("status") or "")


def _fixture_expected_change_ids(expected: Dict[str, object]) -> List[int]:
    explicit = [int(value) for value in expected.get("changed_segment_ids") or []]
    text_ids = [int(segment_id) for segment_id in (expected.get("text_by_id") or {}).keys()]
    speaker_ids = [int(segment_id) for segment_id in (expected.get("speaker_by_id") or {}).keys()]
    ordered: List[int] = []
    seen = set()
    for segment_id in explicit + text_ids + speaker_ids:
        if segment_id not in seen:
            seen.add(segment_id)
            ordered.append(segment_id)
    return ordered


def _compare_fixture_outputs(
    baseline_segments: List[SegmentItem],
    reviewed_segments: List[SegmentItem],
    expected: Dict[str, object],
    preferred_terms: Optional[List[str]] = None,
    expected_unchanged_segment_ids: Optional[List[int]] = None,
    max_allowed_changed_segment_count: Optional[int] = None,
) -> Dict[str, object]:
    baseline_by_id = {int(segment.id): segment for segment in baseline_segments}
    reviewed_by_id = {int(segment.id): segment for segment in reviewed_segments}

    actual_changed_ids = sorted(
        segment_id
        for segment_id, baseline in baseline_by_id.items()
        if segment_id in reviewed_by_id
        and (
            str(reviewed_by_id[segment_id].text) != str(baseline.text)
            or str(reviewed_by_id[segment_id].speaker or "") != str(baseline.speaker or "")
        )
    )
    expected_changed_ids = _fixture_expected_change_ids(expected)

    actual_set = set(actual_changed_ids)
    expected_set = set(expected_changed_ids)
    true_positive = len(actual_set & expected_set)
    false_positive = len(actual_set - expected_set)
    false_negative = len(expected_set - actual_set)
    precision = true_positive / len(actual_set) if actual_set else (1.0 if not expected_set else 0.0)
    recall = true_positive / len(expected_set) if expected_set else (1.0 if not actual_set else 0.0)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)

    expected_texts = {int(key): str(value) for key, value in (expected.get("text_by_id") or {}).items()}
    expected_speakers = {int(key): str(value) for key, value in (expected.get("speaker_by_id") or {}).items()}
    text_matches = sum(
        1
        for segment_id, expected_text in expected_texts.items()
        if segment_id in reviewed_by_id and str(reviewed_by_id[segment_id].text) == expected_text
    )
    speaker_matches = sum(
        1
        for segment_id, expected_speaker in expected_speakers.items()
        if segment_id in reviewed_by_id and str(reviewed_by_id[segment_id].speaker or "") == expected_speaker
    )
    text_accuracy = text_matches / len(expected_texts) if expected_texts else 1.0
    speaker_accuracy = speaker_matches / len(expected_speakers) if expected_speakers else 1.0
    no_change_discipline = 1.0 if false_positive == 0 else max(0.0, 1.0 - (false_positive / max(len(actual_set), 1)))
    protected_unchanged = [int(value) for value in (expected_unchanged_segment_ids or [])]
    unchanged_matches = 0
    for segment_id in protected_unchanged:
        baseline = baseline_by_id.get(segment_id)
        reviewed = reviewed_by_id.get(segment_id)
        if baseline is not None and reviewed is not None:
            if str(baseline.text) == str(reviewed.text) and str(baseline.speaker or "") == str(reviewed.speaker or ""):
                unchanged_matches += 1
    unchanged_discipline = (
        unchanged_matches / len(protected_unchanged)
        if protected_unchanged
        else (1.0 if false_positive == 0 else no_change_discipline)
    )
    protected_terms = [str(term).strip() for term in (preferred_terms or []) if str(term).strip()]
    protected_term_violations = 0
    if protected_terms:
        for segment_id, baseline in baseline_by_id.items():
            reviewed = reviewed_by_id.get(segment_id)
            if reviewed is None:
                continue
            for term in protected_terms:
                if term in str(baseline.text or "") and term not in str(reviewed.text or ""):
                    protected_term_violations += 1
    glossary_safety = 1.0 if protected_term_violations == 0 else 0.0
    returned_changed_segment_count = len(actual_changed_ids)
    expected_changed_segment_count = len(expected_changed_ids)
    if max_allowed_changed_segment_count is not None:
        excess_changed = max(0, returned_changed_segment_count - int(max_allowed_changed_segment_count))
    else:
        excess_changed = max(0, returned_changed_segment_count - max(expected_changed_segment_count, 1))
    patch_compactness = 1.0 if excess_changed == 0 else max(
        0.0,
        1.0 - (excess_changed / max(returned_changed_segment_count, 1)),
    )
    overproduction_ratio = round(
        returned_changed_segment_count / max(expected_changed_segment_count, 1),
        4,
    ) if returned_changed_segment_count else 0.0
    fixture_quality_score = round(
        100.0 * mean([f1, text_accuracy, speaker_accuracy, unchanged_discipline, patch_compactness, glossary_safety]),
        2,
    )

    return {
        "expected_changed_segment_ids": expected_changed_ids,
        "actual_changed_segment_ids": actual_changed_ids,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "text_accuracy": round(text_accuracy, 4),
        "speaker_accuracy": round(speaker_accuracy, 4),
        "over_edit_count": false_positive,
        "unsupported_edit_count": false_positive,
        "missed_edit_count": false_negative,
        "no_change_discipline": round(no_change_discipline, 4),
        "unchanged_discipline": round(unchanged_discipline, 4),
        "patch_compactness": round(patch_compactness, 4),
        "returned_changed_segment_count": returned_changed_segment_count,
        "expected_changed_segment_count": expected_changed_segment_count,
        "overproduction_ratio": overproduction_ratio,
        "protected_term_violation_count": protected_term_violations,
        "glossary_safety": round(glossary_safety, 4),
        "fixture_quality_score": fixture_quality_score,
    }


def _aggregate_fixture_metrics(rows: List[Dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(mean(values), 4) if values else 0.0


def _stage_definition(stage_name: str) -> Dict[str, object]:
    return next(stage for stage in STAGE_DEFINITIONS if str(stage["name"]) == stage_name)


def _repeat_segments_to_budget(segments: List[SegmentItem], target_budget: int) -> List[SegmentItem]:
    if not segments:
        return []
    repeated: List[SegmentItem] = []
    estimate = 0
    cycle = 0
    while estimate < target_budget and cycle < 256:
        for index, segment in enumerate(segments, start=1):
            clone = SegmentItem(**segment.__dict__)
            clone.id = int(segment.id) + cycle * 1000 + index
            repeated.append(clone)
            estimate += _estimated_segment_tokens(segment)
            if estimate >= target_budget:
                break
        cycle += 1
    return repeated


def _probe_stage_budget(
    backend_capabilities: Dict[str, object],
    stage_definition: Dict[str, object],
    sample_segments: List[SegmentItem],
    stage_mode: str,
    target_budget: int,
    preferred_terms: Optional[List[str]] = None,
) -> Dict[str, object]:
    corpus = _repeat_segments_to_budget(sample_segments, target_budget)
    if stage_mode == "full_episode":
        probe_segments = corpus
    else:
        windows = _split_segments_by_token_budget(corpus, target_budget, overlap_segments=0)
        probe_segments = windows[0] if windows else corpus[:1]
    actual_budget = sum(_estimated_segment_tokens(segment) for segment in probe_segments)
    try:
        payload = _execute_stage_backend_request(
            probe_segments,
            backend_capabilities,
            stage_definition,
            stage_mode,
            preferred_terms=preferred_terms,
            debug_context={"audio_path": f"benchmark_capacity_{stage_definition['name']}"},
        )
        if not isinstance(payload.get("reviewed_segments"), list) or payload.get("corrected_segment_count") is None:
            raise RuntimeError("Calibration probe returned incomplete review JSON keys.")
        return {
            "success": True,
            "actual_budget": actual_budget,
            "failure_reason": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "actual_budget": actual_budget,
            "failure_reason": str(exc),
        }


def _capacity_probe_for_mode(
    backend_capabilities: Dict[str, object],
    stage_name: str,
    sample_segments: List[SegmentItem],
    stage_mode: str,
    preferred_terms: Optional[List[str]] = None,
) -> Dict[str, object]:
    session = ReviewCalibrationSession(backend_capabilities)
    family_state = session.families[_review_stage_family(stage_name)]
    floor_budget = int(family_state["floor_budget"])
    hard_ceiling = int(family_state["hard_ceiling"])
    stage_definition = _stage_definition(stage_name)
    low = floor_budget
    high = hard_ceiling
    best_success = None
    first_failure = None
    dominant_failure_reason = ""
    probe_count = 0

    while low <= high and probe_count < 8:
        target = (low + high) // 2
        result = _probe_stage_budget(
            backend_capabilities,
            stage_definition,
            sample_segments,
            stage_mode,
            target,
            preferred_terms=preferred_terms,
        )
        probe_count += 1
        if result["success"]:
            best_success = int(result["actual_budget"])
            low = target + 1
        else:
            first_failure = int(result["actual_budget"])
            dominant_failure_reason = str(result["failure_reason"] or "")
            high = target - 1

    recommended_budget = 0
    if best_success is not None:
        recommended_budget = max(floor_budget, min(hard_ceiling, int(best_success * 0.8)))
    return {
        "stage_mode": stage_mode,
        "probe_count": probe_count,
        "max_successful_input_budget": int(best_success or 0),
        "failure_boundary": int(first_failure or 0),
        "recommended_operating_budget": int(recommended_budget),
        "failure_reason": dominant_failure_reason,
    }


def _capacity_profile(
    backend_capabilities: Dict[str, object],
    fixtures: List[Dict[str, object]],
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for stage_definition in STAGE_DEFINITIONS:
        stage_name = str(stage_definition["name"])
        stage_fixtures = [fixture for fixture in fixtures if str(fixture["focus_stage"]) == stage_name]
        if not stage_fixtures:
            continue
        if stage_name == "episode_qa_review" and not backend_capabilities.get("long_context_available"):
            results[stage_name] = {
                "full_episode": {
                    "stage_mode": "full_episode",
                    "probe_count": 0,
                    "max_successful_input_budget": 0,
                    "failure_boundary": 0,
                    "recommended_operating_budget": 0,
                    "failure_reason": "long_context_unavailable",
                },
                "chunked": {
                    "stage_mode": "chunked",
                    "probe_count": 0,
                    "max_successful_input_budget": 0,
                    "failure_boundary": 0,
                    "recommended_operating_budget": 0,
                    "failure_reason": "long_context_unavailable",
                },
                "full_episode_supported": False,
            }
            continue
        sample_segments = _segment_items_from_cleaned_payload(stage_fixtures[0]["cleaned_payload"])
        preferred_terms = list(stage_fixtures[0].get("preferred_terms") or [])
        if stage_name == "episode_qa_review":
            full_episode = _capacity_probe_for_mode(
                backend_capabilities,
                stage_name,
                sample_segments,
                "full_episode",
                preferred_terms=preferred_terms,
            )
            chunked = _capacity_probe_for_mode(
                backend_capabilities,
                stage_name,
                sample_segments,
                "chunked",
                preferred_terms=preferred_terms,
            )
            results[stage_name] = {
                "full_episode": full_episode,
                "chunked": chunked,
                "full_episode_supported": bool(full_episode["max_successful_input_budget"]),
                "boundary_stability": _classify_boundary_stability(full_episode, chunked),
            }
            continue
        single_mode = _capacity_probe_for_mode(
            backend_capabilities,
            stage_name,
            sample_segments,
            "local_batch",
            preferred_terms=preferred_terms,
        )
        single_mode["boundary_stability"] = _classify_single_mode_boundary_stability(single_mode)
        results[stage_name] = single_mode
    return results


def _classify_single_mode_boundary_stability(capacity: Dict[str, object]) -> str:
    if str(capacity.get("failure_reason") or ""):
        return "edge_sensitive"
    max_success = int(capacity.get("max_successful_input_budget") or 0)
    recommended = int(capacity.get("recommended_operating_budget") or 0)
    if max_success and recommended and recommended >= int(max_success * 0.9):
        return "edge_sensitive"
    if max_success:
        return "stable"
    return "truncation_prone"


def _classify_boundary_stability(full_episode: Dict[str, object], chunked: Dict[str, object]) -> str:
    full_reason = str(full_episode.get("failure_reason") or "")
    chunk_reason = str(chunked.get("failure_reason") or "")
    if chunk_reason:
        return "truncation_prone"
    if full_reason:
        return "edge_sensitive"
    return "stable"


def _model_verdict(report: Dict[str, object]) -> List[str]:
    verdicts: List[str] = []
    quality = report.get("quality") or {}
    speed = report.get("speed") or {}
    usable_capacity = report.get("usable_capacity") or {}
    if float(quality.get("average_glossary_safety") or 0.0) >= 1.0 and int(quality.get("protected_term_violation_count") or 0) == 0:
        verdicts.append("Strong glossary safety: no protected-term regressions were observed.")
    if float(quality.get("average_patch_compactness") or 0.0) >= 0.9:
        verdicts.append("Disciplined patch behavior: fixture edits stayed compact.")
    if float(speed.get("average_elapsed_seconds") or 0.0) > 0:
        verdicts.append(
            f"Average per-fixture latency was {speed.get('average_elapsed_seconds')}s with "
            f"{speed.get('average_time_to_first_success_seconds')}s to first success."
        )
    glossary_capacity = usable_capacity.get("glossary_correction_review") or {}
    if isinstance(glossary_capacity, dict) and int(glossary_capacity.get("recommended_operating_budget") or 0) > 0:
        verdicts.append(
            f"Glossary review recommended operating budget: {glossary_capacity.get('recommended_operating_budget')}."
        )
    if not verdicts:
        verdicts.append("No strong differentiators were detected in this benchmark run.")
    return verdicts


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _derive_stage_usefulness(fixture_results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for stage_name in sorted({str(row["focus_stage"]) for row in fixture_results}):
        stage_rows = [row for row in fixture_results if row["focus_stage"] == stage_name]
        if not stage_rows:
            continue
        result[stage_name] = {
            "fixture_count": len(stage_rows),
            "hit_rate": round(
                sum(1 for row in stage_rows if int(row["quality"]["returned_changed_segment_count"]) > 0)
                / len(stage_rows),
                4,
            ),
            "average_quality_score": round(
                mean(float(row["quality"]["fixture_quality_score"]) for row in stage_rows),
                2,
            ),
            "average_patch_compactness": round(
                mean(float(row["quality"]["patch_compactness"]) for row in stage_rows),
                4,
            ),
        }
    return result


def _derive_scores(
    speed_metrics: Dict[str, object],
    stability_metrics: Dict[str, object],
    quality_metrics: Dict[str, object],
    fixture_results: List[Dict[str, object]],
) -> Dict[str, float]:
    elapsed = float(speed_metrics.get("total_elapsed_seconds") or 0.0)
    useful_corrections = sum(int(row["quality"]["expected_changed_segment_count"]) for row in fixture_results)
    quality_per_second = round(
        _safe_divide(float(quality_metrics.get("average_fixture_quality_score") or 0.0), max(float(speed_metrics.get("average_elapsed_seconds") or 0.0), 0.001)),
        4,
    )
    useful_corrections_per_second = round(_safe_divide(useful_corrections, elapsed), 4)
    over_edit_penalty = round(
        _safe_divide(float(quality_metrics.get("over_edit_count") or 0.0), max(len(fixture_results), 1)),
        4,
    )
    preferred_term_safety_rate = round(float(quality_metrics.get("average_glossary_safety") or 0.0), 4)
    stability_score = round(
        max(
            0.0,
            100.0
            - float(stability_metrics.get("adaptive_split_count") or 0) * 2.0
            - float(stability_metrics.get("hard_failure_count") or 0) * 10.0
            - float(stability_metrics.get("budget_reduction_count") or 0) * 1.5,
        ),
        2,
    )
    return {
        "quality_per_second": quality_per_second,
        "useful_corrections_per_second": useful_corrections_per_second,
        "over_edit_penalty": over_edit_penalty,
        "preferred_term_safety_rate": preferred_term_safety_rate,
        "stability_score": stability_score,
    }


def _production_recommendations(
    quality_metrics: Dict[str, object],
    derived_scores: Dict[str, float],
    usable_capacity: Dict[str, object],
) -> Dict[str, object]:
    cleanup_capacity = usable_capacity.get("transcript_cleanup_review") or {}
    speaker_capacity = usable_capacity.get("speaker_consistency_review") or {}
    episode_qa_capacity = usable_capacity.get("episode_qa_review") or {}
    full_episode_supported = bool(episode_qa_capacity.get("full_episode_supported"))
    return {
        "recommended_for_fast_default": bool(
            derived_scores.get("quality_per_second", 0.0) >= 20.0
            and float(quality_metrics.get("average_fixture_quality_score") or 0.0) >= 80.0
        ),
        "recommended_for_quality_pass": bool(
            float(quality_metrics.get("average_fixture_quality_score") or 0.0) >= 85.0
            and float(quality_metrics.get("average_glossary_safety") or 0.0) >= 1.0
        ),
        "recommended_for_speaker_consistency": bool(
            float((quality_metrics.get("focus_stage_scores") or {}).get("speaker_consistency_review") or 0.0) >= 80.0
            and int(speaker_capacity.get("recommended_operating_budget") or 0) >= 1000
        ),
        "recommended_for_long_context_qa": bool(
            full_episode_supported
            or int(((episode_qa_capacity.get("chunked") or {}).get("recommended_operating_budget") or 0)) >= 4000
        ),
        "recommended_cleanup_budget": int(cleanup_capacity.get("recommended_operating_budget") or 0),
    }


def run_review_benchmark(
    runtime_review_config: Dict[str, object],
    output_dir: Path,
    fixture_dir: Optional[Path] = None,
) -> Dict[str, object]:
    fixtures = load_benchmark_fixtures(fixture_dir)
    backend_capabilities = enrich_backend_capabilities_with_identity(resolve_backend_capabilities(runtime_review_config))
    calibration_session = ReviewCalibrationSession(backend_capabilities)
    run_started = time.perf_counter()
    fixture_results: List[Dict[str, object]] = []

    for fixture in fixtures:
        telemetry = BenchmarkTelemetry()
        cleaned_segments = _segment_items_from_cleaned_payload(fixture["cleaned_payload"])
        fixture_runtime_config = dict(runtime_review_config)
        if fixture.get("preferred_terms"):
            fixture_runtime_config["preferred_terms"] = list(fixture.get("preferred_terms") or [])
        fixture_started = time.perf_counter()
        review_result = review_segments(
            cleaned_segments,
            fixture_runtime_config,
            review_input_source="benchmark_fixture",
            calibration_session=calibration_session,
            progress_callback=telemetry.callback,
            debug_context={
                "audio_path": fixture["name"],
                "output_dir": str(output_dir),
                "review_input_source": "benchmark_fixture",
            },
        )
        elapsed_seconds = time.perf_counter() - fixture_started
        reviewed_segments = review_result.get("segments") or cleaned_segments
        quality = _compare_fixture_outputs(
            cleaned_segments,
            reviewed_segments,
            fixture["expected"],
            preferred_terms=list(fixture.get("preferred_terms") or []),
            expected_unchanged_segment_ids=list(fixture.get("expected_unchanged_segment_ids") or []),
            max_allowed_changed_segment_count=fixture.get("max_allowed_changed_segment_count"),
        )
        changed_segments_payload = [
            {
                "id": int(segment.id),
                "text": str(segment.text),
                "speaker": str(segment.speaker or ""),
            }
            for segment in reviewed_segments
            if any(
                (
                    int(segment.id) == int(baseline.id)
                    and (
                        str(segment.text) != str(baseline.text)
                        or str(segment.speaker or "") != str(baseline.speaker or "")
                    )
                )
                for baseline in cleaned_segments
            )
        ]
        response_shape = {
            "response_characters": len(json.dumps(changed_segments_payload, ensure_ascii=False)),
            "returned_changed_segment_count": int(quality["returned_changed_segment_count"]),
            "expected_changed_segment_count": int(quality["expected_changed_segment_count"]),
            "overproduction_ratio": float(quality["overproduction_ratio"]),
            "bytes_per_accepted_correction": round(
                len(json.dumps(changed_segments_payload, ensure_ascii=False).encode("utf-8"))
                / max(int(quality["expected_changed_segment_count"]) or 1, 1),
                2,
            ),
        }
        fixture_results.append(
            {
                "name": fixture["name"],
                "focus_stage": fixture["focus_stage"],
                "description": fixture["description"],
                "benchmark_tags": list(fixture.get("benchmark_tags") or []),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "time_to_first_success_seconds": round(telemetry.first_success_seconds or 0.0, 3),
                "responses_processed": telemetry.responses_processed,
                "window_events": telemetry.window_events,
                "chunk_events": telemetry.chunk_events,
                "adaptive_split_count": telemetry.adaptive_split_count,
                "budget_reduction_count": telemetry.budget_reduction_count,
                "budget_increase_count": telemetry.budget_increase_count,
                "stage_timings": {key: round(value, 4) for key, value in telemetry.stage_timings.items()},
                "stage_statuses": telemetry.stage_statuses,
                "review_result": {
                    "attempted": bool(review_result.get("attempted")),
                    "skipped": bool(review_result.get("skipped")),
                    "skip_reason": str(review_result.get("skip_reason") or ""),
                    "review_status": str((review_result.get("metadata") or {}).get("review_status") or ""),
                    "review_completed_stages": list((review_result.get("metadata") or {}).get("review_completed_stages") or []),
                    "review_skipped_stages": list((review_result.get("metadata") or {}).get("review_skipped_stages") or []),
                },
                "quality": quality,
                "response_shape": response_shape,
                "review_calibration": (review_result.get("metadata") or {}).get("review_calibration") or {},
            }
        )

    speed_metrics = {
        "fixture_count": len(fixture_results),
        "total_elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "average_elapsed_seconds": round(mean(float(row["elapsed_seconds"]) for row in fixture_results), 3),
        "average_time_to_first_success_seconds": round(
            mean(float(row["time_to_first_success_seconds"]) for row in fixture_results),
            3,
        ),
        "responses_processed": sum(int(row["responses_processed"]) for row in fixture_results),
        "windows_processed": sum(int(row["window_events"]) for row in fixture_results),
        "chunks_processed": sum(int(row["chunk_events"]) for row in fixture_results),
        "stage_elapsed_seconds": {
            stage_name: round(
                sum(float(row["stage_timings"].get(stage_name) or 0.0) for row in fixture_results),
                4,
            )
            for stage_name in {
                stage_name
                for row in fixture_results
                for stage_name in row["stage_timings"].keys()
            }
        },
    }
    stability_metrics = {
        "adaptive_split_count": sum(int(row["adaptive_split_count"]) for row in fixture_results),
        "budget_reduction_count": sum(int(row["budget_reduction_count"]) for row in fixture_results),
        "budget_increase_count": sum(int(row["budget_increase_count"]) for row in fixture_results),
        "hard_failure_count": sum(
            1
            for row in fixture_results
            if row["review_result"]["skip_reason"] not in {"", "review_disabled"}
        ),
        "fixture_completion_without_split_rate": round(
            sum(1 for row in fixture_results if int(row["adaptive_split_count"]) == 0) / max(len(fixture_results), 1),
            4,
        ),
        "stage_completion_rate": round(
            sum(len(row["review_result"]["review_completed_stages"]) for row in fixture_results)
            / max(
                sum(
                    len(row["review_result"]["review_completed_stages"]) + len(row["review_result"]["review_skipped_stages"])
                    for row in fixture_results
                ),
                1,
            ),
            4,
        ),
        "final_budgets": calibration_session.metadata_snapshot().get("families", {}),
    }
    quality_metrics = {
        "average_fixture_quality_score": round(mean(float(row["quality"]["fixture_quality_score"]) for row in fixture_results), 2),
        "average_precision": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "precision"),
        "average_recall": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "recall"),
        "average_f1": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "f1"),
        "average_text_accuracy": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "text_accuracy"),
        "average_speaker_accuracy": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "speaker_accuracy"),
        "average_glossary_safety": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "glossary_safety"),
        "average_no_change_discipline": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "no_change_discipline"),
        "average_unchanged_discipline": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "unchanged_discipline"),
        "average_patch_compactness": _aggregate_fixture_metrics([row["quality"] for row in fixture_results], "patch_compactness"),
        "over_edit_count": sum(int(row["quality"]["over_edit_count"]) for row in fixture_results),
        "protected_term_violation_count": sum(int(row["quality"]["protected_term_violation_count"]) for row in fixture_results),
        "average_overproduction_ratio": round(
            mean(float(row["response_shape"]["overproduction_ratio"]) for row in fixture_results),
            4,
        ),
        "focus_stage_scores": {
            focus_stage: round(
                mean(float(row["quality"]["fixture_quality_score"]) for row in fixture_results if row["focus_stage"] == focus_stage),
                2,
            )
            for focus_stage in sorted({str(row["focus_stage"]) for row in fixture_results})
        },
    }
    usable_capacity = _capacity_profile(backend_capabilities, fixtures)
    stage_usefulness = _derive_stage_usefulness(fixture_results)
    derived_scores = _derive_scores(speed_metrics, stability_metrics, quality_metrics, fixture_results)
    production_recommendations = _production_recommendations(quality_metrics, derived_scores, usable_capacity)
    return {
        "benchmark_metadata": {
            "fixture_dir": str(fixture_dir or default_fixture_dir()),
            "fixture_count": len(fixtures),
            "generated_at_epoch_ms": int(time.time() * 1000),
        },
        "backend_identity": {
            "runtime_profile": str(backend_capabilities.get("runtime_profile") or ""),
            "backend_name": str(backend_capabilities.get("backend_name") or ""),
            "review_model_name": str(backend_capabilities.get("review_model_name") or ""),
            "review_base_url": str(backend_capabilities.get("review_base_url") or ""),
            "backend_identity_source": str(backend_capabilities.get("backend_identity_source") or ""),
            "backend_identity_model_id": str(backend_capabilities.get("backend_identity_model_id") or ""),
            "backend_identity_digest": str(backend_capabilities.get("backend_identity_digest") or ""),
            "backend_identity_model_ids": list(backend_capabilities.get("backend_identity_model_ids") or []),
        },
        "review_calibration": calibration_session.metadata_snapshot(),
        "speed": speed_metrics,
        "stability": stability_metrics,
        "quality": quality_metrics,
        "usable_capacity": usable_capacity,
        "stage_usefulness": stage_usefulness,
        "derived_scores": derived_scores,
        "production_recommendations": production_recommendations,
        "fixtures": fixture_results,
        "model_verdict": _model_verdict(
            {
                "quality": quality_metrics,
                "speed": speed_metrics,
                "usable_capacity": usable_capacity,
            }
        ),
    }


def write_review_benchmark_reports(output_dir: Path, report: Dict[str, object]) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "review_benchmark_report.json"
    md_path = output_dir / "review_benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Review Benchmark Report",
        "",
        "## Run Identity",
        f"- runtime_profile: {report['backend_identity']['runtime_profile']}",
        f"- backend: {report['backend_identity']['backend_name']}",
        f"- review_model_name: {report['backend_identity']['review_model_name']}",
        f"- backend_identity_source: {report['backend_identity']['backend_identity_source']}",
        f"- backend_identity_model_id: {report['backend_identity']['backend_identity_model_id']}",
        "",
        "## Speed",
        f"- fixture_count: {report['speed']['fixture_count']}",
        f"- total_elapsed_seconds: {report['speed']['total_elapsed_seconds']}",
        f"- average_elapsed_seconds: {report['speed']['average_elapsed_seconds']}",
        f"- average_time_to_first_success_seconds: {report['speed']['average_time_to_first_success_seconds']}",
        "",
        "## Stability",
        f"- adaptive_split_count: {report['stability']['adaptive_split_count']}",
        f"- budget_reduction_count: {report['stability']['budget_reduction_count']}",
        f"- budget_increase_count: {report['stability']['budget_increase_count']}",
        f"- hard_failure_count: {report['stability']['hard_failure_count']}",
        f"- stage_completion_rate: {report['stability']['stage_completion_rate']}",
        "",
        "## Quality",
        f"- average_fixture_quality_score: {report['quality']['average_fixture_quality_score']}",
        f"- average_precision: {report['quality']['average_precision']}",
        f"- average_recall: {report['quality']['average_recall']}",
        f"- average_f1: {report['quality']['average_f1']}",
        f"- average_glossary_safety: {report['quality']['average_glossary_safety']}",
        f"- average_no_change_discipline: {report['quality']['average_no_change_discipline']}",
        f"- average_patch_compactness: {report['quality']['average_patch_compactness']}",
        f"- average_overproduction_ratio: {report['quality']['average_overproduction_ratio']}",
        f"- protected_term_violation_count: {report['quality']['protected_term_violation_count']}",
        "",
        "## Derived Scores",
        f"- quality_per_second: {report['derived_scores']['quality_per_second']}",
        f"- useful_corrections_per_second: {report['derived_scores']['useful_corrections_per_second']}",
        f"- over_edit_penalty: {report['derived_scores']['over_edit_penalty']}",
        f"- preferred_term_safety_rate: {report['derived_scores']['preferred_term_safety_rate']}",
        f"- stability_score: {report['derived_scores']['stability_score']}",
        "",
        "## Production Recommendations",
    ]
    for key, value in sorted((report.get("production_recommendations") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Verdict",
    ])
    for line in report.get("model_verdict") or []:
        lines.append(f"- {line}")
    lines.extend([
        "",
        "## Focus Stage Scores",
    ])
    for stage_name, score in sorted((report["quality"].get("focus_stage_scores") or {}).items()):
        lines.append(f"- {stage_name}: {score}")
    lines.extend(["", "## Stage Usefulness"])
    for stage_name, item in sorted((report.get("stage_usefulness") or {}).items()):
        lines.append(
            f"- {stage_name}: hit_rate={item.get('hit_rate')}, average_quality_score={item.get('average_quality_score')}, "
            f"average_patch_compactness={item.get('average_patch_compactness')}"
        )
    lines.extend(["", "## Usable Capacity"])
    for stage_name, capacity in sorted((report.get("usable_capacity") or {}).items()):
        if isinstance(capacity, dict) and "full_episode" in capacity:
            lines.extend(
                [
                    f"### {stage_name}",
                    f"- full_episode_supported: {capacity.get('full_episode_supported')}",
                    f"- boundary_stability: {capacity.get('boundary_stability')}",
                    f"- full_episode_recommended_budget: {capacity['full_episode'].get('recommended_operating_budget')}",
                    f"- chunked_recommended_budget: {capacity['chunked'].get('recommended_operating_budget')}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"### {stage_name}",
                    f"- boundary_stability: {capacity.get('boundary_stability')}",
                    f"- recommended_operating_budget: {capacity.get('recommended_operating_budget')}",
                    f"- max_successful_input_budget: {capacity.get('max_successful_input_budget')}",
                    f"- failure_reason: {capacity.get('failure_reason')}",
                    "",
                ]
            )
    lines.extend(["", "## Fixture Results"])
    for fixture in report.get("fixtures") or []:
        lines.extend(
            [
                f"### {fixture['name']}",
                f"- focus_stage: {fixture['focus_stage']}",
                f"- elapsed_seconds: {fixture['elapsed_seconds']}",
                f"- adaptive_split_count: {fixture['adaptive_split_count']}",
                f"- review_status: {fixture['review_result']['review_status']}",
                f"- fixture_quality_score: {fixture['quality']['fixture_quality_score']}",
                f"- precision: {fixture['quality']['precision']}",
                f"- recall: {fixture['quality']['recall']}",
                f"- patch_compactness: {fixture['quality']['patch_compactness']}",
                f"- no_change_discipline: {fixture['quality']['no_change_discipline']}",
                f"- response_characters: {fixture['response_shape']['response_characters']}",
                f"- overproduction_ratio: {fixture['response_shape']['overproduction_ratio']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return json_path, md_path
