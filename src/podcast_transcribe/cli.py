import argparse
import csv
import gc
import hashlib
import ctypes
from ctypes import wintypes
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
import wave
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_FFMPEG_DLL_DIRECTORY_HANDLE = None
_FFMPEG_DLL_HANDLES = []


def configure_ffmpeg_dll_directory():
    global _FFMPEG_DLL_DIRECTORY_HANDLE
    ffmpeg_bin_dir = os.getenv("PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR") or os.getenv("FFMPEG_BIN_DIR")
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if not ffmpeg_bin_dir and Path(r"C:\ffmpeg7\bin").is_dir():
        ffmpeg_bin_dir = r"C:\ffmpeg7\bin"

    if ffmpeg_bin_dir and os.path.isdir(ffmpeg_bin_dir):
        if list(Path(ffmpeg_bin_dir).glob("avcodec-62.dll")):
            raise RuntimeError(
                "The configured FFmpeg directory contains FFmpeg 8 shared libraries, "
                "which TorchCodec 0.8.1 does not support on Windows. Configure a shared "
                "FFmpeg 4-7 build, such as C:\\ffmpeg7\\bin."
            )
        _FFMPEG_DLL_DIRECTORY_HANDLE = os.add_dll_directory(ffmpeg_bin_dir)
        # TorchCodec probes several FFmpeg ABI variants. Preloading the configured
        # shared build prevents an incompatible FFmpeg elsewhere on PATH from
        # triggering a modal Windows loader error before the probe can recover.
        for pattern in (
            "avutil-*.dll",
            "swresample-*.dll",
            "swscale-*.dll",
            "avcodec-*.dll",
            "avformat-*.dll",
            "avfilter-*.dll",
            "avdevice-*.dll",
        ):
            for dll_path in sorted(Path(ffmpeg_bin_dir).glob(pattern)):
                _FFMPEG_DLL_HANDLES.append(ctypes.WinDLL(str(dll_path)))


configure_ffmpeg_dll_directory()

warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    module=r"pyannote\.audio\.core\.io",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Requested Pretrainer collection using symlinks on Windows.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
)

import huggingface_hub
import numpy as np
import scipy
import torch
import torchaudio
from faster_whisper import WhisperModel


def _patch_huggingface_hub_auth_compat():
    signature = inspect.signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" in signature.parameters:
        return

    original_hf_hub_download = huggingface_hub.hf_hub_download

    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return original_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = compat_hf_hub_download

    try:
        import huggingface_hub.file_download as file_download

        file_download.hf_hub_download = compat_hf_hub_download
    except Exception:
        pass


_patch_huggingface_hub_auth_compat()

import pyannote.audio as pyannote_audio
from pyannote.audio import Pipeline
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from podcast_transcribe.cleanup import build_cleaned_segments
from podcast_transcribe.config import (
    DEFAULT_REVIEW_BATCH_TOKEN_LIMIT,
    DEFAULT_REVIEW_BACKEND,
    DEFAULT_RUNTIME_PROFILE,
    REVIEW_BACKENDS,
    RUNTIME_PROFILES,
    load_replacement_map as config_load_replacement_map,
    resolve_review_runtime_config,
    WORKFLOW_PROFILES,
)
from podcast_transcribe.contract import (
    validate_reviewed_transcript_payload,
    validate_transcript_payload,
)
from podcast_transcribe.contract_v2 import (
    EPISODE_CONTRACT_V2,
    archive_legacy_episode_bundle,
    episode_contract_status,
    load_correction_lineage,
    stable_episode_uid,
    upgrade_episode_bundle_v2,
)
from podcast_transcribe.outputs import (
    build_episode_metadata,
    write_batch_report_md as output_write_batch_report_md,
    write_json_output as output_write_json_output,
    write_output_manifest as output_write_output_manifest,
    write_review_run_report as output_write_review_run_report,
    write_review_csv as output_write_review_csv,
    write_speaker_workflow_report as output_write_speaker_workflow_report,
    write_speaker_identity_review_csv as output_write_speaker_identity_review_csv,
    write_text_transcript as output_write_text_transcript,
)
from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.evaluation import run_pipeline_benchmark, write_pipeline_benchmark_reports
from podcast_transcribe.orchestration.fingerprints import build_stage_fingerprint
from podcast_transcribe.providers.alignment import ALIGNMENT_PROVIDERS, create_alignment_provider
from podcast_transcribe.providers.asr import FasterWhisperASRProvider, ParakeetASRProvider
from podcast_transcribe.providers.contracts import ProviderIdentity
from podcast_transcribe.providers.diarization import pyannote_provider_identity
from podcast_transcribe.providers.governance import (
    acquire_provider_artifact,
    artifact_directory,
    build_speech_provider_run,
    provider_preflight,
    resolve_execution_profile,
    write_immutable_speech_run,
)
from podcast_transcribe.providers.speaker_embedding import SpeechBrainECAPAProvider, SpeechBrainXVectorProvider
from podcast_transcribe.quality import language_model_warnings
from podcast_transcribe.review import (
    ReviewCalibrationSession,
    enrich_backend_capabilities_with_identity,
    resolve_backend_capabilities,
    review_debug_directory,
    review_segments,
)
from podcast_transcribe.review_benchmark import run_review_benchmark, write_review_benchmark_reports
from podcast_transcribe.state import (
    ARTIFACT_DIRNAME,
    CHECKPOINT_DIRNAME,
    DIARIZATION_HISTORY_FILENAME,
    RESUME_STATE_FILENAME,
    REVIEW_CALIBRATION_FILENAME,
    SUMMARY_FILENAME,
    audio_file_fingerprint,
    atomic_write_text as state_atomic_write_text,
    clear_stage_artifacts as state_clear_stage_artifacts,
    clear_debug_artifacts as state_clear_debug_artifacts,
    expected_output_paths as state_expected_output_paths,
    is_file_already_processed as state_is_file_already_processed,
    load_diarization_history_state as state_load_diarization_history_state,
    load_review_calibration_state as state_load_review_calibration_state,
    load_stage_artifact as state_load_stage_artifact,
    load_episode_summary_rows as state_load_episode_summary_rows,
    load_processed_files as state_load_processed_files,
    save_diarization_history_state as state_save_diarization_history_state,
    save_review_calibration_state as state_save_review_calibration_state,
    save_stage_artifact as state_save_stage_artifact,
    save_processed_files as state_save_processed_files,
)
from podcast_transcribe.speakers import (
    average_embeddings as speaker_average_embeddings,
    cosine_similarity as speaker_cosine_similarity,
    detect_speaker_similarity_drift,
    final_host_profile_update,
    merge_profile as speaker_merge_profile,
    reference_sample_quality,
)
from podcast_transcribe.speaker_workflow import build_cross_episode_speaker_view


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
LONG_FILE_WARNING_HOURS = 4.0
SPEAKER_AUDIO_CACHE_VERSION = 1
SPEAKER_AUDIO_CACHE_SAMPLE_RATE = 16000
SPEAKER_AUDIO_CACHE_SOURCE_SUFFIXES = {
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".webm",
}
DIARIZATION_CHUNK_MINUTES = 25.0
DIARIZATION_CHUNK_OVERLAP_SECONDS = 90.0
DIARIZATION_PROBE_MARGIN_SECONDS = 30.0 * 60.0
DIARIZATION_PROBE_NEARBY_FAILURE_SECONDS = 15.0 * 60.0
DIARIZATION_PROBE_RECENT_WINDOW = 5
DIARIZATION_PROBE_COOLDOWN_EPISODES = 3
DIARIZATION_PROBE_COOLDOWN_SECONDS = 24.0 * 60.0 * 60.0
DIARIZATION_RECONCILIATION_SIMILARITY = 0.72
DIARIZATION_MIN_EMBEDDING_SECONDS = 0.75
SPEAKER_PROFILE_SCHEMA_VERSION = 2


ANONYMOUS_SPEAKER_IDENTITY = ProviderIdentity(
    stage="speaker_attribution",
    provider="anonymous_meeting",
    model="diarization_labels_only",
    model_revision="builtin-v1",
    acquisition="bundled",
    capabilities={"speaker_separation": True, "speaker_identity": False},
    confidence_semantics="anonymous diarization labels only",
    license="project license",
)


class AnonymousSpeakerEmbeddingProvider:
    """Identity-only adapter used when a meeting needs separation, not identity."""

    @property
    def identity(self) -> ProviderIdentity:
        return ANONYMOUS_SPEAKER_IDENTITY

    def encode(self, waveform):
        raise RuntimeError("Anonymous meeting profile does not compute speaker embeddings.")


class ProgressHook:
    """Adapter that renders faster-whisper and pyannote progress with Rich bars."""

    def __init__(self, transient: bool = False, hidden: bool = False):
        self.transient = transient
        self.hidden = hidden
        self._current_task_name = None
        self._current_task_id = None
        self._current_task_is_indeterminate = False

    def __enter__(self):
        if self.hidden:
            return self

        self.progress = create_stage_progress(transient=self.transient)
        self.progress.start()
        return self

    def __exit__(self, *args):
        if self.hidden:
            return

        self._finish_current_task()
        self.progress.stop()
        return

    def _finish_current_task(self):
        if self._current_task_id is None:
            return

        if self._current_task_is_indeterminate:
            self.progress.update(self._current_task_id, total=1, completed=1)
        self.progress.refresh()

    def __call__(
        self,
        step_name,
        step_artifact,
        file: Optional[Dict[str, object]] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        if self.hidden:
            return

        is_indeterminate = total is None and completed is None

        if self._current_task_name != step_name:
            self._finish_current_task()
            self._current_task_name = step_name
            self._current_task_is_indeterminate = is_indeterminate
            if is_indeterminate:
                self._current_task_id = self.progress.add_task(step_name, total=None)
            else:
                if completed is None:
                    completed = 0
                if total is None:
                    total = max(completed, 1)
                self._current_task_id = self.progress.add_task(step_name, total=total, completed=completed)
            return

        if is_indeterminate:
            self.progress.refresh()
            return

        if completed is None:
            completed = 0
        if total is None:
            total = max(completed, 1)

        self._current_task_is_indeterminate = False
        self.progress.update(self._current_task_id, completed=completed, total=total)

        if completed >= total:
            self.progress.refresh()


def progress_spinner_name(output_stream=None) -> str:
    stream = output_stream if output_stream is not None else sys.stdout
    encoding = str(getattr(stream, "encoding", "") or "").lower()
    return "dots" if "utf" in encoding else "line"


def create_stage_progress(transient: bool = False) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(progress_spinner_name()),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
        TimeElapsedColumn(),
        transient=transient,
    )


def print_episode_mode(mode: str):
    print(f"Episode mode: {mode}")


def print_episode_stage(current: int, total: int, label: str):
    print(f"Episode stage {current}/{total}: {label}")


TIMING_COMPONENT_REPORT_THRESHOLD_SECONDS = 60.0


@contextmanager
def timed_component(component_timings: Dict[str, float], component_name: str):
    """Accumulate one logical operation, including any loops inside it."""

    started = time.perf_counter()
    try:
        yield
    finally:
        component_timings[component_name] = component_timings.get(component_name, 0.0) + (
            time.perf_counter() - started
        )


def print_timing_component_summary(stage_label: str, component_timings: Dict[str, float]):
    slow_components = [
        (name, elapsed)
        for name, elapsed in component_timings.items()
        if float(elapsed) >= TIMING_COMPONENT_REPORT_THRESHOLD_SECONDS
    ]
    if not slow_components:
        return
    print(f"  {stage_label} time components (>=60s):")
    for name, elapsed in sorted(slow_components, key=lambda item: item[1], reverse=True):
        print(f"    {name}: {elapsed:.1f}s")


def start_finalization_operation(label: str) -> float:
    return start_console_operation("finalization", label)


def start_console_operation(scope: str, label: str) -> float:
    print(f"  {scope}: {label}...")
    return time.perf_counter()


def finish_finalization_operation(label: str, started: float):
    finish_console_operation("finalization", label, started)


def finish_console_operation(scope: str, label: str, started: float):
    print(f"  {scope}: {label} complete in {time.perf_counter() - started:.1f}s")


def make_review_progress_callback(component_timings: Optional[Dict[str, float]] = None):
    state = {"current_stage": None, "stage_index": None, "stage_total": None}

    def callback(event: Dict[str, object]):
        event_type = str(event.get("event") or "")
        stage_label = str(event.get("stage_label") or event.get("stage_name") or "review")
        pretty_label = stage_label.replace("_", " ")
        if event_type == "stage_index":
            state["current_stage"] = str(event.get("stage_name") or "")
            state["stage_index"] = int(event.get("current") or 0)
            state["stage_total"] = int(event.get("total") or 0)
            print(f"  Review stage {state['stage_index']}/{state['stage_total']}: {pretty_label}")
        elif event_type == "stage_window_progress":
            mode = str(event.get("mode") or "")
            current = int(event.get("current") or 0)
            total = int(event.get("total") or 0)
            if mode == "chunked":
                print(f"    {pretty_label} chunk {current}/{total}")
            elif total > 1:
                print(f"    {pretty_label} window {current}/{total}")
        elif event_type == "stage_skipped":
            reason = str(event.get("reason") or "skipped")
            print(f"  {pretty_label} skipped: {reason}")
        elif event_type == "calibration_complete":
            if component_timings is not None:
                component_timings["review calibration"] = float(event.get("elapsed_seconds") or 0.0)
            print(f"  {str(event.get('summary') or 'Review calibration complete.')}")
        elif event_type == "stage_finished":
            if component_timings is not None:
                component_timings[f"review {pretty_label}"] = float(event.get("elapsed_seconds") or 0.0)
        elif event_type == "budget_reduced":
            family_name = str(event.get("family_name") or "review")
            old_budget = int(event.get("old_budget") or 0)
            new_budget = int(event.get("new_budget") or 0)
            family_label = family_name.replace("_review", "").replace("_", " ")
            print(f"  Review budget reduced for {family_label}: {old_budget} -> {new_budget}")
        elif event_type == "budget_increased":
            family_name = str(event.get("family_name") or "review")
            old_budget = int(event.get("old_budget") or 0)
            new_budget = int(event.get("new_budget") or 0)
            family_label = family_name.replace("_review", "").replace("_", " ")
            print(f"  Review budget increased for {family_label}: {old_budget} -> {new_budget}")

    return callback


def make_checkpointed_review_progress_callback(
    output_dir: Path,
    audio_path: Path,
    component_timings: Optional[Dict[str, float]] = None,
):
    console_callback = make_review_progress_callback(component_timings)

    def callback(event: Dict[str, object]):
        console_callback(event)
        if str(event.get("event") or "") in {"stage_started", "stage_response_success", "stage_finished", "stage_skipped"}:
            write_processing_checkpoint(
                output_dir,
                audio_path,
                "review_in_progress",
                {
                    "event": str(event.get("event") or ""),
                    "stage_name": str(event.get("stage_name") or ""),
                    "current": event.get("current"),
                    "total": event.get("total"),
                },
            )

    return callback


def review_calibration_state_path(output_dir: Path) -> Path:
    return output_dir / REVIEW_CALIBRATION_FILENAME


def diarization_history_state_path(output_dir: Path) -> Path:
    return output_dir / DIARIZATION_HISTORY_FILENAME


def current_time_epoch_seconds() -> float:
    return float(time.time())


def diarization_runtime_fingerprint(
    diarization_model_id: str,
    input_mode: str,
) -> Dict[str, object]:
    return {
        "diarization_model_id": str(diarization_model_id or ""),
        "pyannote_version": str(getattr(pyannote_audio, "__version__", "") or ""),
        "scipy_version": str(getattr(scipy, "__version__", "") or ""),
        "torch_version": str(getattr(torch, "__version__", "") or ""),
        "input_mode": str(input_mode or ""),
    }


def _normalize_diarization_history_record(record: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(record)
    normalized["duration_seconds"] = float(record.get("duration_seconds") or 0.0)
    normalized["timestamp_epoch_seconds"] = float(record.get("timestamp_epoch_seconds") or 0.0)
    normalized["probe"] = bool(record.get("probe"))
    normalized["invalidated"] = bool(record.get("invalidated"))
    normalized["mode"] = str(record.get("mode") or "")
    normalized["outcome"] = str(record.get("outcome") or "")
    return normalized


def load_diarization_history(output_dir: Path) -> Dict[str, object]:
    payload = state_load_diarization_history_state(diarization_history_state_path(output_dir))
    if not isinstance(payload, dict):
        return {"records": []}
    records = payload.get("records") or []
    return {
        "records": [
            _normalize_diarization_history_record(record)
            for record in records
            if isinstance(record, dict)
        ]
    }


def save_diarization_history(output_dir: Path, payload: Dict[str, object]):
    state_save_diarization_history_state(diarization_history_state_path(output_dir), payload)


def diarization_history_records_for_fingerprint(
    history_state: Dict[str, object],
    fingerprint: Dict[str, object],
) -> List[Dict[str, object]]:
    return [
        record
        for record in (history_state.get("records") or [])
        if isinstance(record, dict) and record.get("runtime_fingerprint") == fingerprint
    ]


def diarization_learning_state(
    history_state: Dict[str, object],
    fingerprint: Dict[str, object],
) -> Dict[str, object]:
    records = diarization_history_records_for_fingerprint(history_state, fingerprint)
    successes = [
        record for record in records
        if record.get("mode") == "global" and record.get("outcome") == "success"
    ]
    failures = [
        record for record in records
        if record.get("mode") == "global" and record.get("outcome") == "memory_error" and not record.get("invalidated")
    ]
    safe_success_ceiling = max((float(record.get("duration_seconds") or 0.0) for record in successes), default=0.0)
    failure_floor = min((float(record.get("duration_seconds") or 0.0) for record in failures), default=0.0)
    recent_probes = [
        record for record in records
        if record.get("probe")
    ]
    recent_probes = sorted(recent_probes, key=lambda item: float(item.get("timestamp_epoch_seconds") or 0.0), reverse=True)
    return {
        "records": records,
        "safe_success_ceiling": safe_success_ceiling,
        "failure_floor": failure_floor,
        "recent_probes": recent_probes,
    }


def should_probe_diarization_duration(
    duration_seconds: float,
    learning_state: Dict[str, object],
    now_epoch_seconds: Optional[float] = None,
) -> Tuple[bool, str]:
    failure_floor = float(learning_state.get("failure_floor") or 0.0)
    if failure_floor <= 0:
        return False, "no_failure_floor"
    if duration_seconds <= failure_floor:
        return False, "below_failure_floor"
    if duration_seconds > failure_floor + DIARIZATION_PROBE_MARGIN_SECONDS:
        return False, "outside_probe_band"

    recent_probes = learning_state.get("recent_probes") or []
    nearby_failed_probe = any(
        record.get("outcome") == "memory_error"
        and abs(float(record.get("duration_seconds") or 0.0) - duration_seconds) <= DIARIZATION_PROBE_NEARBY_FAILURE_SECONDS
        for record in recent_probes[:DIARIZATION_PROBE_RECENT_WINDOW]
    )
    if nearby_failed_probe:
        return False, "recent_nearby_probe_failed"

    if recent_probes:
        latest_probe = recent_probes[0]
        completed_episodes_since_probe = sum(
            1
            for record in learning_state.get("records") or []
            if float(record.get("timestamp_epoch_seconds") or 0.0) > float(latest_probe.get("timestamp_epoch_seconds") or 0.0)
        )
        now_value = now_epoch_seconds if now_epoch_seconds is not None else current_time_epoch_seconds()
        seconds_since_probe = now_value - float(latest_probe.get("timestamp_epoch_seconds") or 0.0)
        if (
            completed_episodes_since_probe < DIARIZATION_PROBE_COOLDOWN_EPISODES
            and seconds_since_probe < DIARIZATION_PROBE_COOLDOWN_SECONDS
        ):
            return False, "probe_cooldown_active"

    return True, "probe_band"


def diarization_route_decision(
    duration_seconds: Optional[float],
    history_state: Dict[str, object],
    fingerprint: Dict[str, object],
) -> Dict[str, object]:
    if duration_seconds is None or duration_seconds <= 0:
        return {
            "mode": "global",
            "probe": False,
            "learned_route": False,
            "reason": "duration_unknown",
            "failure_floor_seconds": 0.0,
            "safe_success_ceiling_seconds": 0.0,
        }
    learning_state = diarization_learning_state(history_state, fingerprint)
    safe_success_ceiling = float(learning_state.get("safe_success_ceiling") or 0.0)
    failure_floor = float(learning_state.get("failure_floor") or 0.0)
    if failure_floor <= 0:
        return {
            "mode": "global",
            "probe": False,
            "learned_route": False,
            "reason": "no_failure_history",
            "failure_floor_seconds": failure_floor,
            "safe_success_ceiling_seconds": safe_success_ceiling,
        }
    if duration_seconds <= safe_success_ceiling:
        return {
            "mode": "global",
            "probe": False,
            "learned_route": True,
            "reason": "below_safe_success_ceiling",
            "failure_floor_seconds": failure_floor,
            "safe_success_ceiling_seconds": safe_success_ceiling,
        }
    if duration_seconds <= failure_floor:
        return {
            "mode": "chunked_preemptive",
            "probe": False,
            "learned_route": True,
            "reason": "at_or_below_failure_floor",
            "failure_floor_seconds": failure_floor,
            "safe_success_ceiling_seconds": safe_success_ceiling,
        }
    if duration_seconds > failure_floor + DIARIZATION_PROBE_MARGIN_SECONDS:
        return {
            "mode": "chunked_preemptive",
            "probe": False,
            "learned_route": True,
            "reason": "above_failure_floor_plus_probe_margin",
            "failure_floor_seconds": failure_floor,
            "safe_success_ceiling_seconds": safe_success_ceiling,
        }
    probe_allowed, probe_reason = should_probe_diarization_duration(duration_seconds, learning_state)
    return {
        "mode": "global" if probe_allowed else "chunked_preemptive",
        "probe": probe_allowed,
        "learned_route": True,
        "reason": probe_reason,
        "failure_floor_seconds": failure_floor,
        "safe_success_ceiling_seconds": safe_success_ceiling,
    }


def update_diarization_history(
    output_dir: Path,
    runtime_fingerprint: Dict[str, object],
    audio_path: Path,
    duration_seconds: float,
    mode: str,
    outcome: str,
    probe: bool = False,
):
    history_state = load_diarization_history(output_dir)
    record = {
        "audio_file": audio_path.name,
        "duration_seconds": float(duration_seconds or 0.0),
        "mode": str(mode or ""),
        "outcome": str(outcome or ""),
        "probe": bool(probe),
        "timestamp_epoch_seconds": current_time_epoch_seconds(),
        "runtime_fingerprint": runtime_fingerprint,
        "invalidated": False,
    }
    history_state.setdefault("records", []).append(record)
    if mode == "global" and outcome == "success":
        for existing in history_state["records"]:
            if (
                isinstance(existing, dict)
                and existing.get("runtime_fingerprint") == runtime_fingerprint
                and existing.get("mode") == "global"
                and existing.get("outcome") == "memory_error"
                and not existing.get("invalidated")
                and float(existing.get("duration_seconds") or 0.0) >= float(duration_seconds or 0.0)
            ):
                existing["invalidated"] = True
                existing["invalidated_by_audio_file"] = audio_path.name
                existing["invalidated_at_epoch_seconds"] = record["timestamp_epoch_seconds"]
    save_diarization_history(output_dir, history_state)


def load_review_calibration_session(
    output_dir: Path,
    backend_capabilities: Dict[str, object],
) -> ReviewCalibrationSession:
    return ReviewCalibrationSession(
        backend_capabilities,
        state_load_review_calibration_state(review_calibration_state_path(output_dir)),
    )


def save_review_calibration_session(output_dir: Path, session: Optional[ReviewCalibrationSession]):
    if session is None:
        return
    state_save_review_calibration_state(review_calibration_state_path(output_dir), session.serialize())


def parse_args():
    """Parse CLI options for parent batch runs and isolated child workers."""

    parser = argparse.ArgumentParser(description="Transcribe podcasts with diarization and host labeling.")
    parser.add_argument("--input-dir", help="Directory containing audio files to process.")
    parser.add_argument("--input-file", help="Optional single audio file to process from input-dir.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to input directory.")
    parser.add_argument(
        "--workflow-profile",
        choices=sorted(WORKFLOW_PROFILES),
        default="podcast",
        help="Processing profile: podcast preserves identity/review behavior; anonymous_meeting keeps diarization labels only.",
    )
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name.")
    parser.add_argument("--model-id", default="", help="Canonical ASR model repository id. Required when --model is only a shorthand.")
    parser.add_argument("--model-revision", default="", help="Immutable ASR model revision used for acquisition and provenance.")
    parser.add_argument(
        "--asr-provider",
        choices=["faster_whisper", "parakeet"],
        default="faster_whisper",
        help="ASR provider. Additional providers remain experimental until benchmarked.",
    )
    parser.add_argument("--language", default="en", help="Language code.")
    parser.add_argument("--device", default="auto", help="Whisper device: auto, cpu, or cuda.")
    # "auto" can pick CPU paths or unsupported configs. 5070 Ti → float16 is correct and fastest
    # parser.add_argument("--compute-type", default="auto", help="faster-whisper compute type.")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute type.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding.")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for ASR. 0 uses a conservative adaptive CPU/CUDA default.")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Hugging Face token for pyannote pipeline.")
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote diarization pipeline id.",
    )
    parser.add_argument(
        "--speaker-model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="Speaker verification model id.",
    )
    parser.add_argument(
        "--alignment-provider",
        choices=sorted(ALIGNMENT_PROVIDERS),
        default="timestamp_passthrough",
        help="Word-alignment provider. timestamp_passthrough preserves current behavior; whisperx enables forced alignment.",
    )
    parser.add_argument(
        "--alignment-model",
        default="",
        help="Optional alignment model override. Blank uses the provider's language default.",
    )
    parser.add_argument("--alignment-model-revision", default="", help="Immutable forced-alignment model revision.")
    parser.add_argument("--diarization-model-revision", default="", help="Immutable diarization model revision.")
    parser.add_argument("--speaker-model-revision", default="", help="Immutable speaker-embedding model revision.")
    parser.add_argument(
        "--provider-cache-dir",
        default="config/provider-models",
        help="Repo-local cache for explicitly acquired provider artifacts.",
    )
    parser.add_argument("--provider-preflight", action="store_true", help="Report selected provider availability without downloading or loading models.")
    parser.add_argument("--download-provider-models", action="store_true", help="Explicitly acquire the selected pinned provider artifacts, then exit.")
    parser.add_argument(
        "--speaker-embedding-provider",
        choices=["speechbrain_ecapa", "speechbrain_xvector"],
        default="speechbrain_ecapa",
        help="Speaker embedding provider used for host and known-speaker identity.",
    )
    parser.add_argument(
        "--host-reference",
        help="Optional audio file containing a clean sample of the host voice. Strongly recommended for stable host labeling.",
    )
    parser.add_argument(
        "--host-profile-json",
        default="host_profile.json",
        help="Path to a JSON file used to persist a host embedding profile across episodes.",
    )
    parser.add_argument(
        "--known-speakers-dir",
        help="Optional directory containing speakers.json plus named reference audio clips for known speakers.",
    )
    parser.add_argument(
        "--preferred-terms-file",
        help="Optional text file with one preferred term per line. Used as prompt/hotword biasing.",
    )
    parser.add_argument(
        "--preferred-term",
        dest="preferred_terms",
        action="append",
        default=[],
        help="Additional protected preferred term. Repeat for multiple inline terms.",
    )
    parser.add_argument(
        "--replacement-map-json",
        help="Optional JSON file mapping preferred spellings to likely mistranscriptions.",
    )
    parser.add_argument(
        "--filename-date-preset",
        choices=["strict_iso", "american_podcast", "mixed_common"],
        default="strict_iso",
        help="Built-in filename date parser preset used to extract episode dates from audio filenames.",
    )
    parser.add_argument(
        "--filename-date-position",
        choices=["first", "last"],
        default="last",
        help="Whether to use the first or last valid date match found in the filename.",
    )
    parser.add_argument(
        "--filename-date-formats",
        nargs="+",
        choices=[
            "YYYYMMDD",
            "YYYY-MM-DD",
            "YYYY_MM_DD",
            "YYYY.MM.DD",
            "MM-DD-YYYY",
            "MM_DD_YYYY",
            "MM.DD.YYYY",
            "DD-MM-YYYY",
            "DD_MM_YYYY",
            "DD.MM.YYYY",
        ],
        help="Optional ordered list of accepted filename date formats. Overrides the selected preset when provided.",
    )
    parser.add_argument(
        "--cleanup-level",
        choices=["disabled", "conservative", "normal", "aggressive"],
        default="normal",
        help="Speech cleanup level for cleaned transcript companion outputs.",
    )
    parser.add_argument(
        "--corrections-dir",
        help=(
            "Optional directory containing manual correction CSVs named "
            "<audio_stem>_corrections.csv. Supported columns: segment_id/id, corrected_text/text, speaker."
        ),
    )
    parser.add_argument(
        "--runtime-profile",
        choices=sorted(RUNTIME_PROFILES),
        default=DEFAULT_RUNTIME_PROFILE,
        help="Optional post-processing runtime profile for additive transcript review.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(REVIEW_BACKENDS),
        default=DEFAULT_REVIEW_BACKEND,
        help="Optional review backend for additive transcript review.",
    )
    parser.add_argument(
        "--review-base-url",
        default="",
        help="OpenAI-compatible base URL for optional transcript review backends.",
    )
    parser.add_argument(
        "--review-model-name",
        default="",
        help="Model name used for optional transcript review calls.",
    )
    parser.add_argument(
        "--review-reasoning-effort",
        choices=["none", "low", "medium", "xhigh"],
        default="none",
        help="Per-request Qwen reasoning mode for transcript review. 'none' disables thinking.",
    )
    parser.add_argument(
        "--review-batch-token-limit",
        type=int,
        default=DEFAULT_REVIEW_BATCH_TOKEN_LIMIT,
        help="Hard token ceiling for each review request. Smaller batches keep remote review latency bounded.",
    )
    parser.add_argument(
        "--review-all-segments",
        dest="review_candidate_filter",
        action="store_false",
        help="Review every segment instead of only changed or uncertain segments.",
    )
    parser.add_argument("--review-context-budget", type=int, default=0, help="Custom-profile review context budget.")
    parser.add_argument(
        "--review-structured-output-support",
        dest="review_structured_output_support",
        action="store_true",
        help="Declare structured-output support for the custom review profile.",
    )
    parser.add_argument(
        "--review-transcript-qa-available",
        dest="review_transcript_qa_available",
        action="store_true",
        help="Declare transcript-QA capability for the custom review profile.",
    )
    parser.add_argument(
        "--review-episode-wide-correction-available",
        dest="review_episode_wide_correction_available",
        action="store_true",
        help="Declare episode-wide correction capability for the custom review profile.",
    )
    parser.add_argument(
        "--review-debug",
        action="store_true",
        help="Write per-stage review request/response debug artifacts for backend troubleshooting.",
    )
    parser.add_argument(
        "--review-debug-dir",
        default="",
        help="Optional directory override for review debug artifacts. Defaults to the episode artifact folder in output.",
    )
    parser.add_argument(
        "--review-auto-calibrate",
        dest="review_auto_calibrate",
        action="store_true",
        help="Probe the active review backend at the first review step and reuse calibrated batching budgets for the rest of the run.",
    )
    parser.add_argument(
        "--no-review-auto-calibrate",
        dest="review_auto_calibrate",
        action="store_false",
        help="Disable review-budget calibration and use fixed review batching defaults.",
    )
    parser.add_argument(
        "--review-auto-adapt-upward",
        dest="review_auto_adapt_upward",
        action="store_true",
        help="Allow conservative upward drift of local review batching budgets after long stable success streaks.",
    )
    parser.add_argument(
        "--no-review-auto-adapt-upward",
        dest="review_auto_adapt_upward",
        action="store_false",
        help="Disable upward drift for calibrated local review batching budgets.",
    )
    parser.add_argument(
        "--transcript-cleanup-review",
        dest="transcript_cleanup_review",
        action="store_true",
        help="Enable additive LLM transcript cleanup review.",
    )
    parser.add_argument(
        "--no-transcript-cleanup-review",
        dest="transcript_cleanup_review",
        action="store_false",
        help="Disable additive LLM transcript cleanup review.",
    )
    parser.add_argument(
        "--glossary-correction-review",
        dest="glossary_correction_review",
        action="store_true",
        help="Enable additive LLM glossary correction review.",
    )
    parser.add_argument(
        "--no-glossary-correction-review",
        dest="glossary_correction_review",
        action="store_false",
        help="Disable additive LLM glossary correction review.",
    )
    parser.add_argument(
        "--speaker-consistency-review",
        dest="speaker_consistency_review",
        action="store_true",
        help="Enable additive LLM speaker consistency review.",
    )
    parser.add_argument(
        "--no-speaker-consistency-review",
        dest="speaker_consistency_review",
        action="store_false",
        help="Disable additive LLM speaker consistency review.",
    )
    parser.add_argument(
        "--episode-qa-review",
        dest="episode_qa_review",
        action="store_true",
        help="Enable additive long-context episode QA review when supported by the active runtime profile.",
    )
    parser.add_argument(
        "--no-episode-qa-review",
        dest="episode_qa_review",
        action="store_false",
        help="Disable additive long-context episode QA review.",
    )
    parser.add_argument(
        "--no-resume-intermediates",
        dest="resume_intermediates",
        action="store_false",
        help="Disable reuse of fingerprint-compatible per-episode stage artifacts.",
    )
    parser.add_argument(
        "--child-timeout-seconds",
        type=int,
        default=0,
        help="Optional timeout for isolated child processes. 0 disables the timeout.",
    )
    parser.add_argument(
        "--archive-debug-artifacts",
        action="store_true",
        help="Keep review/debug material after successful output writing.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run preflight and print a benchmark plan without loading ML models or processing audio.",
    )
    parser.add_argument(
        "--review-benchmark",
        action="store_true",
        help="Run the dedicated tier-2 review benchmark suite against the checked-in cleaned-transcript fixtures.",
    )
    parser.add_argument(
        "--pipeline-benchmark",
        action="store_true",
        help="Evaluate processed transcript outputs against the versioned podcast pipeline gold set.",
    )
    parser.add_argument(
        "--gold-set-dir",
        default="",
        help="Directory containing the pipeline gold-set manifest and reference transcript JSON files.",
    )
    parser.add_argument(
        "--evaluation-pack-path",
        default="",
        help="External private evaluation-pack directory. Takes precedence over --gold-set-dir.",
    )
    parser.add_argument(
        "--benchmark-candidate-dir",
        default="",
        help="Processed output directory containing candidate cleaned/reviewed transcript JSON files.",
    )
    parser.add_argument(
        "--benchmark-baseline-dir",
        default="",
        help="Optional processed-output directory used as the baseline for candidate comparison and promotion gates.",
    )
    parser.add_argument("--speech-run-id", default="", help="Publish this benchmark as an immutable shadow speech run.")
    parser.add_argument(
        "--speech-shadow-root",
        default="",
        help="Immutable speech-run root. Defaults to <output-dir>/speech-shadow-runs.",
    )
    parser.add_argument(
        "--assume-dominant-speaker-is-host",
        action="store_true",
        help="When no host reference/profile exists, label the speaker with the most talk time as HOST and bootstrap the profile.",
    )
    parser.add_argument(
        "--host-threshold",
        type=float,
        default=0.45,
        help="Cosine similarity threshold for matching a speaker to the host profile/reference.",
    )
    parser.add_argument(
        "--min-host-seconds",
        type=float,
        default=20.0,
        help="Minimum diarized speech duration required before using a speaker to update the host profile.",
    )
    parser.add_argument(
        "--max-embedding-seconds",
        type=float,
        default=90.0,
        help="Maximum total speech duration per speaker used to build an embedding.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Optional fixed speaker count for pyannote diarization.",
    )
    parser.add_argument(
        "--isolate-files",
        dest="isolate_files",
        action="store_true",
        help="Process each episode in a separate Python child process so native memory is released between files.",
    )
    parser.add_argument(
        "--no-isolate-files",
        dest="isolate_files",
        action="store_false",
        help="Process all episodes in the current Python process.",
    )
    parser.set_defaults(
        transcript_cleanup_review=None,
        glossary_correction_review=None,
        speaker_consistency_review=None,
        episode_qa_review=None,
        review_auto_calibrate=None,
        review_auto_adapt_upward=None,
        review_structured_output_support=False,
        review_transcript_qa_available=False,
        review_episode_wide_correction_available=False,
        review_candidate_filter=True,
    )
    parser.set_defaults(isolate_files=False)
    parser.set_defaults(resume_intermediates=True)
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_runtime_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_ffprobe_path() -> Optional[str]:
    ffmpeg_bin_dir = os.getenv("PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR") or os.getenv("FFMPEG_BIN_DIR")
    if ffmpeg_bin_dir:
        candidate = Path(ffmpeg_bin_dir) / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if candidate.exists():
            return str(candidate)

    fallback = shutil.which("ffprobe")
    conda_prefix = os.getenv("CONDA_PREFIX")
    if fallback and conda_prefix:
        try:
            if Path(fallback).resolve().is_relative_to(Path(conda_prefix).resolve()):
                # Conda's ffprobe can resolve against incompatible GTK DLLs on Windows.
                # The launcher supplies the supported external FFmpeg build; without it,
                # returning no probe is safer than opening a native DLL error dialog.
                return None
        except (OSError, ValueError):
            pass
    return fallback


def resolve_ffmpeg_path() -> Optional[str]:
    ffmpeg_bin_dir = os.getenv("PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR") or os.getenv("FFMPEG_BIN_DIR")
    if ffmpeg_bin_dir:
        candidate = Path(ffmpeg_bin_dir) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if candidate.exists():
            return str(candidate)

    fallback = shutil.which("ffmpeg")
    conda_prefix = os.getenv("CONDA_PREFIX")
    if fallback and conda_prefix:
        try:
            if Path(fallback).resolve().is_relative_to(Path(conda_prefix).resolve()):
                # Conda's FFmpeg can resolve against incompatible GTK DLLs on Windows.
                # The launcher supplies the supported external build; without it,
                # returning no encoder is safer than opening a native DLL error dialog.
                return None
        except (OSError, ValueError):
            pass
    return fallback


def get_audio_metadata(path: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    ffprobe_path = resolve_ffprobe_path()
    if ffprobe_path:
        try:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate,duration",
                    "-of",
                    "json",
                    path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)
            streams = payload.get("streams", [])
            if streams:
                stream = streams[0]
                sample_rate_text = stream.get("sample_rate")
                duration_text = stream.get("duration")
                sample_rate = int(sample_rate_text) if sample_rate_text else None
                duration_seconds = float(duration_text) if duration_text else None
                num_frames = (
                    int(round(duration_seconds * sample_rate))
                    if duration_seconds is not None and sample_rate is not None and sample_rate > 0
                    else None
                )
                return sample_rate, num_frames, duration_seconds
        except Exception:
            pass

    try:
        metadata = torchaudio.info(path)
        sample_rate = metadata.sample_rate if metadata.sample_rate > 0 else None
        num_frames = metadata.num_frames if metadata.num_frames > 0 else None
        duration_seconds = (
            float(num_frames) / float(sample_rate)
            if sample_rate is not None and num_frames is not None
            else None
        )
        return sample_rate, num_frames, duration_seconds
    except Exception:
        return None, None, None


def get_audio_duration_seconds(path: str) -> Optional[float]:
    _, _, duration_seconds = get_audio_metadata(path)
    return duration_seconds


def get_process_memory_mb() -> Optional[float]:
    if os.name != "nt":
        return None

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        process_handle = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if success:
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "$p = Get-Process -Id $PID; [math]::Round($p.WorkingSet64 / 1MB, 2)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        value = result.stdout.strip()
        return float(value) if value else None
    except Exception:
        return None


def format_memory_mb(memory_mb: Optional[float]) -> str:
    if memory_mb is None:
        return "unknown"
    return f"{memory_mb:.0f} MiB"


RESOURCE_USAGE_TRACKER: Dict[str, float] = {}

SPEAKER_TELEMETRY_VERSION = 1
SPEAKER_TELEMETRY_PROGRESS_INTERVAL = 10


def new_speaker_telemetry(progress_path: Optional[Path] = None) -> Dict[str, object]:
    """Create counters for the audio and embedding portions of speaker work."""

    return {
        "telemetry_version": SPEAKER_TELEMETRY_VERSION,
        "audio_cache_mode": "not_needed",
        "audio_cache_reused": False,
        "audio_cache_conversion_wall_seconds": 0.0,
        "audio_cache_path": "",
        "audio_span_read_count": 0,
        "audio_span_error_count": 0,
        "audio_span_requested_seconds": 0.0,
        "audio_span_loaded_seconds": 0.0,
        "audio_span_wall_seconds": 0.0,
        "audio_span_wall_seconds_by_operation": {},
        "embedding_call_count": 0,
        "embedding_failed_call_count": 0,
        "embedding_input_seconds": 0.0,
        "embedding_wall_seconds": 0.0,
        "embedding_dispatch_seconds": 0.0,
        "embedding_sync_copy_seconds": 0.0,
        "embedding_calls_by_kind": {},
        "embedding_wall_seconds_by_kind": {},
        "embedding_input_seconds_by_kind": {},
        "chunk_reconciliation_boundary_count": 0,
        "_progress_path": str(progress_path) if progress_path else "",
        "_last_progress_operation_count": 0,
    }


def _telemetry_map_add(telemetry: Optional[Dict[str, object]], key: str, name: str, value: float):
    if telemetry is None:
        return
    values = telemetry.setdefault(key, {})
    if not isinstance(values, dict):
        values = {}
        telemetry[key] = values
    values[name] = float(values.get(name, 0.0)) + float(value)


def _telemetry_map_increment(telemetry: Optional[Dict[str, object]], key: str, name: str):
    if telemetry is None:
        return
    values = telemetry.setdefault(key, {})
    if not isinstance(values, dict):
        values = {}
        telemetry[key] = values
    values[name] = int(values.get(name, 0)) + 1


def finalize_speaker_telemetry(telemetry: Dict[str, object]) -> Dict[str, object]:
    """Return JSON-safe telemetry without internal live-progress fields."""

    result = {}
    for key, value in telemetry.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, float):
            result[key] = round(value, 4)
        elif isinstance(value, dict):
            result[key] = {
                str(item_key): round(item_value, 4) if isinstance(item_value, float) else item_value
                for item_key, item_value in value.items()
            }
        else:
            result[key] = value
    return result


def _maybe_emit_speaker_telemetry(telemetry: Optional[Dict[str, object]]):
    """Print and checkpoint progress periodically during long speaker runs."""

    if telemetry is None:
        return
    audio_reads = int(telemetry.get("audio_span_read_count") or 0)
    embedding_calls = int(telemetry.get("embedding_call_count") or 0)
    operation_count = audio_reads + embedding_calls
    last_count = int(telemetry.get("_last_progress_operation_count") or 0)
    if operation_count <= 0 or operation_count - last_count < SPEAKER_TELEMETRY_PROGRESS_INTERVAL:
        return
    telemetry["_last_progress_operation_count"] = operation_count
    print(
        "    speaker telemetry: "
        f"audio_reads={audio_reads}, embeddings={embedding_calls}, "
        f"audio_wall={float(telemetry.get('audio_span_wall_seconds') or 0.0):.1f}s, "
        f"embedding_wall={float(telemetry.get('embedding_wall_seconds') or 0.0):.1f}s"
    )
    progress_path = str(telemetry.get("_progress_path") or "").strip()
    if progress_path:
        try:
            state_atomic_write_text(
                Path(progress_path),
                json.dumps({**finalize_speaker_telemetry(telemetry), "updated_at_epoch": time.time()}, indent=2),
            )
        except OSError:
            # Telemetry must never interrupt the processing path.
            pass


def _record_audio_span_telemetry(
    telemetry: Optional[Dict[str, object]],
    operation: str,
    requested_seconds: float,
    loaded_seconds: float,
    elapsed_seconds: float,
):
    if telemetry is None:
        return
    telemetry["audio_span_read_count"] = int(telemetry.get("audio_span_read_count") or 0) + 1
    telemetry["audio_span_requested_seconds"] = float(telemetry.get("audio_span_requested_seconds") or 0.0) + requested_seconds
    telemetry["audio_span_loaded_seconds"] = float(telemetry.get("audio_span_loaded_seconds") or 0.0) + loaded_seconds
    telemetry["audio_span_wall_seconds"] = float(telemetry.get("audio_span_wall_seconds") or 0.0) + elapsed_seconds
    _telemetry_map_add(telemetry, "audio_span_wall_seconds_by_operation", operation, elapsed_seconds)
    _maybe_emit_speaker_telemetry(telemetry)


def _record_embedding_telemetry(
    telemetry: Optional[Dict[str, object]],
    kind: str,
    input_seconds: float,
    wall_seconds: float,
    dispatch_seconds: float,
    sync_copy_seconds: float,
    failed: bool = False,
):
    if telemetry is None:
        return
    telemetry["embedding_call_count"] = int(telemetry.get("embedding_call_count") or 0) + 1
    if failed:
        telemetry["embedding_failed_call_count"] = int(telemetry.get("embedding_failed_call_count") or 0) + 1
    telemetry["embedding_input_seconds"] = float(telemetry.get("embedding_input_seconds") or 0.0) + input_seconds
    telemetry["embedding_wall_seconds"] = float(telemetry.get("embedding_wall_seconds") or 0.0) + wall_seconds
    telemetry["embedding_dispatch_seconds"] = float(telemetry.get("embedding_dispatch_seconds") or 0.0) + dispatch_seconds
    telemetry["embedding_sync_copy_seconds"] = float(telemetry.get("embedding_sync_copy_seconds") or 0.0) + sync_copy_seconds
    _telemetry_map_increment(telemetry, "embedding_calls_by_kind", kind)
    _telemetry_map_add(telemetry, "embedding_wall_seconds_by_kind", kind, wall_seconds)
    _telemetry_map_add(telemetry, "embedding_input_seconds_by_kind", kind, input_seconds)
    _maybe_emit_speaker_telemetry(telemetry)


def log_memory_usage(stage_label: str):
    process_memory = get_process_memory_mb()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        RESOURCE_USAGE_TRACKER["peak_gpu_allocated_mib"] = max(allocated, RESOURCE_USAGE_TRACKER.get("peak_gpu_allocated_mib", 0.0))
        RESOURCE_USAGE_TRACKER["peak_gpu_reserved_mib"] = max(reserved, RESOURCE_USAGE_TRACKER.get("peak_gpu_reserved_mib", 0.0))
        print(
            f"  memory [{stage_label}]: cpu_working_set={format_memory_mb(process_memory)}, "
            f"gpu_allocated={allocated:.0f} MiB, gpu_reserved={reserved:.0f} MiB"
        )
    else:
        print(f"  memory [{stage_label}]: cpu_working_set={format_memory_mb(process_memory)}")
    if process_memory is not None:
        RESOURCE_USAGE_TRACKER["peak_cpu_working_set_mib"] = max(process_memory, RESOURCE_USAGE_TRACKER.get("peak_cpu_working_set_mib", 0.0))


def load_preferred_terms(path: Optional[str]) -> List[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_speaker_verifier(model_id: str, device: str, provider_name: str = "speechbrain_ecapa"):
    if provider_name == "speechbrain_xvector":
        from speechbrain.inference.classifiers import EncoderClassifier

        return EncoderClassifier.from_hparams(
            source=model_id or "speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_speaker_model_xvector",
            run_opts={"device": normalize_runtime_device(device)},
        )
    from speechbrain.inference.speaker import SpeakerRecognition

    return SpeakerRecognition.from_hparams(
        source=model_id,
        savedir="pretrained_speaker_model",
        run_opts={"device": normalize_runtime_device(device)},
    )


def build_prompt_bias(terms: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not terms:
        return None, None
    hotwords = ", ".join(terms)
    initial_prompt = (
        "Domain vocabulary and preferred spellings: "
        f"{hotwords}. Use these spellings when they match the audio."
    )
    return initial_prompt, hotwords


def load_replacement_map(path: Optional[str]) -> Dict[str, List[str]]:
    return config_load_replacement_map(path)


def apply_replacements(text: str, replacement_map: Dict[str, List[str]]) -> str:
    updated = text
    for preferred, aliases in replacement_map.items():
        for alias in aliases:
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            updated = pattern.sub(preferred, updated)
    return updated


def detect_replacement_hits(text: str, replacement_map: Dict[str, List[str]]) -> List[Dict[str, str]]:
    hits = []
    for preferred, aliases in replacement_map.items():
        for alias in aliases:
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            if pattern.search(text):
                hits.append({"preferred": preferred, "alias": alias})
    return hits


def load_audio_mono_16k(path: str, chunk_seconds: float = 300.0) -> torch.Tensor:
    sample_rate, num_frames, _ = get_audio_metadata(path)

    if sample_rate is None or sample_rate <= 0 or num_frames is None or num_frames <= 0:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return waveform.squeeze(0)

    frames_per_chunk = max(sample_rate, int(sample_rate * chunk_seconds))
    resampler = (
        torchaudio.transforms.Resample(sample_rate, 16000)
        if sample_rate != 16000
        else None
    )
    chunks = []

    for frame_offset in range(0, num_frames, frames_per_chunk):
        frames_to_read = min(frames_per_chunk, num_frames - frame_offset)
        waveform, _ = torchaudio.load(path, frame_offset=frame_offset, num_frames=frames_to_read)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if resampler is not None:
            waveform = resampler(waveform)
        chunks.append(waveform.squeeze(0).contiguous())
        del waveform

    if not chunks:
        return torch.empty(0, dtype=torch.float32)

    if len(chunks) == 1:
        return chunks[0]

    return torch.cat(chunks, dim=0)


def load_audio_span_mono_16k(
    path: str,
    start_seconds: float,
    end_seconds: float,
    sample_rate: Optional[int] = None,
    resampler: Optional[torchaudio.transforms.Resample] = None,
) -> torch.Tensor:
    if sample_rate is None:
        sample_rate, _, _ = get_audio_metadata(path)
    if sample_rate is None or sample_rate <= 0:
        waveform = load_audio_mono_16k(path)
        start_frame = max(0, int(start_seconds * 16000))
        end_frame = max(start_frame, int(end_seconds * 16000))
        return waveform[start_frame:end_frame].contiguous()

    start_frame = max(0, int(start_seconds * sample_rate))
    end_frame = max(start_frame, int(end_seconds * sample_rate))
    num_frames = max(0, end_frame - start_frame)
    if num_frames == 0:
        return torch.empty(0, dtype=torch.float32)

    # Speaker matching normally reads the temporary 16 kHz PCM cache.  Read
    # that format directly so a native compressed-audio decoder cannot retain
    # or allocate a large file-sized buffer for every timestamped span.
    if Path(path).suffix.lower() in {".wav", ".wave"} and sample_rate == 16000 and resampler is None:
        pcm_span = _load_pcm_wav_span_mono_16k(path, start_frame, num_frames)
        if pcm_span is not None:
            return pcm_span

    waveform, _ = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)
    # Some backends are forgiving about frame requests and may return more
    # samples than requested.  Never retain those extra samples in a speaker
    # span; the requested duration is the memory bound for this operation.
    if waveform.shape[-1] > num_frames:
        waveform = waveform[..., :num_frames]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if resampler is not None:
        waveform = resampler(waveform)
    elif sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.squeeze(0).contiguous()


def _load_pcm_wav_span_mono_16k(
    path: str,
    start_frame: int,
    num_frames: int,
) -> Optional[torch.Tensor]:
    """Read a bounded span from the PCM WAV speaker cache.

    The cache is deliberately mono, 16 kHz, signed 16-bit PCM.  The standard
    library reader seeks to the requested frame and reads only that span, which
    avoids the whole-file allocations seen with some native audio backends.
    Returning ``None`` lets callers fall back to torchaudio for an unexpected
    WAV variant.
    """

    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getframerate() != 16000
                or reader.getsampwidth() != 2
                or reader.getnchannels() <= 0
            ):
                return None
            total_frames = reader.getnframes()
            bounded_start = min(max(0, int(start_frame)), total_frames)
            bounded_count = min(max(0, int(num_frames)), max(0, total_frames - bounded_start))
            if bounded_count == 0:
                return torch.empty(0, dtype=torch.float32)
            reader.setpos(bounded_start)
            raw = reader.readframes(bounded_count)
            channels = reader.getnchannels()
    except (OSError, EOFError, wave.Error):
        return None

    if not raw:
        return torch.empty(0, dtype=torch.float32)

    pcm = np.frombuffer(raw, dtype="<i2")
    complete_samples = pcm.size - (pcm.size % channels)
    if complete_samples <= 0:
        return torch.empty(0, dtype=torch.float32)
    pcm = pcm[:complete_samples].reshape(-1, channels)
    if channels == 1:
        samples = pcm[:, 0].astype(np.float32) / 32768.0
    else:
        samples = pcm.astype(np.float32).mean(axis=1) / 32768.0
    return torch.from_numpy(samples).contiguous()


def speaker_audio_cache_paths(output_dir: Path, audio_path: Path) -> Tuple[Path, Path]:
    cache_dir = output_dir / ARTIFACT_DIRNAME / audio_path.stem
    return cache_dir / "speaker_audio_16k_mono.wav", cache_dir / "speaker_audio_16k_mono.json"


def _speaker_audio_cache_metadata_matches(
    metadata_path: Path,
    cache_path: Path,
    source_fingerprint: Dict[str, object],
) -> bool:
    try:
        if not cache_path.exists() or cache_path.stat().st_size <= 44 or not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("cache_version") == SPEAKER_AUDIO_CACHE_VERSION
        and metadata.get("source_fingerprint") == source_fingerprint
        and int(metadata.get("sample_rate") or 0) == SPEAKER_AUDIO_CACHE_SAMPLE_RATE
        and int(metadata.get("channels") or 0) == 1
        and str(metadata.get("codec") or "") == "pcm_s16le"
    )


def prepare_speaker_audio_cache(
    audio_path: Path,
    output_dir: Path,
    telemetry: Optional[Dict[str, object]] = None,
    component_timings: Optional[Dict[str, float]] = None,
) -> Path:
    """Use a seek-friendly PCM source for repeated speaker-span reads.

    The original source remains the input to transcription and diarization.  This
    cache is only used after those stages, when speaker matching requests many
    timestamped spans from a compressed source.
    """

    suffix = audio_path.suffix.lower()
    if suffix not in SPEAKER_AUDIO_CACHE_SOURCE_SUFFIXES:
        if telemetry is not None:
            telemetry["audio_cache_mode"] = "not_needed"
            telemetry["audio_cache_path"] = str(audio_path)
        return audio_path

    source_fingerprint = audio_file_fingerprint(audio_path)
    cache_path, metadata_path = speaker_audio_cache_paths(output_dir, audio_path)
    if _speaker_audio_cache_metadata_matches(metadata_path, cache_path, source_fingerprint):
        if telemetry is not None:
            telemetry["audio_cache_mode"] = "reused"
            telemetry["audio_cache_reused"] = True
            telemetry["audio_cache_path"] = str(cache_path)
        print(f"  speaker audio cache: reusing {cache_path.name}")
        return cache_path

    ffmpeg_path = resolve_ffmpeg_path()
    if not ffmpeg_path:
        if telemetry is not None:
            telemetry["audio_cache_mode"] = "unavailable"
            telemetry["audio_cache_path"] = str(audio_path)
        print("  speaker audio cache unavailable: FFmpeg was not found; using the original source")
        return audio_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    conversion_started = time.perf_counter()
    print("  speaker audio cache: converting source to mono 16 kHz PCM WAV")
    succeeded = False
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(audio_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SPEAKER_AUDIO_CACHE_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(temporary_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if not temporary_path.exists() or temporary_path.stat().st_size <= 44:
            raise RuntimeError("FFmpeg completed without producing a valid WAV file")
        os.replace(temporary_path, cache_path)
        state_atomic_write_text(
            metadata_path,
            json.dumps(
                {
                    "cache_version": SPEAKER_AUDIO_CACHE_VERSION,
                    "source_file": audio_path.name,
                    "source_fingerprint": source_fingerprint,
                    "sample_rate": SPEAKER_AUDIO_CACHE_SAMPLE_RATE,
                    "channels": 1,
                    "codec": "pcm_s16le",
                },
                indent=2,
            ),
        )
        succeeded = True
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        detail = str(getattr(exc, "stderr", "") or exc).strip().replace("\r", " ").replace("\n", " ")
        print(f"  speaker audio cache unavailable: {detail or 'conversion failed'}; using the original source")
        try:
            if cache_path.exists() and not metadata_path.exists():
                cache_path.unlink()
        except OSError:
            pass
    finally:
        try:
            if temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            pass

    conversion_elapsed = time.perf_counter() - conversion_started
    if component_timings is not None:
        component_timings["audio cache conversion"] = (
            component_timings.get("audio cache conversion", 0.0) + conversion_elapsed
        )
    if telemetry is not None:
        telemetry["audio_cache_mode"] = "created" if succeeded else "fallback"
        telemetry["audio_cache_conversion_wall_seconds"] = conversion_elapsed
        telemetry["audio_cache_path"] = str(cache_path if succeeded else audio_path)
    if succeeded:
        print(f"  speaker audio cache: ready in {conversion_elapsed:.1f}s")
        return cache_path
    return audio_path


def load_host_profile(
    path: Optional[str],
    expected_provider: Optional[ProviderIdentity] = None,
) -> Optional[np.ndarray]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    provider_payload = payload.get("embedding_provider")
    if expected_provider is not None and isinstance(provider_payload, dict) and provider_payload:
        if (
            str(provider_payload.get("provider") or "") != expected_provider.provider
            or str(provider_payload.get("model") or "") != expected_provider.model
        ):
            print(
                "Host profile provider mismatch; ignoring incompatible profile "
                f"({provider_payload.get('provider')}:{provider_payload.get('model')} != "
                f"{expected_provider.provider}:{expected_provider.model})."
            )
            return None
    elif expected_provider is not None and expected_provider.provider != "speechbrain_ecapa":
        print("Legacy host profile has no provider identity and cannot be used with a non-ECAPA provider.")
        return None
    vector = payload.get("embedding")
    if not isinstance(vector, list):
        return None
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return None
    return arr / norm


def load_known_speakers_config(known_speakers_dir: Optional[str]) -> List[Dict[str, object]]:
    if not known_speakers_dir:
        return []

    config_path = Path(known_speakers_dir) / "speakers.json"
    if not config_path.exists():
        return []

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    speakers = payload.get("speakers", [])
    return speakers if isinstance(speakers, list) else []


def audio_reference_quality(waveform: torch.Tensor) -> Dict[str, object]:
    if waveform.numel() == 0:
        return reference_sample_quality(0.0, rms=0.0, peak=0.0, speech_ratio=0.0)
    flat = waveform.flatten().float()
    duration_seconds = flat.numel() / 16000.0
    rms = float(torch.sqrt(torch.mean(flat * flat)).item())
    peak = float(torch.max(torch.abs(flat)).item())
    frame_size = 1600
    if flat.numel() < frame_size:
        speech_ratio = 1.0 if rms >= 0.005 else 0.0
    else:
        frames = flat[: flat.numel() - (flat.numel() % frame_size)].reshape(-1, frame_size)
        frame_rms = torch.sqrt(torch.mean(frames * frames, dim=1))
        speech_ratio = float((frame_rms >= 0.005).float().mean().item())
    return reference_sample_quality(duration_seconds, rms=rms, peak=peak, speech_ratio=speech_ratio)


def save_host_profile(
    path: Optional[str],
    embedding: Optional[np.ndarray],
    source: str,
    provider: Optional[ProviderIdentity] = None,
):
    if not path or embedding is None:
        return
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_schema_version": SPEAKER_PROFILE_SCHEMA_VERSION,
        "source": source,
        "updated_from": source,
        "embedding_provider": provider.to_payload() if provider else {},
        "embedding_dimension": int(embedding.shape[0]) if embedding.ndim == 1 else int(embedding.size),
        "normalization": "l2",
        "embedding": embedding.tolist(),
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def average_embeddings(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    return speaker_average_embeddings(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return speaker_cosine_similarity(a, b)


def merge_profile(existing: Optional[np.ndarray], new_embedding: np.ndarray) -> np.ndarray:
    return speaker_merge_profile(existing, new_embedding)


def transcribe_audio(
    model: WhisperModel,
    audio_path: str,
    language: str,
    beam_size: int,
    batch_size: int,
    initial_prompt: Optional[str],
    hotwords: Optional[str],
) -> Tuple[List[SegmentItem], Dict[str, object]]:
    """Run faster-whisper and normalize its generator output into segment models."""

    transcribe_kwargs = {
        "language": language,
        "beam_size": beam_size,
        "vad_filter": True,
        "word_timestamps": True,
        "condition_on_previous_text": True,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
    }

    transcribe_signature = inspect.signature(model.transcribe)
    if "batch_size" in transcribe_signature.parameters:
        transcribe_kwargs["batch_size"] = batch_size

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    results = []
    total_duration = getattr(info, "duration", None)
    progress_total = float(total_duration) if total_duration and total_duration > 0 else None

    progress = create_stage_progress()
    progress.start()
    task_id = progress.add_task("transcription", total=progress_total)
    try:
        for idx, segment in enumerate(segments):
            if progress_total is not None:
                progress.update(task_id, completed=min(float(segment.end), progress_total))
            else:
                progress.refresh()

            words = []
            if segment.words:
                for word in segment.words:
                    words.append(
                        WordItem(
                            start=getattr(word, "start", None),
                            end=getattr(word, "end", None),
                            word=getattr(word, "word", ""),
                            speaker=None,
                        )
                    )

            results.append(
                SegmentItem(
                    id=idx,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    speaker=None,
                    avg_logprob=getattr(segment, "avg_logprob", None),
                    no_speech_prob=getattr(segment, "no_speech_prob", None),
                    words=words,
                )
            )
    finally:
        if progress_total is not None:
            progress.update(task_id, completed=progress_total)
        progress.stop()

    info_payload = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
    }
    return results, info_payload


def word_to_payload(word: WordItem) -> Dict[str, object]:
    return asdict(word)


def segment_to_payload(segment: SegmentItem) -> Dict[str, object]:
    payload = {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker": segment.speaker,
        "avg_logprob": segment.avg_logprob,
        "no_speech_prob": segment.no_speech_prob,
        "words": [word_to_payload(word) for word in segment.words],
    }
    for optional_attr in (
        "original_text",
        "cleanup_applied",
        "cleanup_level",
        "manual_correction_applied",
        "original_speaker",
    ):
        if hasattr(segment, optional_attr):
            payload[optional_attr] = getattr(segment, optional_attr)
    return payload


def segment_from_payload(payload: Dict[str, object]) -> SegmentItem:
    segment = SegmentItem(
        id=int(payload["id"]),
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload.get("text", "")),
        speaker=payload.get("speaker"),
        avg_logprob=payload.get("avg_logprob"),
        no_speech_prob=payload.get("no_speech_prob"),
        words=[
            WordItem(
                start=word.get("start"),
                end=word.get("end"),
                word=str(word.get("word", "")),
                speaker=word.get("speaker"),
            )
            for word in payload.get("words", [])
            if isinstance(word, dict)
        ],
    )
    for optional_attr in (
        "original_text",
        "cleanup_applied",
        "cleanup_level",
        "manual_correction_applied",
        "original_speaker",
    ):
        if optional_attr in payload:
            setattr(segment, optional_attr, payload[optional_attr])
    return segment


def save_transcription_artifact(
    output_dir: Path,
    audio_path: Path,
    segments: List[SegmentItem],
    info_payload: Dict[str, object],
    stage_fingerprint: Optional[Dict[str, object]] = None,
):
    state_save_stage_artifact(
        output_dir,
        audio_path,
        "transcription",
        {
            "segments": [segment_to_payload(segment) for segment in segments],
            "info_payload": info_payload,
        },
        stage_fingerprint=stage_fingerprint,
    )


def load_transcription_artifact(
    output_dir: Path,
    audio_path: Path,
    stage_fingerprint: Optional[Dict[str, object]] = None,
    allow_legacy: bool = True,
) -> Optional[Tuple[List[SegmentItem], Dict[str, object]]]:
    payload = state_load_stage_artifact(
        output_dir,
        audio_path,
        "transcription",
        expected_stage_fingerprint=stage_fingerprint,
        allow_legacy=allow_legacy,
    )
    if not payload:
        return None
    raw_segments = payload.get("segments")
    info_payload = payload.get("info_payload")
    if not isinstance(raw_segments, list) or not isinstance(info_payload, dict):
        return None
    return [segment_from_payload(segment) for segment in raw_segments if isinstance(segment, dict)], info_payload


def save_diarization_artifact(
    output_dir: Path,
    audio_path: Path,
    diarized_turns: List[Dict[str, object]],
    stage_fingerprint: Optional[Dict[str, object]] = None,
):
    state_save_stage_artifact(
        output_dir,
        audio_path,
        "diarization",
        {"diarized_turns": diarized_turns},
        stage_fingerprint=stage_fingerprint,
    )


def load_diarization_artifact(
    output_dir: Path,
    audio_path: Path,
    stage_fingerprint: Optional[Dict[str, object]] = None,
    allow_legacy: bool = True,
) -> Optional[List[Dict[str, object]]]:
    payload = state_load_stage_artifact(
        output_dir,
        audio_path,
        "diarization",
        expected_stage_fingerprint=stage_fingerprint,
        allow_legacy=allow_legacy,
    )
    if not payload or not isinstance(payload.get("diarized_turns"), list):
        return None
    return [turn for turn in payload["diarized_turns"] if isinstance(turn, dict)]


def run_transcription_stage(
    output_dir: Path,
    audio_path: Path,
    asr_provider,
    language: str,
    beam_size: int,
    batch_size: int,
    initial_prompt: Optional[str],
    hotwords: Optional[str],
    resume_intermediates: bool,
) -> Tuple[List[SegmentItem], Dict[str, object], bool]:
    stage_fingerprint = build_stage_fingerprint(
        "transcription",
        asr_provider.identity,
        {
            "language": language,
            "beam_size": beam_size,
            "batch_size": batch_size,
            "initial_prompt": initial_prompt or "",
            "hotwords": hotwords or "",
        },
    )
    if resume_intermediates:
        cached = load_transcription_artifact(
            output_dir,
            audio_path,
            stage_fingerprint,
            allow_legacy=(
                asr_provider.identity.provider == "faster_whisper"
                and asr_provider.identity.model == "distil-large-v3"
            ),
        )
        if cached:
            print("  stage: transcription (reused cached artifact)")
            return cached[0], cached[1], True

    print("  stage: transcription")
    result = asr_provider.transcribe(
        audio_path=str(audio_path),
        language=language,
        beam_size=beam_size,
        batch_size=batch_size,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )
    segments = result.value
    info_payload = {**result.metadata, "provider": result.provider.to_payload(), "stage_fingerprint": stage_fingerprint}
    save_transcription_artifact(output_dir, audio_path, segments, info_payload, stage_fingerprint)
    return segments, info_payload, False


def save_alignment_artifact(
    output_dir: Path,
    audio_path: Path,
    segments: List[SegmentItem],
    metadata: Dict[str, object],
    stage_fingerprint: Dict[str, object],
    dependencies: List[Dict[str, object]],
):
    state_save_stage_artifact(
        output_dir,
        audio_path,
        "alignment",
        {"segments": [segment_to_payload(segment) for segment in segments], "metadata": metadata},
        stage_fingerprint=stage_fingerprint,
        dependencies=dependencies,
    )


def load_alignment_artifact(
    output_dir: Path,
    audio_path: Path,
    stage_fingerprint: Dict[str, object],
) -> Optional[Tuple[List[SegmentItem], Dict[str, object]]]:
    payload = state_load_stage_artifact(
        output_dir,
        audio_path,
        "alignment",
        expected_stage_fingerprint=stage_fingerprint,
        allow_legacy=False,
    )
    if not payload or not isinstance(payload.get("segments"), list):
        return None
    return (
        [segment_from_payload(item) for item in payload["segments"] if isinstance(item, dict)],
        payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    )


def load_segment_stage_artifact(
    output_dir: Path,
    audio_path: Path,
    stage: str,
    stage_fingerprint: Dict[str, object],
) -> Optional[Dict[str, object]]:
    payload = state_load_stage_artifact(
        output_dir,
        audio_path,
        stage,
        expected_stage_fingerprint=stage_fingerprint,
        allow_legacy=False,
    )
    if not payload or not isinstance(payload.get("segments"), list):
        return None
    return {
        **payload,
        "segments": [segment_from_payload(item) for item in payload["segments"] if isinstance(item, dict)],
    }


def save_segment_stage_artifact(
    output_dir: Path,
    audio_path: Path,
    stage: str,
    segments: List[SegmentItem],
    metadata: Dict[str, object],
    stage_fingerprint: Dict[str, object],
    dependencies: List[Dict[str, object]],
):
    state_save_stage_artifact(
        output_dir,
        audio_path,
        stage,
        {"segments": [segment_to_payload(item) for item in segments], "metadata": metadata},
        stage_fingerprint=stage_fingerprint,
        dependencies=dependencies,
    )


def run_alignment_stage(
    output_dir: Path,
    audio_path: Path,
    segments: List[SegmentItem],
    alignment_provider,
    language: str,
    transcription_fingerprint: Dict[str, object],
    resume_intermediates: bool,
) -> Tuple[List[SegmentItem], Dict[str, object], bool, Dict[str, object]]:
    stage_fingerprint = build_stage_fingerprint(
        "alignment",
        alignment_provider.identity,
        {"language": language},
        [transcription_fingerprint],
    )
    if resume_intermediates:
        cached = load_alignment_artifact(output_dir, audio_path, stage_fingerprint)
        if cached:
            print("  stage: alignment (reused cached artifact)")
            return cached[0], cached[1], True, stage_fingerprint
    print(f"  stage: alignment ({alignment_provider.identity.provider})")
    result = alignment_provider.align(str(audio_path), segments, language)
    metadata = {**result.metadata, "provider": result.provider.to_payload()}
    save_alignment_artifact(
        output_dir,
        audio_path,
        result.value,
        metadata,
        stage_fingerprint,
        [transcription_fingerprint],
    )
    return result.value, metadata, False, stage_fingerprint


def run_diarization_stage(
    output_dir: Path,
    audio_path: Path,
    diarization_pipeline: Pipeline,
    diarization_model_id: str,
    diarization_model_revision: str,
    verifier: Any,
    num_speakers: Optional[int],
    max_embedding_seconds: float,
    resume_intermediates: bool,
    speaker_telemetry: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], bool, Dict[str, object]]:
    diarization_identity = pyannote_provider_identity(diarization_model_id, diarization_model_revision)
    stage_fingerprint = build_stage_fingerprint(
        "diarization",
        diarization_identity,
        {"num_speakers": num_speakers, "chunk_overlap_seconds": DIARIZATION_CHUNK_OVERLAP_SECONDS},
    )
    if resume_intermediates:
        cached = load_diarization_artifact(
            output_dir,
            audio_path,
            stage_fingerprint,
            allow_legacy=(diarization_model_id == "pyannote/speaker-diarization-community-1"),
        )
        if cached is not None:
            print("  stage: diarization (reused cached artifact)")
            metadata = {
                "mode": "global",
                "probe": False,
                "learned_route": False,
                "reason": "reused_cached_artifact",
                "failure_floor_seconds": 0.0,
                "safe_success_ceiling_seconds": 0.0,
                "chunk_count": 0,
                "chunk_overlap_seconds": 0.0,
                "reconciliation_merge_count": 0,
                "reconciliation_ambiguous_count": 0,
                "provider": diarization_identity.to_payload(),
                "stage_fingerprint": stage_fingerprint,
            }
            return cached, True, metadata

    print("  stage: diarization")
    diarized_turns, metadata = diarize_audio(
        output_dir=output_dir,
        pipeline=diarization_pipeline,
        diarization_model_id=diarization_model_id,
        verifier=verifier,
        audio_path=str(audio_path),
        num_speakers=num_speakers,
        max_embedding_seconds=max_embedding_seconds,
        telemetry=speaker_telemetry,
    )
    metadata = {**metadata, "provider": diarization_identity.to_payload(), "stage_fingerprint": stage_fingerprint}
    save_diarization_artifact(output_dir, audio_path, diarized_turns, stage_fingerprint)
    return diarized_turns, False, metadata


def pyannote_path_input_available() -> bool:
    try:
        import pyannote.audio.core.io as pyannote_io
    except Exception:
        return False

    return bool(getattr(pyannote_io, "TORCHCODEC_AVAILABLE", False)) and callable(
        getattr(pyannote_io, "AudioDecoder", None)
    )


MP3_SAMPLE_DRIFT_TOLERANCE_SECONDS = 0.030


def _audio_file_path(file: object) -> Optional[Path]:
    if isinstance(file, (str, Path)):
        return Path(file)
    if isinstance(file, Mapping):
        audio = file.get("audio")
        if isinstance(audio, (str, Path)):
            return Path(audio)
    return None


def _is_mp3_audio_file(file: object) -> bool:
    path = _audio_file_path(file)
    return path is not None and path.suffix.lower() == ".mp3"


def _recover_mp3_crop_with_sample_tolerance(audio: object, file: object, segment: object, mode: str, original_error: ValueError):
    """Retry a TorchCodec MP3 crop when only a codec-frame-sized drift occurred.

    pyannote.audio currently rejects anything beyond a one-sample discrepancy.
    MP3 range decoding can be short by part of a frame, even when the requested
    time range is valid.  We normalize only small end-of-range differences and
    preserve the original exception for larger or unrelated failures.
    """
    message = str(original_error)
    if not re.search(
        r"requested chunk .* resulted in \d+ samples instead of the expected \d+ samples",
        message,
    ) or not _is_mp3_audio_file(file):
        raise original_error

    import pyannote.audio.core.io as pyannote_io

    audio_cls = type(audio)
    validated_file = audio_cls.validate_file(file)
    if "waveform" in validated_file:
        raise original_error

    decoder_cls = getattr(pyannote_io, "AudioDecoder", None)
    if not callable(decoder_cls):
        raise original_error

    channel = validated_file.get("channel", None)
    decoder = decoder_cls(validated_file["audio"])
    metadata = decoder.metadata
    sample_rate = metadata.sample_rate
    duration = metadata.duration_seconds_from_header
    start = float(segment.start)
    end = float(segment.end)
    get_num_samples = audio.get_num_samples

    pad_start = max(0, get_num_samples(-start, sample_rate))
    if start < 0:
        if mode == "raise":
            raise original_error
        start = 0.0

    num_samples = get_num_samples(duration, sample_rate)
    pad_end = max(get_num_samples(end, sample_rate), num_samples) - num_samples
    if end > duration:
        if mode == "raise":
            raise original_error
        end = duration

    samples = decoder.get_samples_played_in_range(start, end)
    data = samples.data
    sample_rate = samples.sample_rate
    actual_num_samples = data.shape[1]
    expected_num_samples = get_num_samples(segment.duration, sample_rate)
    difference = pad_start + actual_num_samples + pad_end - expected_num_samples
    tolerance = max(1, round(sample_rate * MP3_SAMPLE_DRIFT_TOLERANCE_SECONDS))
    if abs(difference) > tolerance:
        raise original_error

    if difference > 0:
        data = data[:, :-difference]
    elif difference < 0:
        data = torch.nn.functional.pad(data, (0, -difference))

    return audio.downmix_and_resample(data, sample_rate, channel=channel)


@contextmanager
def tolerant_mp3_path_input():
    """Temporarily allow bounded MP3 range-decoder drift in pyannote."""
    try:
        import pyannote.audio.core.io as pyannote_io

        audio_cls = getattr(pyannote_io, "Audio", None)
        original_crop = getattr(audio_cls, "crop", None)
    except Exception:
        yield
        return

    if audio_cls is None or not callable(original_crop):
        yield
        return

    def tolerant_crop(audio, file, segment, mode="raise"):
        try:
            return original_crop(audio, file, segment, mode=mode)
        except ValueError as exc:
            return _recover_mp3_crop_with_sample_tolerance(audio, file, segment, mode, exc)

    audio_cls.crop = tolerant_crop
    try:
        yield
    finally:
        audio_cls.crop = original_crop


def diarization_kwargs(num_speakers: Optional[int]) -> Dict[str, object]:
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    return kwargs


def _call_global_diarization(
    pipeline: Pipeline,
    audio_path: str,
    kwargs: Dict[str, object],
) -> Tuple[List[Dict[str, object]], str]:
    """Run pyannote diarization and return plain speaker-turn dictionaries plus input mode."""

    if pyannote_path_input_available():
        with tolerant_mp3_path_input():
            try:
                with ProgressHook() as hook:
                    diarization = pipeline(audio_path, hook=hook, **kwargs)
            except Exception as path_exc:
                print(
                    "  diarization path input failed unexpectedly; falling back to preloaded audio. "
                    f"Path-input error: {path_exc}"
                )
            else:
                return diarization_to_turns(diarization), "path_input"

    waveform, sample_rate = torchaudio.load(audio_path)
    diarization_input = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }
    with ProgressHook() as hook:
        diarization = pipeline(diarization_input, hook=hook, **kwargs)

    return diarization_to_turns(diarization), "waveform_preload"


def _is_diarization_memory_error(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, MemoryError) or "unable to allocate array data" in message.lower()


def plan_diarization_chunks(duration_seconds: float) -> List[Dict[str, float]]:
    chunk_seconds = DIARIZATION_CHUNK_MINUTES * 60.0
    overlap_seconds = DIARIZATION_CHUNK_OVERLAP_SECONDS
    if duration_seconds <= chunk_seconds:
        return [
            {
                "index": 1,
                "start": 0.0,
                "end": float(duration_seconds),
                "core_start": 0.0,
                "core_end": float(duration_seconds),
            }
        ]
    chunks = []
    step = max(60.0, chunk_seconds - overlap_seconds)
    start = 0.0
    index = 1
    while start < duration_seconds:
        end = min(duration_seconds, start + chunk_seconds)
        chunks.append({"index": index, "start": start, "end": end})
        if end >= duration_seconds:
            break
        start += step
        index += 1
    for idx, chunk in enumerate(chunks):
        prev_end = chunks[idx - 1]["end"] if idx > 0 else chunk["start"]
        next_start = chunks[idx + 1]["start"] if idx + 1 < len(chunks) else chunk["end"]
        chunk["core_start"] = chunk["start"] if idx == 0 else max(chunk["start"], (chunk["start"] + prev_end) / 2.0)
        chunk["core_end"] = chunk["end"] if idx == len(chunks) - 1 else min(chunk["end"], (chunk["end"] + next_start) / 2.0)
    return chunks


def _offset_and_prefix_turns(turns: List[Dict[str, object]], chunk: Dict[str, float]) -> List[Dict[str, object]]:
    result = []
    chunk_id = int(chunk["index"])
    for turn in turns:
        result.append(
            {
                "start": float(turn["start"]) + float(chunk["start"]),
                "end": float(turn["end"]) + float(chunk["start"]),
                "speaker": f"chunk{chunk_id:03d}:{turn['speaker']}",
                "local_speaker": str(turn["speaker"]),
                "chunk_index": chunk_id,
            }
        )
    return result


def turns_in_window(diarized_turns: List[Dict[str, object]], start_seconds: float, end_seconds: float) -> List[Dict[str, object]]:
    return [
        turn
        for turn in diarized_turns
        if overlap_seconds(float(turn["start"]), float(turn["end"]), start_seconds, end_seconds) > 0.0
    ]


def build_chunk_speaker_embeddings(
    verifier: Any,
    audio_path: str,
    diarized_turns: List[Dict[str, object]],
    max_seconds: float,
    telemetry: Optional[Dict[str, object]] = None,
) -> Dict[str, np.ndarray]:
    speaker_audio = build_speaker_audio_samples(
        audio_path,
        diarized_turns,
        max_seconds,
        telemetry=telemetry,
        telemetry_operation="chunk_reconciliation",
    )
    embeddings = {}
    for speaker, clip in speaker_audio.items():
        if clip.numel() == 0:
            continue
        clip_seconds = float(clip.shape[0]) / 16000.0
        if clip_seconds < DIARIZATION_MIN_EMBEDDING_SECONDS:
            continue
        try:
            embeddings[speaker] = compute_embedding(
                verifier,
                clip,
                telemetry=telemetry,
                telemetry_kind="chunk_reconciliation",
            )
        except RuntimeError as exc:
            if "Padding size should be less than the corresponding input dimension" in str(exc):
                continue
            raise
    speaker_audio.clear()
    gc.collect()
    return embeddings


def reconcile_chunk_speakers(
    verifier: Any,
    audio_path: str,
    chunk_payloads: List[Dict[str, object]],
    max_embedding_seconds: float,
    telemetry: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    parent: Dict[str, str] = {}
    ambiguous_count = 0
    merge_count = 0

    def find(item: str) -> str:
        root = parent.setdefault(item, item)
        if root != item:
            parent[item] = find(root)
        return parent[item]

    def union(a: str, b: str):
        nonlocal merge_count
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        chosen = min(root_a, root_b)
        other = root_b if chosen == root_a else root_a
        parent[other] = chosen
        merge_count += 1

    for payload in chunk_payloads:
        for turn in payload["turns"]:
            parent.setdefault(str(turn["speaker"]), str(turn["speaker"]))

    # Anonymous meeting mode deliberately has no speaker-embedding verifier.
    # Keep each diarization label distinct and deterministic across chunks;
    # identity reconciliation would otherwise invoke the expensive embedding
    # path that this profile is designed to avoid.
    if verifier is None:
        root_to_global = {speaker: f"SPEAKER_{index:02d}" for index, speaker in enumerate(sorted(parent))}
        return (
            {speaker: root_to_global[speaker] for speaker in parent},
            {
                "reconciliation_merge_count": 0,
                "reconciliation_ambiguous_count": 0,
                "reconciliation_skipped": True,
            },
        )

    for idx in range(len(chunk_payloads) - 1):
        left = chunk_payloads[idx]
        right = chunk_payloads[idx + 1]
        overlap_start = max(float(left["chunk"]["start"]), float(right["chunk"]["start"]))
        overlap_end = min(float(left["chunk"]["end"]), float(right["chunk"]["end"]))
        if overlap_end <= overlap_start:
            continue
        left_overlap = turns_in_window(left["turns"], overlap_start, overlap_end)
        right_overlap = turns_in_window(right["turns"], overlap_start, overlap_end)
        if not left_overlap or not right_overlap:
            continue
        if telemetry is not None:
            telemetry["chunk_reconciliation_boundary_count"] = int(
                telemetry.get("chunk_reconciliation_boundary_count") or 0
            ) + 1
        left_embeddings = build_chunk_speaker_embeddings(
            verifier,
            audio_path,
            left_overlap,
            min(max_embedding_seconds, 45.0),
            telemetry=telemetry,
        )
        right_embeddings = build_chunk_speaker_embeddings(
            verifier,
            audio_path,
            right_overlap,
            min(max_embedding_seconds, 45.0),
            telemetry=telemetry,
        )
        if not left_embeddings or not right_embeddings:
            continue
        score_pairs: List[Tuple[float, str, str]] = []
        for right_speaker, right_embedding in right_embeddings.items():
            speaker_scores = []
            for left_speaker, left_embedding in left_embeddings.items():
                score = cosine_similarity(left_embedding, right_embedding)
                speaker_scores.append((score, left_speaker))
            speaker_scores.sort(reverse=True)
            if not speaker_scores:
                continue
            if len(speaker_scores) > 1 and (speaker_scores[0][0] - speaker_scores[1][0]) < 0.03:
                ambiguous_count += 1
            best_score, best_left = speaker_scores[0]
            if best_score >= DIARIZATION_RECONCILIATION_SIMILARITY:
                score_pairs.append((best_score, best_left, right_speaker))
        assigned_left = set()
        assigned_right = set()
        for _score, left_speaker, right_speaker in sorted(score_pairs, reverse=True):
            if left_speaker in assigned_left or right_speaker in assigned_right:
                continue
            union(left_speaker, right_speaker)
            assigned_left.add(left_speaker)
            assigned_right.add(right_speaker)

    root_to_global: Dict[str, str] = {}
    next_index = 0
    for speaker in sorted(parent.keys()):
        root = find(speaker)
        if root not in root_to_global:
            root_to_global[root] = f"SPEAKER_{next_index:02d}"
            next_index += 1
    return (
        {speaker: root_to_global[find(speaker)] for speaker in parent.keys()},
        {
            "reconciliation_merge_count": merge_count,
            "reconciliation_ambiguous_count": ambiguous_count,
        },
    )


def stitch_chunk_turns(
    chunk_payloads: List[Dict[str, object]],
    speaker_mapping: Dict[str, str],
) -> List[Dict[str, object]]:
    stitched: List[Dict[str, object]] = []
    for payload in chunk_payloads:
        chunk = payload["chunk"]
        for turn in payload["turns"]:
            clipped_start = max(float(turn["start"]), float(chunk["core_start"]))
            clipped_end = min(float(turn["end"]), float(chunk["core_end"]))
            if clipped_end <= clipped_start:
                continue
            stitched.append(
                {
                    "start": clipped_start,
                    "end": clipped_end,
                    "speaker": speaker_mapping.get(str(turn["speaker"]), str(turn["speaker"])),
                }
            )
    stitched.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    merged: List[Dict[str, object]] = []
    for turn in stitched:
        if (
            merged
            and merged[-1]["speaker"] == turn["speaker"]
            and abs(float(merged[-1]["end"]) - float(turn["start"])) <= 0.25
        ):
            merged[-1]["end"] = max(float(merged[-1]["end"]), float(turn["end"]))
        else:
            merged.append(dict(turn))
    return merged


def diarize_audio_chunked(
    pipeline: Pipeline,
    verifier: Any,
    audio_path: str,
    num_speakers: Optional[int],
    duration_seconds: float,
    max_embedding_seconds: float,
    telemetry: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    kwargs = diarization_kwargs(num_speakers)
    chunks = plan_diarization_chunks(duration_seconds)
    print(f"  diarization mode: chunked fallback ({len(chunks)} chunk(s), overlap {int(DIARIZATION_CHUNK_OVERLAP_SECONDS)}s)")
    chunk_payloads: List[Dict[str, object]] = []
    sample_rate, _, _ = get_audio_metadata(audio_path)
    if sample_rate is None or sample_rate <= 0:
        sample_rate = 16000
    resampler = torchaudio.transforms.Resample(sample_rate, 16000) if sample_rate != 16000 else None
    for chunk in chunks:
        print(f"    diarization chunk {int(chunk['index'])}/{len(chunks)}")
        waveform = load_audio_span_mono_16k(
            audio_path,
            float(chunk["start"]),
            float(chunk["end"]),
            sample_rate=sample_rate,
            resampler=resampler,
        )
        diarization_input = {
            "waveform": waveform.unsqueeze(0) if waveform.ndim == 1 else waveform,
            "sample_rate": 16000,
        }
        with ProgressHook(hidden=False) as hook:
            diarization = pipeline(diarization_input, hook=hook, **kwargs)
        turns = _offset_and_prefix_turns(diarization_to_turns(diarization), chunk)
        chunk_payloads.append({"chunk": chunk, "turns": turns})
    print("    reconciling chunk speakers")
    local_to_global, reconciliation_stats = reconcile_chunk_speakers(
        verifier,
        audio_path,
        chunk_payloads,
        max_embedding_seconds,
        telemetry=telemetry,
    )
    final_turns = stitch_chunk_turns(chunk_payloads, local_to_global)
    metadata = {
        "mode": "chunked_fallback_after_failure",
        "probe": False,
        "learned_route": False,
        "reason": "global_memory_error",
        "failure_floor_seconds": 0.0,
        "safe_success_ceiling_seconds": 0.0,
        "chunk_count": len(chunks),
        "chunk_overlap_seconds": float(DIARIZATION_CHUNK_OVERLAP_SECONDS),
        **reconciliation_stats,
    }
    return final_turns, metadata


def diarize_audio(
    output_dir: Path,
    pipeline: Pipeline,
    diarization_model_id: str,
    verifier: Any,
    audio_path: str,
    num_speakers: Optional[int],
    max_embedding_seconds: float,
    telemetry: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Run pyannote diarization and return plain speaker-turn dictionaries plus route metadata."""
    duration_seconds = get_audio_duration_seconds(audio_path)
    expected_input_mode = "path_input" if pyannote_path_input_available() else "waveform_preload"
    runtime_fingerprint = diarization_runtime_fingerprint(diarization_model_id, expected_input_mode)
    history_state = load_diarization_history(output_dir)
    route = diarization_route_decision(duration_seconds, history_state, runtime_fingerprint)
    if route["mode"] == "chunked_preemptive" and duration_seconds:
        print(
            "  diarization mode: preemptive chunked "
            f"(failure_floor={format_timestamp(route['failure_floor_seconds'])}, "
            f"success_ceiling={format_timestamp(route['safe_success_ceiling_seconds'])}, "
            f"reason={route['reason']})"
        )
        turns, metadata = diarize_audio_chunked(
            pipeline,
            verifier,
            audio_path,
            num_speakers,
            duration_seconds,
            max_embedding_seconds,
            telemetry=telemetry,
        )
        metadata.update(route)
        metadata["mode"] = "chunked_preemptive"
        update_diarization_history(
            output_dir,
            runtime_fingerprint,
            Path(audio_path),
            float(duration_seconds or 0.0),
            "chunked_preemptive",
            "success",
            probe=False,
        )
        return turns, metadata

    if route.get("probe"):
        print(
            "  diarization mode: global diarization probe "
            f"(failure_floor={format_timestamp(route['failure_floor_seconds'])}, "
            f"success_ceiling={format_timestamp(route['safe_success_ceiling_seconds'])})"
        )
    else:
        print("  diarization mode: global")

    kwargs = diarization_kwargs(num_speakers)
    try:
        turns, actual_input_mode = _call_global_diarization(pipeline, audio_path, kwargs)
    except Exception as exc:
        if _is_diarization_memory_error(exc) and duration_seconds:
            print("  global diarization hit MemoryError; retrying with chunked fallback.")
            actual_fingerprint = diarization_runtime_fingerprint(diarization_model_id, expected_input_mode)
            update_diarization_history(
                output_dir,
                actual_fingerprint,
                Path(audio_path),
                float(duration_seconds or 0.0),
                "global",
                "memory_error",
                probe=bool(route.get("probe")),
            )
            turns, metadata = diarize_audio_chunked(
                pipeline,
                verifier,
                audio_path,
                num_speakers,
                duration_seconds,
                max_embedding_seconds,
                telemetry=telemetry,
            )
            metadata.update(route)
            metadata["mode"] = "chunked_fallback_after_failure"
            metadata["reason"] = "global_memory_error"
            metadata["probe"] = bool(route.get("probe"))
            update_diarization_history(
                output_dir,
                actual_fingerprint,
                Path(audio_path),
                float(duration_seconds or 0.0),
                "chunked_fallback_after_failure",
                "success",
                probe=False,
            )
            return turns, metadata
        raise
    if duration_seconds:
        actual_fingerprint = diarization_runtime_fingerprint(diarization_model_id, actual_input_mode)
        update_diarization_history(
            output_dir,
            actual_fingerprint,
            Path(audio_path),
            float(duration_seconds or 0.0),
            "global",
            "success",
            probe=bool(route.get("probe")),
        )
    route["mode"] = "global"
    route["chunk_count"] = 0
    route["chunk_overlap_seconds"] = 0.0
    route["reconciliation_merge_count"] = 0
    route["reconciliation_ambiguous_count"] = 0
    return turns, route


def diarization_to_turns(diarization) -> List[Dict[str, object]]:
    diarization_annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )
    turns = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return turns


def parse_version_major(version_text: str) -> int:
    match = re.match(r"^(\d+)", version_text or "")
    return int(match.group(1)) if match else 0


def resolve_compatible_diarization_model(model_id: str) -> Tuple[str, Optional[str]]:
    pyannote_version = getattr(pyannote_audio, "__version__", "")
    pyannote_major = parse_version_major(pyannote_version)

    if model_id == "pyannote/speaker-diarization-community-1" and pyannote_major and pyannote_major < 4:
        return (
            "pyannote/speaker-diarization-3.1",
            (
                f"pyannote.audio {pyannote_version} is installed, so switching diarization model from "
                "'pyannote/speaker-diarization-community-1' to the compatible legacy pipeline "
                "'pyannote/speaker-diarization-3.1'."
            ),
        )

    return model_id, None


def load_diarization_pipeline(model_id: str, hf_token: str) -> Tuple[Pipeline, str]:
    resolved_model_id, compatibility_note = resolve_compatible_diarization_model(model_id)
    if compatibility_note:
        print(compatibility_note)

    signature = inspect.signature(Pipeline.from_pretrained)
    parameters = signature.parameters

    if "token" in parameters:
        return Pipeline.from_pretrained(resolved_model_id, token=hf_token), resolved_model_id

    if "use_auth_token" in parameters:
        return Pipeline.from_pretrained(resolved_model_id, use_auth_token=hf_token), resolved_model_id

    raise RuntimeError(
        "Unsupported pyannote.audio installation: Pipeline.from_pretrained accepts neither "
        "'token' nor 'use_auth_token'."
    )


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers_to_segments(segments: List[SegmentItem], diarized_turns: List[Dict[str, object]]):
    """Assign best-overlap diarized speakers to transcript segments and words."""

    # Both lists are time ordered in normal ASR output. The old implementation
    # compared every segment and every word with every diarization turn, making
    # a cached 4,000-segment / 2,000-turn episode spend tens of seconds doing
    # work that did not invoke the diarization model at all.
    ordered_turns = sorted(
        diarized_turns,
        key=lambda turn: (float(turn.get("start") or 0.0), float(turn.get("end") or 0.0)),
    )

    def best_speaker(start: float, end: float, cursor: int, previous_start: float) -> Tuple[Optional[str], int, float]:
        if start < previous_start:
            cursor = 0
        while cursor < len(ordered_turns) and float(ordered_turns[cursor]["end"]) <= start:
            cursor += 1
        overlap_by_speaker = defaultdict(float)
        index = cursor
        while index < len(ordered_turns) and float(ordered_turns[index]["start"]) < end:
            turn = ordered_turns[index]
            overlap = overlap_seconds(start, end, float(turn["start"]), float(turn["end"]))
            if overlap > 0:
                overlap_by_speaker[turn["speaker"]] += overlap
            index += 1
        if not overlap_by_speaker:
            return None, cursor, start
        return max(overlap_by_speaker.items(), key=lambda item: item[1])[0], cursor, start

    segment_cursor = 0
    segment_previous_start = float("-inf")
    word_cursor = 0
    word_previous_start = float("-inf")
    for segment in segments:
        speaker, segment_cursor, segment_previous_start = best_speaker(
            float(segment.start), float(segment.end), segment_cursor, segment_previous_start
        )
        if speaker:
            segment.speaker = speaker
        elif not segment.speaker:
            segment.speaker = "UNKNOWN"

        for word in segment.words:
            if word.start is None or word.end is None:
                word.speaker = segment.speaker
                continue

            word_speaker, word_cursor, word_previous_start = best_speaker(
                float(word.start), float(word.end), word_cursor, word_previous_start
            )
            word.speaker = word_speaker or segment.speaker
            if not word.speaker:
                word.speaker = "UNKNOWN"


def speaker_durations(diarized_turns: List[Dict[str, object]]) -> Dict[str, float]:
    totals = defaultdict(float)
    for turn in diarized_turns:
        totals[turn["speaker"]] += max(0.0, turn["end"] - turn["start"])
    return dict(totals)


def _build_speaker_audio_samples_for_turns(
    audio_path: str,
    speaker_turns: List[Dict[str, object]],
    max_seconds: float,
    telemetry: Optional[Dict[str, object]] = None,
    telemetry_operation: str = "episode_speaker_samples",
    sample_rate: Optional[int] = None,
    resampler: Optional[Any] = None,
) -> torch.Tensor:
    clips = []
    duration_seconds = 0.0
    if sample_rate is None:
        sample_rate, _, _ = get_audio_metadata(audio_path)
    if sample_rate is None or sample_rate <= 0:
        sample_rate = 16000
    if resampler is None and sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)

    for turn in speaker_turns:
        if duration_seconds >= max_seconds:
            break

        remaining = max_seconds - duration_seconds
        clipped_end = min(float(turn["end"]), float(turn["start"]) + remaining)
        if clipped_end <= float(turn["start"]):
            continue

        requested_seconds = max(0.0, clipped_end - float(turn["start"]))
        span_started = time.perf_counter()
        try:
            segment = load_audio_span_mono_16k(
                audio_path,
                start_seconds=float(turn["start"]),
                end_seconds=clipped_end,
                sample_rate=sample_rate,
                resampler=resampler,
            )
        except Exception:
            if telemetry is not None:
                telemetry["audio_span_error_count"] = int(telemetry.get("audio_span_error_count") or 0) + 1
            raise
        _record_audio_span_telemetry(
            telemetry,
            telemetry_operation,
            requested_seconds,
            float(segment.numel()) / 16000.0,
            time.perf_counter() - span_started,
        )
        if segment.numel() == 0:
            continue

        clips.append(segment)
        duration_seconds += segment.shape[0] / 16000.0

    if not clips:
        return torch.empty(0, dtype=torch.float32)
    if len(clips) == 1:
        return clips[0]
    merged = torch.cat(clips)
    clips.clear()
    return merged


def build_speaker_audio_samples(
    audio_path: str,
    diarized_turns: List[Dict[str, object]],
    max_seconds: float,
    telemetry: Optional[Dict[str, object]] = None,
    telemetry_operation: str = "episode_speaker_samples",
) -> Dict[str, torch.Tensor]:
    """Collect bounded audio samples, one speaker at a time.

    The returned mapping is retained by callers that need all clips (for
    example chunk reconciliation), but each speaker's collection is bounded by
    ``max_seconds`` and is no longer assembled from an unbounded cross-speaker
    list of native decoder results.
    """

    turns_by_speaker = defaultdict(list)
    for turn in diarized_turns:
        turns_by_speaker[turn["speaker"]].append(turn)

    sample_rate, _, _ = get_audio_metadata(audio_path)
    if sample_rate is None or sample_rate <= 0:
        sample_rate = 16000
    resampler = (
        torchaudio.transforms.Resample(sample_rate, 16000)
        if sample_rate != 16000
        else None
    )

    return {
        speaker: _build_speaker_audio_samples_for_turns(
            audio_path,
            speaker_turns,
            max_seconds,
            telemetry=telemetry,
            telemetry_operation=telemetry_operation,
            sample_rate=sample_rate,
            resampler=resampler,
        )
        for speaker, speaker_turns in turns_by_speaker.items()
    }


def compute_embedding(
    verifier: Any,
    waveform_16k: torch.Tensor,
    telemetry: Optional[Dict[str, object]] = None,
    telemetry_kind: str = "unspecified",
) -> np.ndarray:
    started = time.perf_counter()
    dispatch_seconds = 0.0
    sync_copy_seconds = 0.0
    input_seconds = float(waveform_16k.numel()) / 16000.0
    signal = waveform_16k.unsqueeze(0)
    try:
        dispatch_started = time.perf_counter()
        with torch.no_grad():
            embedding = verifier.encode_batch(signal)
        dispatch_seconds = time.perf_counter() - dispatch_started
        sync_started = time.perf_counter()
        # Tensor.cpu() is intentionally timed separately: on CUDA this is the
        # synchronization point that waits for the model and copies its result.
        vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
        sync_copy_seconds = time.perf_counter() - sync_started
    except Exception:
        _record_embedding_telemetry(
            telemetry,
            telemetry_kind,
            input_seconds,
            time.perf_counter() - started,
            dispatch_seconds,
            sync_copy_seconds,
            failed=True,
        )
        raise
    _record_embedding_telemetry(
        telemetry,
        telemetry_kind,
        input_seconds,
        time.perf_counter() - started,
        dispatch_seconds,
        sync_copy_seconds,
    )
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def load_known_speaker_profiles(
    verifier: Any,
    known_speakers_dir: Optional[str],
    telemetry: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    config_entries = load_known_speakers_config(known_speakers_dir)
    if not config_entries:
        return {}

    base_dir = Path(known_speakers_dir)
    profiles = {}

    for entry in config_entries:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("display_name") or entry.get("name") or "").strip()
        files = entry.get("files", [])
        if not name or not isinstance(files, list):
            continue

        embeddings = []
        resolved_files = []
        sample_quality = []
        for relative_path in files:
            sample_path = base_dir / str(relative_path)
            if not sample_path.exists():
                continue
            waveform = load_audio_mono_16k(str(sample_path))
            quality = audio_reference_quality(waveform)
            sample_quality.append({"file": str(sample_path), **quality})
            if quality["rating"] == "poor":
                print(f"  reference sample warning for {name}: {sample_path.name} -> {', '.join(quality['warnings'])}")
            embeddings.append(
                compute_embedding(
                    verifier,
                    waveform,
                    telemetry=telemetry,
                    telemetry_kind="known_profile",
                )
            )
            resolved_files.append(str(sample_path))

        averaged = average_embeddings(embeddings)
        if averaged is None:
            continue

        profiles[name] = {
            "name": name,
            "speaker_id": str(entry.get("speaker_id") or ""),
            "aliases": list(entry.get("aliases") or []),
            "roles": list(entry.get("roles") or (["host"] if entry.get("is_host") else ["guest"])),
            "embedding": averaged,
            "is_host": (
                bool(entry.get("is_host", False))
                or name.upper() == "HOST"
                or any(str(role).lower() in {"host", "co-host"} for role in entry.get("roles") or [])
            ),
            "sample_files": resolved_files,
            "sample_quality": sample_quality,
        }

    return profiles


def choose_host_speaker(
    verifier: Any,
    audio_path: str,
    diarized_turns: List[Dict[str, object]],
    host_reference_path: Optional[str],
    existing_profile: Optional[np.ndarray],
    host_threshold: float,
    assume_dominant: bool,
    max_embedding_seconds: float,
    min_host_seconds: float,
    telemetry: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[str], Dict[str, np.ndarray], Optional[np.ndarray], Dict[str, float], Dict[str, float]]:
    durations = speaker_durations(diarized_turns)
    if not durations:
        return None, {}, existing_profile, {}, {}

    turns_by_speaker = defaultdict(list)
    for turn in diarized_turns:
        turns_by_speaker[turn["speaker"]].append(turn)

    sample_rate, _, _ = get_audio_metadata(audio_path)
    if sample_rate is None or sample_rate <= 0:
        sample_rate = 16000
    resampler = (
        torchaudio.transforms.Resample(sample_rate, 16000)
        if sample_rate != 16000
        else None
    )

    speaker_embeddings = {}
    for speaker, speaker_turns in turns_by_speaker.items():
        # Do not retain audio for every speaker while waiting to compute the
        # embeddings.  A long episode can have enough diarized turns for that
        # old pattern to exhaust system RAM before the first embedding runs.
        clip = None
        try:
            clip = _build_speaker_audio_samples_for_turns(
                audio_path,
                speaker_turns,
                max_embedding_seconds,
                telemetry=telemetry,
                telemetry_operation="episode_speaker_samples",
                sample_rate=sample_rate,
                resampler=resampler,
            )
            if durations.get(speaker, 0.0) >= min_host_seconds and clip.numel() > 0:
                speaker_embeddings[speaker] = compute_embedding(
                    verifier,
                    clip,
                    telemetry=telemetry,
                    telemetry_kind="episode_speaker",
                )
        finally:
            if clip is not None:
                del clip
            # Release Python/native decoder temporaries before reading the next
            # speaker's spans.  The PCM WAV reader already bounds each read.
            gc.collect()

    reference_embedding = existing_profile
    if host_reference_path:
        ref_waveform = load_audio_mono_16k(host_reference_path)
        reference_embedding = compute_embedding(
            verifier,
            ref_waveform,
            telemetry=telemetry,
            telemetry_kind="host_reference",
        )
        del ref_waveform
        gc.collect()

    best_match = None
    best_score = -1.0
    similarity_scores = {}

    if reference_embedding is not None:
        for speaker, embedding in speaker_embeddings.items():
            score = cosine_similarity(reference_embedding, embedding)
            similarity_scores[speaker] = score
            if score > best_score:
                best_score = score
                best_match = speaker

        if best_match is not None and best_score >= host_threshold:
            updated_profile = merge_profile(existing_profile, speaker_embeddings[best_match])
            return best_match, speaker_embeddings, updated_profile, durations, similarity_scores

    if assume_dominant:
        dominant_speaker = max(durations.items(), key=lambda item: item[1])[0]
        updated_profile = existing_profile
        if dominant_speaker in speaker_embeddings:
            updated_profile = merge_profile(existing_profile, speaker_embeddings[dominant_speaker])
        return dominant_speaker, speaker_embeddings, updated_profile, durations, similarity_scores

    return None, speaker_embeddings, existing_profile, durations, similarity_scores


def match_known_speakers(
    speaker_embeddings: Dict[str, np.ndarray],
    known_profiles: Dict[str, Dict[str, object]],
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    assignments = {}
    candidates = []

    for diarized_speaker, diarized_embedding in speaker_embeddings.items():
        for known_name, profile in known_profiles.items():
            score = cosine_similarity(diarized_embedding, profile["embedding"])
            if score >= threshold:
                candidates.append((score, diarized_speaker, known_name))

    for score, diarized_speaker, known_name in sorted(candidates, reverse=True):
        if diarized_speaker in assignments:
            continue
        if any(match["known_name"] == known_name for match in assignments.values()):
            continue
        assignments[diarized_speaker] = {
            "known_name": known_name,
            "speaker_id": str(known_profiles[known_name].get("speaker_id") or ""),
            "roles": list(known_profiles[known_name].get("roles") or []),
            "score": score,
            "is_host": bool(known_profiles[known_name].get("is_host", False)),
        }

    return assignments


def rename_speakers(
    segments: List[SegmentItem],
    diarized_turns: List[Dict[str, object]],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    known_assignments: Optional[Dict[str, Dict[str, object]]] = None,
):
    """Choose the diarized speaker most likely to be the host and prepare profile updates."""

    ordered = sorted(durations.items(), key=lambda item: item[1], reverse=True)
    mapping = {}
    guest_index = 1
    known_assignments = known_assignments or {}
    for speaker, _ in ordered:
        if speaker in known_assignments:
            mapping[speaker] = known_assignments[speaker]["known_name"]
        elif speaker == host_speaker:
            mapping[speaker] = "HOST"
        else:
            mapping[speaker] = f"SPEAKER_{guest_index:02d}"
            guest_index += 1

    for segment in segments:
        if segment.speaker in mapping:
            segment.original_speaker = segment.speaker
            segment.speaker = mapping[segment.speaker]
        for word in segment.words:
            if word.speaker in mapping:
                word.speaker = mapping[word.speaker]

    for turn in diarized_turns:
        if turn["speaker"] in mapping:
            turn["speaker_label"] = mapping[turn["speaker"]]
        else:
            turn["speaker_label"] = turn["speaker"]

    return mapping


def coalesce_segments(
    segments: List[SegmentItem],
    replacement_map: Dict[str, List[str]],
) -> Tuple[List[SegmentItem], List[Dict[str, object]]]:
    cleaned = []
    replacement_events = []
    for segment in segments:
        replacement_hits = detect_replacement_hits(segment.text, replacement_map)
        for hit in replacement_hits:
            replacement_events.append(
                {
                    "issue_type": "glossary_replacement_candidate",
                    "speaker": segment.speaker or "UNKNOWN",
                    "start": format_timestamp(segment.start),
                    "end": format_timestamp(segment.end),
                    "score": "",
                    "details": f"Detected alias '{hit['alias']}' and normalized to '{hit['preferred']}'.",
                    "text": segment.text,
                }
            )

        segment.text = apply_replacements(segment.text, replacement_map).strip()
        if not segment.text:
            continue

        for word in segment.words:
            word.word = apply_replacements(word.word, replacement_map)

        if cleaned and cleaned[-1].speaker == segment.speaker and segment.start - cleaned[-1].end <= 0.8:
            cleaned[-1].text = (cleaned[-1].text + " " + segment.text).strip()
            cleaned[-1].end = segment.end
            cleaned[-1].words.extend(segment.words)
        else:
            cleaned.append(segment)
    return cleaned, replacement_events


def collect_review_rows(
    source_file: str,
    segments: List[SegmentItem],
    replacement_events: List[Dict[str, object]],
    host_speaker: Optional[str],
    host_threshold: float,
    durations: Dict[str, float],
    similarity_scores: Dict[str, float],
    speaker_mapping: Dict[str, str],
    host_output_labels: Optional[set[str]] = None,
    episode_metadata: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    rows = []
    host_output_labels = host_output_labels or {"HOST"}
    episode_metadata = episode_metadata or build_episode_metadata(source_file)
    review_metadata = {
        "episode_date": episode_metadata.get("episode_date", ""),
        "episode_date_compact": episode_metadata.get("episode_date_compact", ""),
        "episode_sort_key": episode_metadata.get("episode_sort_key", ""),
    }

    if host_speaker is None:
        rows.append(
            {
                "issue_type": "host_not_detected",
                "speaker": "",
                "start": "",
                "end": "",
                "score": "",
                "details": "No host speaker met the configured threshold and no fallback label was established.",
                "text": "",
                "source_file": source_file,
            }
        )

    sorted_scores = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    if sorted_scores:
        top_speaker, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else None
        margin = top_score - second_score if second_score is not None else None

        if top_score < host_threshold + 0.05:
            rows.append(
                {
                    "issue_type": "host_match_near_threshold",
                    "speaker": speaker_mapping.get(top_speaker, top_speaker),
                    "start": "",
                    "end": "",
                    "score": round(top_score, 4),
                    "details": f"Top host similarity is close to threshold {host_threshold:.2f}.",
                    "text": "",
                    "source_file": source_file,
                }
            )

        if margin is not None and margin < 0.05:
            rows.append(
                {
                    "issue_type": "host_match_ambiguous",
                    "speaker": speaker_mapping.get(top_speaker, top_speaker),
                    "start": "",
                    "end": "",
                    "score": round(top_score, 4),
                    "details": f"Top two host similarity scores are close; margin={margin:.4f}.",
                    "text": "",
                    "source_file": source_file,
                }
            )

    if host_speaker is not None and host_speaker in durations and durations[host_speaker] < 60:
        rows.append(
            {
                "issue_type": "host_low_coverage",
                "speaker": speaker_mapping.get(host_speaker, host_speaker),
                "start": "",
                "end": "",
                "score": round(durations[host_speaker], 2),
                "details": "Detected host has less than 60 seconds of diarized speech in this episode.",
                "text": "",
                "source_file": source_file,
            }
        )

    for event in replacement_events:
        rows.append({**event, "source_file": source_file})

    for segment in segments:
        if segment.speaker in host_output_labels and similarity_scores:
            top_score = max(similarity_scores.values())
            if top_score < host_threshold + 0.05:
                rows.append(
                    {
                        "issue_type": "host_segment_review",
                        "speaker": segment.speaker,
                        "start": format_timestamp(segment.start),
                        "end": format_timestamp(segment.end),
                        "score": round(top_score, 4),
                        "details": "Host label came from a weak overall speaker match; review this segment if accuracy is important.",
                        "text": segment.text,
                        "source_file": source_file,
                    }
                )

    for row in rows:
        row.update(review_metadata)

    return rows


def write_review_csv(path: Path, rows: List[Dict[str, object]]):
    fieldnames = [
        "issue_type",
        "speaker",
        "start",
        "end",
        "score",
        "details",
        "text",
        "source_file",
        "episode_date",
        "episode_date_compact",
        "episode_sort_key",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_review_rows(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        counts[row.get("issue_type", "unknown")] += 1
    return dict(counts)


def build_episode_summary_row(
    audio_path: Path,
    normalized_segments: List[SegmentItem],
    review_rows: List[Dict[str, object]],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    similarity_scores: Dict[str, float],
    speaker_mapping: Dict[str, str],
    known_assignments: Dict[str, Dict[str, object]],
    episode_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    episode_metadata = episode_metadata or build_episode_metadata(str(audio_path))
    review_counts = summarize_review_rows(review_rows)
    sorted_scores = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1] if sorted_scores else ""
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else ""
    score_margin = (sorted_scores[0][1] - sorted_scores[1][1]) if len(sorted_scores) > 1 else ""
    host_duration = durations.get(host_speaker, 0.0) if host_speaker else 0.0
    total_duration = sum(durations.values())
    host_share = (host_duration / total_duration) if total_duration else 0.0
    review_priority_score = 0.0
    priority_reasons = []

    if host_speaker is None:
        review_priority_score += 100.0
        priority_reasons.append("host not detected")

    if top_score != "":
        review_priority_score += max(0.0, (0.7 - top_score) * 100.0)
        if top_score < 0.55:
            priority_reasons.append(f"low host similarity ({top_score:.2f})")
    else:
        review_priority_score += 30.0
        priority_reasons.append("no host similarity score available")

    if score_margin != "":
        review_priority_score += max(0.0, (0.1 - score_margin) * 120.0)
        if score_margin < 0.05:
            priority_reasons.append(f"ambiguous top speaker margin ({score_margin:.2f})")
    else:
        review_priority_score += 10.0
        priority_reasons.append("only one speaker candidate scored")

    review_priority_score += max(0.0, (0.35 - host_share) * 80.0)
    if host_share < 0.35:
        priority_reasons.append(f"low host share of speech ({host_share:.0%})")

    review_priority_score += review_counts.get("host_match_near_threshold", 0) * 12.0
    review_priority_score += review_counts.get("host_match_ambiguous", 0) * 20.0
    review_priority_score += review_counts.get("host_low_coverage", 0) * 18.0
    review_priority_score += review_counts.get("host_segment_review", 0) * 1.5
    review_priority_score += review_counts.get("host_not_detected", 0) * 40.0
    review_priority_score += review_counts.get("speaker_similarity_drift", 0) * 15.0
    review_priority_score += min(review_counts.get("glossary_replacement_candidate", 0), 20) * 0.5

    if review_counts.get("host_low_coverage", 0):
        priority_reasons.append("host coverage is low")
    if review_counts.get("host_segment_review", 0):
        priority_reasons.append(f"{review_counts.get('host_segment_review', 0)} host segments need review")
    if review_counts.get("glossary_replacement_candidate", 0) >= 5:
        priority_reasons.append(f"{review_counts.get('glossary_replacement_candidate', 0)} glossary corrections applied")
    if review_counts.get("speaker_similarity_drift", 0):
        priority_reasons.append("speaker similarity drift detected")

    if not priority_reasons:
        priority_reasons.append("no major review issues detected")

    return {
        "episode": audio_path.name,
        "episode_date": episode_metadata.get("episode_date", ""),
        "episode_date_compact": episode_metadata.get("episode_date_compact", ""),
        "episode_year": episode_metadata.get("episode_year", ""),
        "episode_month": episode_metadata.get("episode_month", ""),
        "episode_day": episode_metadata.get("episode_day", ""),
        "episode_sort_key": episode_metadata.get("episode_sort_key", ""),
        "review_priority_score": round(review_priority_score, 2),
        "review_priority_reason": "; ".join(dict.fromkeys(priority_reasons)),
        "host_detected": host_speaker is not None,
        "host_label": speaker_mapping.get(host_speaker, "") if host_speaker else "",
        "known_speakers_detected": ", ".join(
            speaker_mapping[speaker_id]
            for speaker_id in sorted(known_assignments.keys(), key=lambda key: speaker_mapping.get(key, key))
        ),
        "host_duration_seconds": round(host_duration, 2),
        "host_share_of_speech": round(host_share, 4),
        "top_host_similarity": round(top_score, 4) if top_score != "" else "",
        "second_host_similarity": round(second_score, 4) if second_score != "" else "",
        "host_similarity_margin": round(score_margin, 4) if score_margin != "" else "",
        "speaker_count": len(durations),
        "transcript_segments": len(normalized_segments),
        "review_row_count": len(review_rows),
        "host_match_near_threshold_count": review_counts.get("host_match_near_threshold", 0),
        "host_match_ambiguous_count": review_counts.get("host_match_ambiguous", 0),
        "host_low_coverage_count": review_counts.get("host_low_coverage", 0),
        "host_segment_review_count": review_counts.get("host_segment_review", 0),
        "glossary_replacement_candidate_count": review_counts.get("glossary_replacement_candidate", 0),
        "host_not_detected_count": review_counts.get("host_not_detected", 0),
        "speaker_similarity_drift_count": review_counts.get("speaker_similarity_drift", 0),
        "cleanup_level": "",
        "cleanup_edit_count": "",
        "manual_correction_count": "",
        "processing_seconds": "",
        "language_model_warnings": "",
        "transcription_artifact_reused": "",
        "alignment_artifact_reused": "",
        "diarization_artifact_reused": "",
        "asr_provider": "",
        "alignment_provider": "",
        "speaker_embedding_provider": "",
        "review_attempted": False,
        "review_status": "",
        "review_skip_reason": "",
        "review_runtime_profile": "",
        "review_backend": "",
        "review_model_name": "",
        "reviewed_segment_count": 0,
        "review_corrected_segment_count": 0,
        "review_candidate_count": 0,
        "review_context_segment_count": 0,
        "review_skipped_segment_count": 0,
        "reviewed_output_written": False,
        "review_pipeline_version": "",
        "review_enabled_stages": "",
        "review_completed_stages": "",
        "review_skipped_stages": "",
        "review_input_source": "",
        "review_episode_qa_mode": "",
        "review_calibration_source": "",
        "review_local_text_budget": "",
        "review_local_speaker_budget": "",
        "review_long_context_budget": "",
        "cleanup_review_corrected_count": 0,
        "glossary_review_corrected_count": 0,
        "speaker_consistency_review_corrected_count": 0,
        "episode_qa_review_corrected_count": 0,
        "review_material_change": False,
        "review_unique_stage_count": 0,
        "review_noop_stage_count": 0,
        "review_returned_change_count": 0,
        "review_applied_change_count": 0,
        "review_overridden_change_count": 0,
        "episode_qa_added_value": False,
        "preferred_term_intervention_count": 0,
        "review_guard_intervention_count": 0,
        "speaker_drift_flag": False,
        "recurring_unnamed_speaker_flag": False,
        "host_profile_stability_flag": False,
        "processing_mode": "",
        "episode_contract_version": "",
        "contract_upgrade_method": "",
        "contract_upgrade_archive_path": "",
        "tier1_reused_from_existing": False,
        "review_backfilled_from_cleaned_json": False,
        "diarization_mode": "",
        "diarization_probe_attempted": False,
        "diarization_learned_route": False,
        "diarization_route_reason": "",
        "diarization_failure_floor_seconds": 0,
        "diarization_safe_success_ceiling_seconds": 0,
        "diarization_chunk_count": 0,
        "diarization_chunk_overlap_seconds": 0,
        "diarization_reconciliation_merge_count": 0,
        "diarization_reconciliation_ambiguous_count": 0,
        "speaker_audio_span_read_count": 0,
        "speaker_audio_span_wall_seconds": 0.0,
        "speaker_embedding_call_count": 0,
        "speaker_embedding_wall_seconds": 0.0,
        "speaker_embedding_input_seconds": 0.0,
        "speaker_embedding_calls_by_kind": "{}",
        "speaker_reconciliation_boundary_count": 0,
        "speaker_attribution_artifact_reused": False,
        "deterministic_cleanup_artifact_reused": False,
    }


def write_episode_summary_csv(path: Path, rows: List[Dict[str, object]]):
    sorted_rows = sorted(rows, key=lambda row: coerce_float(row.get("review_priority_score"), 0.0), reverse=True)
    fieldnames = [
        "episode",
        "episode_date",
        "episode_date_compact",
        "episode_year",
        "episode_month",
        "episode_day",
        "episode_sort_key",
        "review_priority_score",
        "review_priority_reason",
        "host_detected",
        "host_label",
        "known_speakers_detected",
        "host_duration_seconds",
        "host_share_of_speech",
        "top_host_similarity",
        "second_host_similarity",
        "host_similarity_margin",
        "speaker_count",
        "transcript_segments",
        "review_row_count",
        "host_match_near_threshold_count",
        "host_match_ambiguous_count",
        "host_low_coverage_count",
        "host_segment_review_count",
        "glossary_replacement_candidate_count",
        "host_not_detected_count",
        "speaker_similarity_drift_count",
        "cleanup_level",
        "cleanup_edit_count",
        "manual_correction_count",
        "processing_seconds",
        "language_model_warnings",
        "transcription_artifact_reused",
        "alignment_artifact_reused",
        "diarization_artifact_reused",
        "asr_provider",
        "alignment_provider",
        "speaker_embedding_provider",
        "speaker_attribution_artifact_reused",
        "deterministic_cleanup_artifact_reused",
        "review_attempted",
        "review_status",
        "review_skip_reason",
        "review_runtime_profile",
        "review_backend",
        "review_model_name",
        "reviewed_segment_count",
        "review_corrected_segment_count",
        "review_candidate_count",
        "review_context_segment_count",
        "review_skipped_segment_count",
        "reviewed_output_written",
        "review_pipeline_version",
        "review_enabled_stages",
        "review_completed_stages",
        "review_skipped_stages",
        "review_input_source",
        "review_episode_qa_mode",
        "active_learned_rule_ids",
        "contributing_learned_rule_ids",
        "review_calibration_source",
        "review_local_text_budget",
        "review_local_speaker_budget",
        "review_long_context_budget",
        "cleanup_review_corrected_count",
        "glossary_review_corrected_count",
        "speaker_consistency_review_corrected_count",
        "episode_qa_review_corrected_count",
        "review_material_change",
        "review_unique_stage_count",
        "review_noop_stage_count",
        "review_returned_change_count",
        "review_applied_change_count",
        "review_overridden_change_count",
        "episode_qa_added_value",
        "preferred_term_intervention_count",
        "review_guard_intervention_count",
        "speaker_drift_flag",
        "recurring_unnamed_speaker_flag",
        "host_profile_stability_flag",
        "processing_mode",
        "episode_contract_version",
        "contract_upgrade_method",
        "contract_upgrade_archive_path",
        "tier1_reused_from_existing",
        "review_backfilled_from_cleaned_json",
        "diarization_mode",
        "diarization_probe_attempted",
        "diarization_learned_route",
        "diarization_route_reason",
        "diarization_failure_floor_seconds",
        "diarization_safe_success_ceiling_seconds",
        "diarization_chunk_count",
        "diarization_chunk_overlap_seconds",
        "diarization_reconciliation_merge_count",
        "diarization_reconciliation_ambiguous_count",
        "speaker_audio_span_read_count",
        "speaker_audio_span_wall_seconds",
        "speaker_embedding_call_count",
        "speaker_embedding_wall_seconds",
        "speaker_embedding_input_seconds",
        "speaker_embedding_calls_by_kind",
        "speaker_reconciliation_boundary_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)


def coerce_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: object, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def normalize_episode_summary_row(row: Dict[str, object]) -> Dict[str, object]:
    float_fields = {
        "review_priority_score",
        "host_duration_seconds",
        "host_share_of_speech",
        "top_host_similarity",
        "second_host_similarity",
        "host_similarity_margin",
        "processing_seconds",
    }
    int_fields = {
        "episode_year",
        "episode_month",
        "episode_day",
        "episode_sort_key",
        "speaker_count",
        "transcript_segments",
        "review_row_count",
        "host_match_near_threshold_count",
        "host_match_ambiguous_count",
        "host_low_coverage_count",
        "host_segment_review_count",
        "glossary_replacement_candidate_count",
        "host_not_detected_count",
        "speaker_similarity_drift_count",
        "cleanup_edit_count",
        "manual_correction_count",
        "reviewed_segment_count",
        "review_corrected_segment_count",
        "review_candidate_count",
        "review_context_segment_count",
        "review_skipped_segment_count",
        "review_local_text_budget",
        "review_local_speaker_budget",
        "review_long_context_budget",
        "cleanup_review_corrected_count",
        "glossary_review_corrected_count",
        "speaker_consistency_review_corrected_count",
        "episode_qa_review_corrected_count",
        "review_unique_stage_count",
        "review_noop_stage_count",
        "review_returned_change_count",
        "review_applied_change_count",
        "review_overridden_change_count",
        "preferred_term_intervention_count",
        "review_guard_intervention_count",
        "diarization_failure_floor_seconds",
        "diarization_safe_success_ceiling_seconds",
        "diarization_chunk_count",
        "diarization_chunk_overlap_seconds",
        "diarization_reconciliation_merge_count",
        "diarization_reconciliation_ambiguous_count",
    }
    bool_fields = {
        "host_detected",
        "transcription_artifact_reused",
        "alignment_artifact_reused",
        "diarization_artifact_reused",
        "review_attempted",
        "reviewed_output_written",
        "review_material_change",
        "episode_qa_added_value",
        "speaker_drift_flag",
        "recurring_unnamed_speaker_flag",
        "host_profile_stability_flag",
        "tier1_reused_from_existing",
        "review_backfilled_from_cleaned_json",
        "diarization_probe_attempted",
        "diarization_learned_route",
    }

    normalized = dict(row)
    for field in float_fields:
        if field in normalized:
            if normalized[field] in ("", None):
                normalized[field] = ""
            else:
                normalized[field] = coerce_float(normalized[field], 0.0)

    for field in int_fields:
        if field in normalized:
            normalized[field] = coerce_int(normalized[field], 0)

    for field in bool_fields:
        if field in normalized:
            normalized[field] = coerce_bool(normalized[field], False)

    return normalized


def apply_review_metadata_to_summary(summary_row: Dict[str, object], review_result: Dict[str, object]):
    review_metadata = review_result["metadata"]
    stage_results = review_metadata.get("review_stage_results") if isinstance(review_metadata.get("review_stage_results"), dict) else {}
    change_summary = review_metadata.get("review_change_summary") if isinstance(review_metadata.get("review_change_summary"), dict) else {}
    guard_interventions = review_metadata.get("review_guard_interventions") if isinstance(review_metadata.get("review_guard_interventions"), dict) else {}
    summary_row["review_attempted"] = bool(review_result["attempted"])
    summary_row["review_status"] = str(review_metadata.get("review_status") or "")
    summary_row["review_skip_reason"] = str(review_metadata.get("review_skip_reason") or "")
    summary_row["review_runtime_profile"] = str(review_metadata.get("review_runtime_profile") or "")
    summary_row["review_backend"] = str(review_metadata.get("review_backend") or "")
    summary_row["review_model_name"] = str(review_metadata.get("review_model_name") or "")
    summary_row["reviewed_segment_count"] = int(review_metadata.get("reviewed_segment_count") or 0)
    summary_row["review_corrected_segment_count"] = int(review_metadata.get("corrected_segment_count") or 0)
    summary_row["review_candidate_count"] = int(review_metadata.get("review_candidate_count") or 0)
    summary_row["review_context_segment_count"] = int(review_metadata.get("review_context_segment_count") or 0)
    summary_row["review_skipped_segment_count"] = int(review_metadata.get("review_skipped_segment_count") or 0)
    summary_row["reviewed_output_written"] = bool(review_result["segments"])
    summary_row["review_pipeline_version"] = str(review_metadata.get("review_pipeline_version") or "")
    summary_row["review_enabled_stages"] = ";".join(str(item) for item in review_metadata.get("review_enabled_stages") or [])
    summary_row["review_completed_stages"] = ";".join(str(item) for item in review_metadata.get("review_completed_stages") or [])
    summary_row["review_skipped_stages"] = ";".join(str(item) for item in review_metadata.get("review_skipped_stages") or [])
    summary_row["review_input_source"] = str(review_metadata.get("review_input_source") or "")
    summary_row["review_episode_qa_mode"] = str(review_metadata.get("episode_qa_mode") or "")
    summary_row["active_learned_rule_ids"] = ";".join(str(item) for item in review_metadata.get("active_learned_rule_ids") or [])
    summary_row["contributing_learned_rule_ids"] = ";".join(str(item) for item in review_metadata.get("contributing_learned_rule_ids") or [])
    calibration = review_metadata.get("review_calibration") if isinstance(review_metadata.get("review_calibration"), dict) else {}
    families = calibration.get("families") if isinstance(calibration.get("families"), dict) else {}
    summary_row["review_calibration_source"] = (
        str(families.get("local_text_review", {}).get("calibration_source") or "")
        or str(families.get("local_speaker_review", {}).get("calibration_source") or "")
        or str(families.get("long_context_review", {}).get("calibration_source") or "")
    )
    summary_row["review_local_text_budget"] = int(families.get("local_text_review", {}).get("current_budget") or 0)
    summary_row["review_local_speaker_budget"] = int(families.get("local_speaker_review", {}).get("current_budget") or 0)
    summary_row["review_long_context_budget"] = int(families.get("long_context_review", {}).get("current_budget") or 0)
    summary_row["cleanup_review_corrected_count"] = int(
        ((stage_results.get("transcript_cleanup_review") or {}).get("corrected_segment_count")) or 0
    )
    summary_row["glossary_review_corrected_count"] = int(
        ((stage_results.get("glossary_correction_review") or {}).get("corrected_segment_count")) or 0
    )
    summary_row["speaker_consistency_review_corrected_count"] = int(
        ((stage_results.get("speaker_consistency_review") or {}).get("corrected_segment_count")) or 0
    )
    summary_row["episode_qa_review_corrected_count"] = int(
        ((stage_results.get("episode_qa_review") or {}).get("corrected_segment_count")) or 0
    )
    summary_row["review_material_change"] = bool(change_summary.get("material_change"))
    summary_row["review_unique_stage_count"] = int(change_summary.get("unique_stage_count") or 0)
    summary_row["review_noop_stage_count"] = int(change_summary.get("no_op_stage_count") or 0)
    summary_row["review_returned_change_count"] = int(change_summary.get("returned_change_count") or 0)
    summary_row["review_applied_change_count"] = int(change_summary.get("applied_change_count") or 0)
    summary_row["review_overridden_change_count"] = int(change_summary.get("overridden_change_count") or 0)
    summary_row["episode_qa_added_value"] = bool(change_summary.get("episode_qa_added_value"))
    summary_row["preferred_term_intervention_count"] = int(change_summary.get("protected_term_intervention_count") or 0)
    summary_row["review_guard_intervention_count"] = int(guard_interventions.get("protected_term_preservations") or 0)


def apply_speaker_risk_flags_to_summary(
    summary_row: Dict[str, object],
    speaker_mapping: Optional[Dict[str, str]] = None,
):
    labels = [str(label or "").strip() for label in (speaker_mapping or {}).values()]
    recurring_unnamed = any(label.upper().startswith("SPEAKER_") for label in labels)
    host_stability = (
        not bool(summary_row.get("host_detected"))
        or coerce_int(summary_row.get("host_match_near_threshold_count"), 0) > 0
        or coerce_int(summary_row.get("host_match_ambiguous_count"), 0) > 0
        or coerce_int(summary_row.get("host_low_coverage_count"), 0) > 0
        or coerce_float(summary_row.get("top_host_similarity"), 1.0) < 0.6
        or (
            summary_row.get("host_similarity_margin") not in ("", None)
            and coerce_float(summary_row.get("host_similarity_margin"), 1.0) < 0.08
        )
    )
    summary_row["speaker_drift_flag"] = coerce_int(summary_row.get("speaker_similarity_drift_count"), 0) > 0
    summary_row["recurring_unnamed_speaker_flag"] = recurring_unnamed
    summary_row["host_profile_stability_flag"] = host_stability


def checkpoint_path(output_dir: Path, audio_path: Path) -> Path:
    return output_dir / CHECKPOINT_DIRNAME / f"{audio_path.stem}.json"


def write_processing_checkpoint(
    output_dir: Path,
    audio_path: Path,
    stage: str,
    details: Optional[Dict[str, object]] = None,
):
    checkpoint_file = checkpoint_path(output_dir, audio_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 2,
        "audio_file": audio_path.name,
        "source_fingerprint": audio_file_fingerprint(audio_path),
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if details:
        payload["details"] = details
    state_atomic_write_text(checkpoint_file, json.dumps(payload, indent=2))


def load_processing_checkpoint(output_dir: Path, audio_path: Path) -> Optional[Dict[str, object]]:
    checkpoint_file = checkpoint_path(output_dir, audio_path)
    if not checkpoint_file.exists():
        return None
    try:
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("audio_file") != audio_path.name:
        return None
    try:
        if payload.get("source_fingerprint") != audio_file_fingerprint(audio_path):
            return None
    except OSError:
        return None
    return payload


def clear_processing_checkpoint(output_dir: Path, audio_path: Path):
    checkpoint_file = checkpoint_path(output_dir, audio_path)
    if checkpoint_file.exists():
        checkpoint_file.unlink()


def load_episode_summary_rows(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            episode = row.get("episode")
            if episode:
                rows[episode] = normalize_episode_summary_row(row)
        return rows


def load_processed_files(path: Path) -> Dict[str, Dict[str, object]]:
    return state_load_processed_files(path)


def save_processed_files(path: Path, processed_files: Dict[str, Dict[str, object]]):
    state_save_processed_files(path, processed_files)


def expected_output_paths(audio_path: Path, output_dir: Path) -> List[Path]:
    return state_expected_output_paths(audio_path, output_dir)


def expected_review_output_paths(audio_path: Path, output_dir: Path) -> List[Path]:
    base_name = audio_path.stem
    return [
        output_dir / f"{base_name}_reviewed_speaker_transcript.txt",
        output_dir / f"{base_name}_reviewed_host_only.txt",
        output_dir / f"{base_name}_reviewed_speaker_transcript.json",
    ]


def expected_cleaned_output_paths(audio_path: Path, output_dir: Path) -> List[Path]:
    base_name = audio_path.stem
    return [
        output_dir / f"{base_name}_cleaned_speaker_transcript.txt",
        output_dir / f"{base_name}_cleaned_host_only.txt",
        output_dir / f"{base_name}_cleaned_speaker_transcript.json",
    ]


def cleaned_json_output_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}_cleaned_speaker_transcript.json"


def reviewed_json_output_path(audio_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{audio_path.stem}_reviewed_speaker_transcript.json"


def baseline_output_bundle_complete(audio_path: Path, output_dir: Path) -> bool:
    expected_paths = expected_output_paths(audio_path, output_dir) + expected_cleaned_output_paths(audio_path, output_dir)
    if not all(path.exists() and path.is_file() for path in expected_paths):
        return False
    # File presence alone is not enough to skip an episode: a killed write can
    # leave an empty or truncated companion beside otherwise valid outputs.
    for path in (
        output_dir / f"{audio_path.stem}_speaker_transcript.json",
        output_dir / f"{audio_path.stem}_cleaned_speaker_transcript.json",
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list) or not payload["segments"]:
            return False
    return True


def load_cleaned_transcript_payload(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cleaned transcript JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cleaned transcript JSON is invalid: {path} ({exc})") from exc

    errors = validate_transcript_payload(payload)
    if errors:
        raise RuntimeError(f"Cleaned transcript JSON failed contract validation: {path} ({'; '.join(errors[:5])})")

    if not isinstance(payload.get("segments"), list) or not payload["segments"]:
        raise RuntimeError(f"Cleaned transcript JSON does not contain any usable segments: {path}")
    return payload


def load_reviewed_transcript_payload(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Reviewed transcript JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Reviewed transcript JSON is invalid: {path} ({exc})") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Reviewed transcript JSON is not an object: {path}")
    if payload.get("text_version") not in {"reviewed_llm", "reviewed_llm_high_context"}:
        raise RuntimeError(f"Reviewed transcript JSON is not a reviewed text variant: {path}")
    if not isinstance(payload.get("segments"), list) or not payload["segments"]:
        raise RuntimeError(f"Reviewed transcript JSON does not contain any usable segments: {path}")
    return payload


def enabled_review_stage_names(runtime_review_config: Optional[Dict[str, object]]) -> List[str]:
    resolved = resolve_review_runtime_config(runtime_review_config or {})
    return [
        stage_name
        for stage_name in (
            "transcript_cleanup_review",
            "glossary_correction_review",
            "speaker_consistency_review",
            "episode_qa_review",
        )
        if resolved.get(stage_name)
    ]


def reviewed_payload_is_usable(payload: Dict[str, object]) -> None:
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Reviewed payload does not contain usable segments.")

    transcript_errors = validate_transcript_payload(payload)
    blocking_errors = [
        error
        for error in transcript_errors
        if not error.startswith("missing top-level fields: review_")
    ]
    if blocking_errors:
        raise RuntimeError(f"Reviewed payload failed transcript validation: {'; '.join(blocking_errors[:5])}")

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise RuntimeError(f"Reviewed payload segment {index} is not an object.")
        missing = [
            field
            for field in ("id", "start", "end", "speaker", "text", "original_text", "llm_reviewed_text")
            if segment.get(field) in ("", None)
        ]
        if missing:
            raise RuntimeError(
                f"Reviewed payload segment {index} is missing required legacy review fields: {', '.join(missing)}"
            )

 
def extract_completed_review_stage_names(payload: Dict[str, object]) -> List[str]:
    review_metadata = payload.get("review_metadata")
    if not isinstance(review_metadata, dict):
        return []

    completed = []
    explicit_completed = review_metadata.get("review_completed_stages")
    if isinstance(explicit_completed, list):
        completed.extend(
            str(stage_name)
            for stage_name in explicit_completed
            if isinstance(stage_name, str) and stage_name.strip()
        )

    stage_results = review_metadata.get("review_stage_results")
    if isinstance(stage_results, dict):
        for stage_name, stage_result in stage_results.items():
            if (
                isinstance(stage_name, str)
                and isinstance(stage_result, dict)
                and str(stage_result.get("status") or "").strip().lower() == "completed"
            ):
                completed.append(stage_name)

    ordered = []
    seen = set()
    for stage_name in completed:
        if stage_name not in seen:
            seen.add(stage_name)
            ordered.append(stage_name)
    return ordered


def classify_reviewed_payload_for_skip(
    payload: Dict[str, object],
    required_stages: List[str],
) -> Dict[str, object]:
    reviewed_payload_is_usable(payload)
    current_errors = validate_reviewed_transcript_payload(payload)
    completed_stages = extract_completed_review_stage_names(payload)
    completed_stage_set = set(completed_stages)
    missing_stages = [stage_name for stage_name in required_stages if stage_name not in completed_stage_set]

    if not required_stages:
        return {
            "status": "current_review_complete" if not current_errors else "review_stage_shortfall",
            "reason": "" if not current_errors else "review payload missing explicit stage-completion evidence",
            "completed_stages": completed_stages,
            "missing_stages": [],
        }

    if missing_stages:
        if completed_stages:
            return {
                "status": "review_stage_shortfall",
                "reason": f"reviewed output is missing required stages: {', '.join(missing_stages)}",
                "completed_stages": completed_stages,
                "missing_stages": missing_stages,
            }
        return {
            "status": "review_stage_shortfall",
            "reason": f"reviewed output does not prove completion of required stages: {', '.join(missing_stages)}",
            "completed_stages": [],
            "missing_stages": missing_stages,
        }

    return {
        "status": "current_review_complete",
        "reason": "",
        "completed_stages": completed_stages,
        "missing_stages": [],
    }


def reviewed_output_bundle_status(
    audio_path: Path,
    output_dir: Path,
    runtime_review_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    expected_paths = expected_review_output_paths(audio_path, output_dir)
    if not all(path.exists() for path in expected_paths):
        return {
            "status": "review_missing",
            "reason": "missing reviewed output files",
            "review_json_path": reviewed_json_output_path(audio_path, output_dir),
            "required_stages": enabled_review_stage_names(runtime_review_config),
            "completed_stages": [],
            "missing_stages": enabled_review_stage_names(runtime_review_config),
        }

    review_json_path = reviewed_json_output_path(audio_path, output_dir)
    required_stages = enabled_review_stage_names(runtime_review_config)
    try:
        payload = load_reviewed_transcript_payload(review_json_path)
        review_state = classify_reviewed_payload_for_skip(payload, required_stages)
        return {
            "status": review_state["status"],
            "reason": review_state["reason"],
            "review_json_path": review_json_path,
            "required_stages": required_stages,
            "completed_stages": review_state["completed_stages"],
            "missing_stages": review_state["missing_stages"],
        }
    except RuntimeError as exc:
        return {
            "status": "review_corrupt",
            "reason": str(exc),
            "review_json_path": review_json_path,
            "required_stages": required_stages,
            "completed_stages": [],
            "missing_stages": required_stages,
        }


def segment_items_from_cleaned_payload(payload: Dict[str, object]) -> List[SegmentItem]:
    rebuilt_segments: List[SegmentItem] = []
    for index, raw_segment in enumerate(payload.get("segments") or []):
        if not isinstance(raw_segment, dict):
            raise RuntimeError(f"Segment {index} in cleaned transcript JSON is not an object.")
        missing = [
            field
            for field in ("id", "start", "end", "speaker", "text")
            if raw_segment.get(field) in ("", None)
        ]
        if missing:
            raise RuntimeError(
                f"Segment {index} in cleaned transcript JSON is missing required fields: {', '.join(missing)}"
            )
        words_payload = raw_segment.get("words") or []
        words = [
            WordItem(
                start=word.get("start") if isinstance(word, dict) else None,
                end=word.get("end") if isinstance(word, dict) else None,
                word=str(word.get("word") or "") if isinstance(word, dict) else "",
                speaker=str(word.get("speaker") or raw_segment.get("speaker") or "") if isinstance(word, dict) else "",
            )
            for word in words_payload
            if isinstance(word, dict)
        ]
        confidence = raw_segment.get("transcription_confidence") if isinstance(raw_segment.get("transcription_confidence"), dict) else {}
        rebuilt_segments.append(
            SegmentItem(
                id=int(raw_segment["id"]),
                start=float(raw_segment["start"]),
                end=float(raw_segment["end"]),
                text=str(raw_segment["text"]),
                speaker=str(raw_segment.get("speaker") or ""),
                avg_logprob=raw_segment.get("avg_logprob", confidence.get("avg_logprob")),
                no_speech_prob=raw_segment.get("no_speech_prob", confidence.get("no_speech_prob")),
                words=words,
                original_text=raw_segment.get("original_text"),
                cleanup_applied=coerce_bool(raw_segment.get("cleanup_applied"), False),
                cleanup_level=str(raw_segment.get("cleanup_level") or ""),
                manual_correction_applied=coerce_bool(raw_segment.get("manual_correction_applied"), False),
                original_speaker=raw_segment.get("original_speaker"),
            )
        )
    return rebuilt_segments


def provider_configuration_shortfall(
    cleaned_payload: Optional[Dict[str, object]],
    runtime_config: Optional[Dict[str, object]],
) -> List[str]:
    if not isinstance(cleaned_payload, dict):
        return []
    metadata = cleaned_payload.get("metadata")
    provenance = metadata.get("stage_provenance") if isinstance(metadata, dict) else None
    config = runtime_config or {}
    workflow_profile = str(config.get("workflow_profile") or "podcast")
    configured_asr_model = str(config.get("model_id") or "").strip()
    if not configured_asr_model:
        configured_asr_model = str(config.get("model") or "").strip()
        if str(config.get("asr_provider") or "faster_whisper") == "faster_whisper":
            configured_asr_model = {
                "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
            }.get(configured_asr_model, f"Systran/faster-whisper-{configured_asr_model}")
    if not isinstance(provenance, dict) or not provenance:
        legacy_shortfalls = []
        if str(config.get("asr_provider") or "faster_whisper") != "faster_whisper":
            legacy_shortfalls.append("transcription:unproven_provider")
        requested_model = str(config.get("model") or "")
        recorded_model = str(cleaned_payload.get("pipeline_version") or "")
        if requested_model and recorded_model and requested_model != recorded_model:
            legacy_shortfalls.append("transcription:model")
        if str(config.get("alignment_provider") or "timestamp_passthrough") != "timestamp_passthrough":
            legacy_shortfalls.append("alignment:unproven_provider")
        if str(config.get("diarization_model") or "pyannote/speaker-diarization-community-1") != "pyannote/speaker-diarization-community-1":
            legacy_shortfalls.append("diarization:unproven_model")
        if workflow_profile != "anonymous_meeting":
            if str(config.get("speaker_embedding_provider") or "speechbrain_ecapa") != "speechbrain_ecapa":
                legacy_shortfalls.append("speaker_embedding:unproven_provider")
        return legacy_shortfalls
    expectations = {
        "transcription": (
            str(config.get("asr_provider") or "faster_whisper"),
            configured_asr_model,
        ),
        "alignment": (
            str(config.get("alignment_provider") or "timestamp_passthrough"),
            str(config.get("alignment_model") or ""),
        ),
        "diarization": (
            "pyannote",
            str(config.get("diarization_model") or ""),
        ),
    }
    if workflow_profile != "anonymous_meeting":
        expectations["speaker_embedding"] = (
            str(config.get("speaker_embedding_provider") or "speechbrain_ecapa"),
            str(config.get("speaker_model") or ""),
        )
    shortfalls: List[str] = []
    for stage_name, (expected_provider, expected_model) in expectations.items():
        stage_payload = provenance.get(stage_name)
        if not isinstance(stage_payload, dict):
            continue
        provider_payload = stage_payload.get("provider")
        if not isinstance(provider_payload, dict):
            continue
        actual_provider = str(provider_payload.get("provider") or "")
        actual_model = str(provider_payload.get("model") or "")
        if expected_provider and actual_provider and expected_provider != actual_provider:
            shortfalls.append(f"{stage_name}:provider")
            continue
        # Blank alignment model means the provider's language default and is not a mismatch.
        if expected_model and actual_model and expected_model != actual_model:
            shortfalls.append(f"{stage_name}:model")
    if workflow_profile == "anonymous_meeting":
        # A prior podcast-mode bundle may look complete while containing
        # identity evidence. Force a rebuild so anonymous mode cannot reuse
        # host/embedding artifacts from that run.
        metadata = cleaned_payload.get("metadata") if isinstance(cleaned_payload, dict) else None
        evidence = cleaned_payload.get("speaker_identity_evidence") if isinstance(cleaned_payload, dict) else None
        if not evidence and isinstance(metadata, dict):
            evidence = metadata.get("speaker_identity_evidence")
        speaker_stage = provenance.get("speaker_embedding") if isinstance(provenance, dict) else None
        speaker_provider = ""
        if isinstance(speaker_stage, dict) and isinstance(speaker_stage.get("provider"), dict):
            speaker_provider = str(speaker_stage["provider"].get("provider") or "")
        if evidence or (speaker_provider and speaker_provider != "anonymous_meeting"):
            shortfalls.append("workflow_profile")
    return shortfalls


def classify_episode_processing_state(
    audio_path: Path,
    output_dir: Path,
    processed_files: Dict[str, Dict[str, object]],
    existing_summary_rows: Dict[str, Dict[str, object]],
    runtime_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    resolved_review = resolve_review_runtime_config(runtime_config or {})
    baseline_bundle_complete = baseline_output_bundle_complete(audio_path, output_dir)
    baseline_resume_complete = state_is_file_already_processed(
        audio_path,
        output_dir,
        processed_files,
        existing_summary_rows,
    )
    cleaned_json_path = cleaned_json_output_path(audio_path, output_dir)
    cleaned_json_usable = False
    cleaned_json_error = ""
    cleaned_payload = None
    if cleaned_json_path.exists():
        try:
            cleaned_payload = load_cleaned_transcript_payload(cleaned_json_path)
            cleaned_json_usable = True
        except RuntimeError as exc:
            cleaned_json_error = str(exc)
    reviewed_bundle = reviewed_output_bundle_status(audio_path, output_dir, resolved_review)
    manifest_path = output_dir / f"{audio_path.stem}_manifest.json"
    manifest_payload: Dict[str, object] = {}
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded_manifest, dict):
                manifest_payload = loaded_manifest
        except (OSError, json.JSONDecodeError):
            pass
    contract_status = episode_contract_status(cleaned_payload, manifest_payload)
    baseline_complete = baseline_resume_complete
    provider_shortfalls = provider_configuration_shortfall(cleaned_payload, runtime_config)
    if resolved_review["any_review_enabled"]:
        baseline_complete = baseline_bundle_complete and cleaned_json_usable

    if provider_shortfalls:
        state = "needs_tier1"
    elif not resolved_review["any_review_enabled"]:
        # A valid output bundle is authoritative even if the bookkeeping files
        # were lost. The source/provider checks above still prevent reuse after
        # an input or required provider changed.
        state = "complete" if baseline_bundle_complete else "needs_tier1"
    elif baseline_complete and cleaned_json_usable:
        if reviewed_bundle["status"] == "current_review_complete":
            state = "complete"
        else:
            state = "needs_tier2_only"
    else:
        state = "needs_tier1"

    legacy_processing_state = state
    if contract_status["status"] != "v2_complete":
        speaker_evidence_complete = bool(
            (cleaned_payload or {}).get("speaker_identity_evidence_complete")
            or (cleaned_payload or {}).get("speaker_identity_evidence")
            or (
                ((cleaned_payload or {}).get("metadata") or {}).get("speaker_identity_evidence_complete")
                if isinstance((cleaned_payload or {}).get("metadata"), dict)
                else False
            )
            or (
                ((cleaned_payload or {}).get("metadata") or {}).get("speaker_identity_evidence")
                if isinstance((cleaned_payload or {}).get("metadata"), dict)
                else []
            )
            or not (cleaned_payload or {}).get("segments")
        )
        speaker_artifact_path = (
            output_dir / ARTIFACT_DIRNAME / audio_path.stem / "speaker_attribution.json"
        )
        cached_speaker_evidence = False
        if speaker_artifact_path.exists():
            try:
                artifact_payload = json.loads(
                    speaker_artifact_path.read_text(encoding="utf-8-sig")
                )
                stage_payload = artifact_payload.get("payload") or {}
                stage_metadata = (
                    stage_payload.get("metadata")
                    if isinstance(stage_payload, dict)
                    else {}
                )
                cached_speaker_evidence = bool(
                    isinstance(stage_metadata, dict)
                    and stage_metadata.get("speaker_identity_evidence")
                )
            except (OSError, json.JSONDecodeError):
                cached_speaker_evidence = False
        if state == "complete" and speaker_evidence_complete:
            state = "needs_v2_delta_upgrade"
        elif state == "needs_tier2_only" and speaker_evidence_complete:
            pass
        elif cached_speaker_evidence or (
            (output_dir / ARTIFACT_DIRNAME / audio_path.stem).exists()
            and any((output_dir / ARTIFACT_DIRNAME / audio_path.stem).glob("*.json"))
        ):
            state = "needs_v2_cached_rebuild"
        else:
            state = "needs_v2_full_reprocess" if audio_path.exists() else "v2_upgrade_blocked"

    return {
        "state": state,
        "legacy_processing_state": legacy_processing_state,
        "episode_contract_target": EPISODE_CONTRACT_V2,
        "episode_contract_status": contract_status["status"],
        "episode_contract_reason": contract_status["reason"],
        "baseline_complete": baseline_complete,
        "baseline_bundle_complete": baseline_bundle_complete,
        "baseline_resume_complete": baseline_resume_complete,
        "cleaned_json_path": cleaned_json_path,
        "cleaned_json_usable": cleaned_json_usable,
        "cleaned_json_error": cleaned_json_error,
        "review_bundle_status": reviewed_bundle["status"],
        "review_bundle_reason": reviewed_bundle["reason"],
        "required_review_stages": reviewed_bundle["required_stages"],
        "completed_review_stages": reviewed_bundle["completed_stages"],
        "missing_review_stages": reviewed_bundle["missing_stages"],
        "review_enabled": bool(resolved_review["any_review_enabled"]),
        "provider_shortfalls": provider_shortfalls,
    }


def is_file_already_processed(
    audio_path: Path,
    output_dir: Path,
    processed_files: Dict[str, Dict[str, object]],
    existing_summary_rows: Dict[str, Dict[str, object]],
    runtime_config: Optional[Dict[str, object]] = None,
) -> bool:
    state = classify_episode_processing_state(
        audio_path,
        output_dir,
        processed_files,
        existing_summary_rows,
        runtime_config,
    )
    return state["state"] in {"complete", "v2_complete"}


def state_requires_tier1(state: str) -> bool:
    return state in {"needs_tier1", "needs_v2_cached_rebuild", "needs_v2_full_reprocess"}


def process_v2_delta_upgrade(
    audio_path: Path,
    output_dir: Path,
    existing_summary_row: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    print(f"Processing {audio_path.name}")
    print_episode_mode("v2 contract delta upgrade")
    archive_started = time.perf_counter()
    archive_components: Dict[str, float] = {}
    print_episode_stage(1, 2, "archive legacy contract artifacts")
    with timed_component(archive_components, "upgrade episode bundle"):
        result = upgrade_episode_bundle_v2(audio_path, output_dir)
    archive_elapsed = time.perf_counter() - archive_started
    print(f"  archive legacy contract artifacts complete in {archive_elapsed:.1f}s")
    print_timing_component_summary("archive legacy contract artifacts", archive_components)

    metadata_started = time.perf_counter()
    metadata_components: Dict[str, float] = {}
    print_episode_stage(2, 2, "write episode-contract-v2 metadata")
    with timed_component(metadata_components, "build contract metadata"):
        summary = dict(existing_summary_row or {})
        summary.setdefault("episode", audio_path.name)
        summary.update(
            {
                "processing_mode": "v2 contract delta upgrade",
                "episode_contract_version": EPISODE_CONTRACT_V2,
                "contract_upgrade_method": result["upgrade_method"],
                "contract_upgrade_archive_path": result["archive_path"],
                "tier1_reused_from_existing": True,
            }
        )
    metadata_elapsed = time.perf_counter() - metadata_started
    print(f"  write episode-contract-v2 metadata complete in {metadata_elapsed:.1f}s")
    print_timing_component_summary("write episode-contract-v2 metadata", metadata_components)
    print(f"  upgraded files: {len(result['upgraded_files'])}")
    return summary


def reviewed_text_version_from_metadata(review_metadata: Dict[str, object]) -> str:
    return (
        "reviewed_llm_high_context"
        if review_metadata.get("review_runtime_profile") == "high_context_5090"
        else "reviewed_llm"
    )


def write_reviewed_output_bundle(
    audio_path: Path,
    output_dir: Path,
    reviewed_segments: List[SegmentItem],
    review_metadata: Dict[str, object],
    host_output_labels: set[str],
    episode_metadata: Dict[str, object],
    info_payload: Dict[str, object],
    diarized_turns: List[Dict[str, object]],
    speaker_mapping: Dict[str, str],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    known_assignments: Dict[str, Dict[str, object]],
    runtime_config: Optional[Dict[str, object]],
) -> List[Path]:
    if not reviewed_segments:
        return []

    reviewed_text_version = reviewed_text_version_from_metadata(review_metadata)
    reviewed_metadata = {
        **episode_metadata,
        "text_version": reviewed_text_version,
    }
    base_name = audio_path.stem
    reviewed_paths = [
        output_dir / f"{base_name}_reviewed_speaker_transcript.txt",
        output_dir / f"{base_name}_reviewed_host_only.txt",
        output_dir / f"{base_name}_reviewed_speaker_transcript.json",
    ]
    output_write_text_transcript(
        reviewed_paths[0],
        reviewed_segments,
        format_timestamp,
        host_only=False,
        metadata=reviewed_metadata,
    )
    output_write_text_transcript(
        reviewed_paths[1],
        reviewed_segments,
        format_timestamp,
        host_only=True,
        host_labels=host_output_labels,
        metadata=reviewed_metadata,
    )
    output_write_json_output(
        reviewed_paths[2],
        source_file=str(audio_path),
        info_payload=info_payload,
        diarized_turns=diarized_turns,
        segments=reviewed_segments,
        speaker_mapping=speaker_mapping,
        host_speaker=host_speaker,
        durations=durations,
        known_assignments=known_assignments,
        metadata=reviewed_metadata,
        text_version=reviewed_text_version,
        pipeline_version=runtime_config.get("model", "") if runtime_config else "",
        review_metadata=review_metadata,
    )
    return reviewed_paths


def build_review_backfill_summary_row(
    audio_path: Path,
    cleaned_payload: Dict[str, object],
    cleaned_segments: List[SegmentItem],
    review_result: Dict[str, object],
    existing_summary_row: Optional[Dict[str, object]] = None,
    processing_seconds: float = 0.0,
) -> Dict[str, object]:
    metadata = cleaned_payload.get("metadata") if isinstance(cleaned_payload.get("metadata"), dict) else {}
    speaker_mapping = {
        str(key): str(value)
        for key, value in (cleaned_payload.get("speaker_mapping") or {}).items()
        if value not in ("", None)
    }
    host_speaker = cleaned_payload.get("host_original_speaker_id")
    host_label = speaker_mapping.get(str(host_speaker), "") if host_speaker not in ("", None) else ""
    source_row = dict(existing_summary_row or {})
    summary_row = build_episode_summary_row(
        audio_path=audio_path,
        normalized_segments=cleaned_segments,
        review_rows=[],
        host_speaker=None,
        durations={},
        similarity_scores={},
        speaker_mapping={},
        known_assignments={},
        episode_metadata=metadata,
    )
    summary_row.update(source_row)
    summary_row["episode"] = audio_path.name
    summary_row["episode_date"] = metadata.get("episode_date", summary_row.get("episode_date", ""))
    summary_row["episode_date_compact"] = metadata.get("episode_date_compact", summary_row.get("episode_date_compact", ""))
    summary_row["episode_year"] = metadata.get("episode_year", summary_row.get("episode_year", ""))
    summary_row["episode_month"] = metadata.get("episode_month", summary_row.get("episode_month", ""))
    summary_row["episode_day"] = metadata.get("episode_day", summary_row.get("episode_day", ""))
    summary_row["episode_sort_key"] = metadata.get("episode_sort_key", summary_row.get("episode_sort_key", ""))
    summary_row["host_detected"] = source_row.get("host_detected", bool(cleaned_payload.get("host_detected")))
    summary_row["host_label"] = source_row.get("host_label") or host_label
    summary_row["transcript_segments"] = len(cleaned_segments)
    summary_row["cleanup_level"] = str(source_row.get("cleanup_level") or "cleaned_json_reuse")
    summary_row["cleanup_edit_count"] = coerce_int(source_row.get("cleanup_edit_count"), 0)
    summary_row["manual_correction_count"] = coerce_int(source_row.get("manual_correction_count"), 0)
    summary_row["processing_seconds"] = round(processing_seconds, 2)
    summary_row["transcription_artifact_reused"] = True
    summary_row["diarization_artifact_reused"] = True
    apply_review_metadata_to_summary(summary_row, review_result)
    apply_speaker_risk_flags_to_summary(summary_row, speaker_mapping)
    summary_row["processing_mode"] = "tier2-only backfill"
    summary_row["tier1_reused_from_existing"] = True
    summary_row["review_backfilled_from_cleaned_json"] = True
    return summary_row


def process_review_backfill_from_cleaned_json(
    audio_path: Path,
    output_dir: Path,
    runtime_config: Optional[Dict[str, object]] = None,
    review_calibration_session: Optional[ReviewCalibrationSession] = None,
    existing_summary_row: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    started = time.perf_counter()
    stage1_started = time.perf_counter()
    stage1_components: Dict[str, float] = {}
    print(f"Processing {audio_path.name}")
    print_episode_mode("tier2-only backfill")
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_checkpoint = load_processing_checkpoint(output_dir, audio_path)
    if prior_checkpoint:
        print(
            f"  resuming from durable checkpoint: "
            f"{str(prior_checkpoint.get('stage') or 'unknown').replace('_', ' ')}"
        )
    print_episode_stage(1, 3, "load cleaned transcript")
    cleaned_path = cleaned_json_output_path(audio_path, output_dir)
    with timed_component(stage1_components, "load cleaned transcript"):
        cleaned_payload = load_cleaned_transcript_payload(cleaned_path)
    write_processing_checkpoint(
        output_dir,
        audio_path,
        "cleaned_transcript_loaded",
        {"segment_count": len(cleaned_payload.get("segments") or [])},
    )
    with timed_component(stage1_components, "archive legacy artifacts"):
        archive_legacy_episode_bundle(audio_path, output_dir)
    with timed_component(stage1_components, "rebuild segment objects"):
        cleaned_segments = segment_items_from_cleaned_payload(cleaned_payload)
    stage1_elapsed = time.perf_counter() - stage1_started
    print(f"  load cleaned transcript complete in {stage1_elapsed:.1f}s")
    print_timing_component_summary("load cleaned transcript", stage1_components)

    runtime_review_config = resolve_review_runtime_config(runtime_config or {})
    review_started = time.perf_counter()
    review_component_timings: Dict[str, float] = {}
    print_episode_stage(2, 3, "review")
    review_debug_dir = review_debug_directory(
        runtime_review_config,
        {
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "review_input_source": "cleaned_json_backfill",
        },
    )
    if review_debug_dir is not None:
        print(f"  review debug output: {review_debug_dir}")
    review_result = review_segments(
        cleaned_segments,
        runtime_review_config,
        review_input_source="cleaned_json_backfill",
        calibration_session=review_calibration_session,
        progress_callback=make_checkpointed_review_progress_callback(
            output_dir, audio_path, review_component_timings
        ),
        debug_context={
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "review_input_source": "cleaned_json_backfill",
        },
    )
    review_elapsed = time.perf_counter() - review_started
    print(f"  review complete in {review_elapsed:.1f}s")
    print_timing_component_summary("review", review_component_timings)
    review_metadata = review_result["metadata"]
    write_processing_checkpoint(
        output_dir,
        audio_path,
        "review_complete",
        {
            "review_status": review_metadata.get("review_status", ""),
            "candidate_count": review_metadata.get("review_candidate_count", 0),
            "corrected_segment_count": review_metadata.get("corrected_segment_count", 0),
        },
    )

    host_speaker = cleaned_payload.get("host_original_speaker_id")
    speaker_mapping = {
        str(key): str(value)
        for key, value in (cleaned_payload.get("speaker_mapping") or {}).items()
        if value not in ("", None)
    }
    known_assignments = {
        str(key): value
        for key, value in (cleaned_payload.get("known_speaker_assignments") or {}).items()
        if isinstance(value, dict)
    }
    resolved_host_label = speaker_mapping.get(str(host_speaker), "HOST") if host_speaker else "HOST"
    host_output_labels = {
        resolved_host_label,
        "HOST",
        *[
            speaker_mapping.get(speaker_id, speaker_id)
            for speaker_id, assignment in known_assignments.items()
            if assignment.get("is_host")
            or any(str(role).lower() in {"host", "co-host"} for role in assignment.get("roles") or [])
        ],
    }
    episode_metadata = cleaned_payload.get("metadata") if isinstance(cleaned_payload.get("metadata"), dict) else {}
    durations = {
        str(key): float(value)
        for key, value in (cleaned_payload.get("speaker_durations_seconds") or {}).items()
        if value not in ("", None)
    }
    diarized_turns = [
        turn for turn in (cleaned_payload.get("diarization_turns") or []) if isinstance(turn, dict)
    ]
    info_payload = cleaned_payload.get("transcription") if isinstance(cleaned_payload.get("transcription"), dict) else {}
    writing_started = time.perf_counter()
    writing_components: Dict[str, float] = {}
    print_episode_stage(3, 3, "writing reviewed outputs")
    with timed_component(writing_components, "reviewed output bundle"):
        reviewed_paths = write_reviewed_output_bundle(
            audio_path=audio_path,
            output_dir=output_dir,
            reviewed_segments=review_result["segments"],
            review_metadata=review_metadata,
            host_output_labels=host_output_labels,
            episode_metadata=episode_metadata,
            info_payload=info_payload,
            diarized_turns=diarized_turns,
            speaker_mapping=speaker_mapping,
            host_speaker=str(host_speaker) if host_speaker not in ("", None) else None,
            durations=durations,
            known_assignments=known_assignments,
            runtime_config=runtime_config,
        )
    with timed_component(writing_components, "build summary and manifest"):
        summary_row = build_review_backfill_summary_row(
            audio_path,
            cleaned_payload,
            cleaned_segments,
            review_result,
            existing_summary_row=existing_summary_row,
            processing_seconds=time.perf_counter() - started,
        )
        outputs = [
            path
            for path in (
                expected_output_paths(audio_path, output_dir)
                + expected_cleaned_output_paths(audio_path, output_dir)
                + reviewed_paths
            )
            if path.exists()
        ]
        output_write_output_manifest(
            output_dir / f"{audio_path.stem}_manifest.json",
            source_file=str(audio_path),
            source_fingerprint=audio_file_fingerprint(audio_path),
            config=runtime_config or {},
            outputs=outputs,
            timings={"review_backfill": time.perf_counter() - started, "total": time.perf_counter() - started},
            summary=summary_row,
        )
    with timed_component(writing_components, "contract metadata upgrade"):
        upgrade_episode_bundle_v2(audio_path, output_dir, method="tier2_backfill_contract_upgrade")
    summary_row["episode_contract_version"] = EPISODE_CONTRACT_V2
    summary_row["contract_upgrade_method"] = "tier2_backfill_contract_upgrade"
    writing_elapsed = time.perf_counter() - writing_started
    print(f"  writing reviewed outputs complete in {writing_elapsed:.1f}s")
    print_timing_component_summary("writing reviewed outputs", writing_components)
    print(f"  reviewed output written: {bool(reviewed_paths)}")
    return summary_row


def write_text_transcript(
    path: Path,
    segments: List[SegmentItem],
    host_only: bool = False,
    host_labels: Optional[set[str]] = None,
):
    lines = []
    host_labels = host_labels or {"HOST"}
    for segment in segments:
        if host_only and segment.speaker not in host_labels:
            continue
        label = segment.speaker or "UNKNOWN"
        lines.append(f"[{format_timestamp(segment.start)}][{label}] {segment.text}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_output(
    path: Path,
    source_file: str,
    info_payload: Dict[str, object],
    diarized_turns: List[Dict[str, object]],
    segments: List[SegmentItem],
    speaker_mapping: Dict[str, str],
    host_speaker: Optional[str],
    durations: Dict[str, float],
    known_assignments: Dict[str, Dict[str, object]],
):
    payload = {
        "source_file": source_file,
        "transcription": info_payload,
        "host_detected": host_speaker is not None,
        "host_original_speaker_id": host_speaker,
        "speaker_mapping": speaker_mapping,
        "known_speaker_assignments": known_assignments,
        "speaker_durations_seconds": durations,
        "diarization_turns": diarized_turns,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "words": [asdict(word) for word in segment.words],
            }
            for segment in segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def runtime_config_payload(args) -> Dict[str, object]:
    keys = [
        "workflow_profile",
        "asr_provider",
        "model",
        "model_id",
        "model_revision",
        "language",
        "device",
        "compute_type",
        "beam_size",
        "batch_size",
        "diarization_model",
        "diarization_model_revision",
        "alignment_provider",
        "alignment_model",
        "alignment_model_revision",
        "speaker_embedding_provider",
        "speaker_model",
        "speaker_model_revision",
        "host_threshold",
        "min_host_seconds",
        "max_embedding_seconds",
        "num_speakers",
        "cleanup_level",
        "assume_dominant_speaker_is_host",
        "resume_intermediates",
        "archive_debug_artifacts",
        "preferred_terms_file",
        "replacement_map_json",
        "known_speakers_dir",
        "filename_date_preset",
        "filename_date_position",
        "filename_date_formats",
        "runtime_profile",
        "backend",
        "review_base_url",
        "review_model_name",
        "review_reasoning_effort",
        "review_batch_token_limit",
        "review_candidate_filter",
        "review_context_budget",
        "review_structured_output_support",
        "review_transcript_qa_available",
        "review_episode_wide_correction_available",
        "review_debug",
        "review_debug_dir",
        "review_auto_calibrate",
        "review_auto_adapt_upward",
        "transcript_cleanup_review",
        "glossary_correction_review",
        "speaker_consistency_review",
        "episode_qa_review",
    ]
    payload = {key: getattr(args, key, None) for key in keys}
    if payload.get("workflow_profile") == "anonymous_meeting":
        payload.update(
            {
                "backend": "none",
                "review_base_url": "",
                "review_model_name": "",
                "review_reasoning_effort": "none",
                "review_candidate_filter": False,
                "review_debug": False,
                "transcript_cleanup_review": False,
                "glossary_correction_review": False,
                "speaker_consistency_review": False,
                "episode_qa_review": False,
                "review_auto_calibrate": False,
                "review_auto_adapt_upward": False,
                "assume_dominant_speaker_is_host": False,
                "host_reference": "",
                "host_profile_json": "",
                "known_speakers_dir": "",
            }
        )
    payload["preferred_terms"] = list(dict.fromkeys(
        load_preferred_terms(payload.get("preferred_terms_file")) + list(getattr(args, "preferred_terms", []) or [])
    ))
    payload["filename_date"] = {
        "preset": getattr(args, "filename_date_preset", "strict_iso"),
        "position": getattr(args, "filename_date_position", "last"),
        "formats": getattr(args, "filename_date_formats", None),
    }
    payload["review_runtime"] = resolve_review_runtime_config(payload)
    return payload


def write_run_reports(
    output_dir: Path,
    rows: List[Dict[str, object]],
    elapsed_seconds: Optional[float] = None,
    workflow_profile: str = "podcast",
):
    output_write_batch_report_md(
        output_dir / "_batch_report.md",
        rows,
        elapsed_seconds=elapsed_seconds,
    )
    output_write_review_run_report(output_dir, rows, elapsed_seconds=elapsed_seconds)
    output_write_speaker_workflow_report(output_dir, rows)
    if workflow_profile == "anonymous_meeting":
        evidence_report = {
            "workflow_version": 1,
            "view": "anonymous_meeting_disabled",
            "row_count": 0,
            "rows": [],
            "recurring_unknown_speakers": [],
            "changed_count": 0,
            "identity_basis": "no speaker identity evidence collected",
            "similarity_threshold": None,
        }
    else:
        evidence_report = build_cross_episode_speaker_view(output_dir)
    (output_dir / "_speaker_workflow_report.json").write_text(
        json.dumps(evidence_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Speaker Workflow Report",
        "",
        f"- Identity basis: {evidence_report['identity_basis']}",
        f"- Evidence rows: {evidence_report['row_count']}",
        f"- Recurring embedding-backed candidates: {len(evidence_report['recurring_unknown_speakers'])}",
        "",
    ]
    for candidate in evidence_report["recurring_unknown_speakers"]:
        lines.append(
            f"- {candidate['candidate_id']}: episodes={candidate['episode_count']}, "
            f"duration={candidate['total_duration_seconds']:.1f}s, "
            f"promotion_eligible={candidate['promotion_eligible']}"
        )
    (output_dir / "_speaker_workflow_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_final_review_summary(rows: List[Dict[str, object]]):
    review_attempted = sum(1 for row in rows if coerce_bool(row.get("review_attempted"), False))
    material_changes = sum(1 for row in rows if coerce_bool(row.get("review_material_change"), False))
    unresolved_risk = sum(
        1
        for row in rows
        if coerce_bool(row.get("speaker_drift_flag"), False)
        or coerce_bool(row.get("recurring_unnamed_speaker_flag"), False)
        or coerce_bool(row.get("host_profile_stability_flag"), False)
    )
    print("Review run summary")
    print(f"  review attempted: {review_attempted}/{len(rows)}")
    print(f"  episodes with material review changes: {material_changes}")
    print(f"  unresolved speaker-risk episodes: {unresolved_risk}")


def correction_path_for_audio(corrections_dir: Optional[str], audio_path: Path) -> Optional[Path]:
    base_dir = Path(corrections_dir) if corrections_dir else Path("corrections")
    path = base_dir / f"{audio_path.stem}_corrections.csv"
    return path if path.exists() else None


def apply_manual_corrections(segments: List[SegmentItem], correction_path: Optional[Path]) -> int:
    if correction_path is None:
        return 0

    by_id = {str(segment.id): segment for segment in segments}
    applied = 0
    with correction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            segment_id = (row.get("segment_id") or row.get("id") or "").strip()
            if not segment_id or segment_id not in by_id:
                continue
            segment = by_id[segment_id]
            corrected_text = (row.get("corrected_text") or row.get("text") or "").strip()
            corrected_speaker = (row.get("speaker") or "").strip()
            changed = False
            if corrected_text and corrected_text != segment.text:
                segment.original_text = getattr(segment, "original_text", segment.text)
                segment.text = corrected_text
                segment.manual_correction_applied = True
                changed = True
            if corrected_speaker and corrected_speaker != segment.speaker:
                segment.original_speaker = segment.speaker
                segment.speaker = corrected_speaker
                segment.manual_correction_applied = True
                changed = True
            if changed:
                applied += 1
    return applied


def build_historical_similarity_scores(rows: List[Dict[str, object]]) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        speaker = str(row.get("host_label") or "").strip()
        score = row.get("top_host_similarity")
        if speaker and score not in ("", None):
            history[speaker].append(float(score))
    return dict(history)


def run_speaker_attribution_stage(
    *, output_dir: Path, audio_path: Path, segments: List[SegmentItem], diarized_turns: List[Dict[str, object]],
    verifier: Any, speaker_embedding_identity: ProviderIdentity, host_reference: Optional[str],
    host_profile_path: Optional[str], known_speaker_profiles: Dict[str, Dict[str, object]], host_threshold: float,
    assume_dominant: bool, max_embedding_seconds: float, min_host_seconds: float,
    historical_similarity_scores: Dict[str, List[float]], alignment_fingerprint: Dict[str, object],
    diarization_fingerprint: Dict[str, object], resume_intermediates: bool,
    speaker_telemetry: Optional[Dict[str, object]] = None,
    component_timings: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[str], Optional[np.ndarray], Dict[str, float], Dict[str, float], Dict[str, Dict[str, object]], Dict[str, str], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object], bool]:
    def file_revision(value: Optional[str]) -> Dict[str, object]:
        if not value:
            return {}
        path = Path(value)
        if not path.exists() or not path.is_file():
            return {"path": str(path), "missing": True}
        stat = path.stat()
        return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    known_profile_sources = {
        name: [file_revision(str(path)) for path in profile.get("sample_files") or []]
        for name, profile in sorted(known_speaker_profiles.items())
    }
    def build_fingerprint() -> Dict[str, object]:
        return build_stage_fingerprint(
            "speaker_attribution", speaker_embedding_identity,
            {"host_threshold": host_threshold, "assume_dominant": assume_dominant, "max_embedding_seconds": max_embedding_seconds,
             "min_host_seconds": min_host_seconds, "known_profile_sources": known_profile_sources,
             "host_reference": file_revision(host_reference), "host_profile": file_revision(host_profile_path)},
            [alignment_fingerprint, diarization_fingerprint],
        )

    fingerprint = build_fingerprint()
    cached = load_segment_stage_artifact(output_dir, audio_path, "speaker_attribution", fingerprint) if resume_intermediates else None
    if cached:
        cached_metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
        if not cached_metadata.get("speaker_identity_evidence"):
            print("  speaker attribution: legacy cache lacks identity evidence; rebuilding")
            cached = None
    if cached:
        print("  speaker attribution: reused cached artifact")
        if speaker_telemetry is not None:
            speaker_telemetry["audio_cache_mode"] = "stage_reused"
        segments[:] = cached["segments"]
        metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
        return (metadata.get("host_speaker"), None, dict(metadata.get("durations") or {}),
                dict(metadata.get("similarity_scores") or {}), dict(metadata.get("known_assignments") or {}),
                dict(metadata.get("speaker_mapping") or {}), list(metadata.get("drift_alerts") or []),
                list(metadata.get("speaker_identity_evidence") or []), fingerprint, True)
    speaker_audio_path = prepare_speaker_audio_cache(
        audio_path,
        output_dir,
        telemetry=speaker_telemetry,
        component_timings=component_timings,
    )
    existing_profile = load_host_profile(host_profile_path, speaker_embedding_identity)
    host_speaker, speaker_embeddings, updated_profile, durations, similarity_scores = choose_host_speaker(
        verifier=verifier, audio_path=str(speaker_audio_path), diarized_turns=diarized_turns, host_reference_path=host_reference,
        existing_profile=existing_profile, host_threshold=host_threshold, assume_dominant=assume_dominant,
        max_embedding_seconds=max_embedding_seconds, min_host_seconds=min_host_seconds,
        telemetry=speaker_telemetry,
    )
    known_assignments = match_known_speakers(speaker_embeddings, known_speaker_profiles, host_threshold)
    known_host = next((speaker_id for speaker_id, assignment in known_assignments.items() if assignment.get("is_host")), None)
    if known_host:
        host_speaker = known_host
    updated_profile = final_host_profile_update(existing_profile, speaker_embeddings, host_speaker, updated_profile)
    speaker_mapping = rename_speakers(segments, diarized_turns, host_speaker, durations, known_assignments=known_assignments)
    current_scores = {speaker_mapping.get(speaker_id, speaker_id): score for speaker_id, score in similarity_scores.items()}
    drift_alerts = detect_speaker_similarity_drift(current_scores, historical_similarity_scores)
    speaker_identity_evidence = []
    for local_speaker, embedding in sorted(speaker_embeddings.items()):
        spans = sorted(
            [
                {
                    "start": round(float(turn.get("start") or 0.0), 3),
                    "end": round(float(turn.get("end") or 0.0), 3),
                }
                for turn in diarized_turns
                if str(turn.get("speaker") or "") == str(local_speaker)
            ],
            key=lambda item: (-(item["end"] - item["start"]), item["start"]),
        )[:5]
        evidence_identity = {
            "episode_id": audio_path.stem,
            "local_speaker": local_speaker,
            "embedding_family": f"{speaker_embedding_identity.provider}:{speaker_embedding_identity.model}",
            "spans": spans,
        }
        speaker_identity_evidence.append(
            {
                "evidence_version": 1,
                "evidence_id": f"speaker_evidence_{hashlib.sha256(json.dumps(evidence_identity, sort_keys=True).encode('utf-8')).hexdigest()}",
                "episode_id": audio_path.stem,
                "source_audio": str(audio_path),
                "local_speaker": local_speaker,
                "assigned_label": speaker_mapping.get(local_speaker, local_speaker),
                "known_speaker_assignment": known_assignments.get(local_speaker) or {},
                "is_host": local_speaker == host_speaker,
                "duration_seconds": round(float(durations.get(local_speaker) or 0.0), 3),
                "quality_score": round(min(1.0, float(durations.get(local_speaker) or 0.0) / 60.0), 4),
                "embedding_family": f"{speaker_embedding_identity.provider}:{speaker_embedding_identity.model}",
                "embedding_provider": speaker_embedding_identity.to_payload(),
                "embedding_dimension": int(embedding.size),
                "embedding": [round(float(value), 8) for value in embedding.reshape(-1).tolist()],
                "spans": spans,
            }
        )
    if updated_profile is not None and host_speaker is not None:
        save_host_profile(host_profile_path, updated_profile, str(audio_path), speaker_embedding_identity)
        fingerprint = build_fingerprint()
    save_segment_stage_artifact(
        output_dir, audio_path, "speaker_attribution", segments,
        {"host_speaker": host_speaker, "durations": durations, "similarity_scores": similarity_scores,
         "known_assignments": known_assignments, "speaker_mapping": speaker_mapping, "drift_alerts": drift_alerts,
         "speaker_identity_evidence": speaker_identity_evidence},
        fingerprint, [alignment_fingerprint, diarization_fingerprint],
    )
    return host_speaker, updated_profile, durations, similarity_scores, known_assignments, speaker_mapping, drift_alerts, speaker_identity_evidence, fingerprint, False


def run_anonymous_speaker_attribution_stage(
    *, output_dir: Path, audio_path: Path, segments: List[SegmentItem], diarized_turns: List[Dict[str, object]],
    alignment_fingerprint: Dict[str, object], diarization_fingerprint: Dict[str, object], resume_intermediates: bool,
    speaker_telemetry: Optional[Dict[str, object]] = None,
) -> Tuple[None, None, Dict[str, float], Dict[str, float], Dict[str, Dict[str, object]], Dict[str, str], List[Dict[str, object]], List[Dict[str, object]], Dict[str, object], bool]:
    """Map diarization labels without computing or persisting speaker identity."""

    fingerprint = build_stage_fingerprint(
        "speaker_attribution", ANONYMOUS_SPEAKER_IDENTITY,
        {"workflow_profile": "anonymous_meeting", "identity_mode": "diarization_labels_only"},
        [alignment_fingerprint, diarization_fingerprint],
    )
    cached = load_segment_stage_artifact(output_dir, audio_path, "speaker_attribution", fingerprint) if resume_intermediates else None
    if cached:
        metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
        segments[:] = cached["segments"]
        return None, None, dict(metadata.get("durations") or {}), {}, {}, dict(metadata.get("speaker_mapping") or {}), [], [], fingerprint, True
    durations = speaker_durations(diarized_turns)
    speaker_mapping = rename_speakers(segments, diarized_turns, None, durations, known_assignments={})
    metadata = {
        "host_speaker": None,
        "durations": durations,
        "similarity_scores": {},
        "known_assignments": {},
        "speaker_mapping": speaker_mapping,
        "drift_alerts": [],
        "speaker_identity_evidence": [],
        "workflow_profile": "anonymous_meeting",
    }
    save_segment_stage_artifact(
        output_dir, audio_path, "speaker_attribution", segments, metadata,
        fingerprint, [alignment_fingerprint, diarization_fingerprint],
    )
    return None, None, durations, {}, {}, speaker_mapping, [], [], fingerprint, False


def run_deterministic_cleanup_stage(
    *, output_dir: Path, audio_path: Path, segments: List[SegmentItem], replacement_map: Dict[str, List[str]],
    correction_path: Optional[Path], cleanup_level: str, speaker_attribution_fingerprint: Dict[str, object],
    resume_intermediates: bool,
) -> Tuple[List[SegmentItem], List[SegmentItem], List[Dict[str, object]], int, List[object], Dict[str, object], bool]:
    correction_revision = ""
    if correction_path and Path(correction_path).exists():
        correction_revision = hashlib.sha256(Path(correction_path).read_bytes()).hexdigest()
    cleanup_identity = ProviderIdentity(stage="deterministic_cleanup", provider="builtin", model="cleanup_rules", version="2")
    fingerprint = build_stage_fingerprint(
        "deterministic_cleanup", cleanup_identity,
        {"cleanup_level": cleanup_level, "replacement_map": replacement_map, "correction_revision": correction_revision},
        [speaker_attribution_fingerprint],
    )
    cached = load_segment_stage_artifact(output_dir, audio_path, "deterministic_cleanup", fingerprint) if resume_intermediates else None
    if cached:
        print("  deterministic cleanup: reused cached artifact")
        metadata = cached.get("metadata") if isinstance(cached.get("metadata"), dict) else {}
        normalized = [segment_from_payload(item) for item in metadata.get("normalized_segments") or [] if isinstance(item, dict)]
        cleaned = cached["segments"]
        if not normalized:
            normalized = [segment_from_payload(segment_to_payload(item)) for item in cleaned]
        return (
            normalized, cleaned, list(metadata.get("replacement_events") or []),
            int(metadata.get("manual_corrections") or 0), list(metadata.get("cleanup_edits") or []), fingerprint, True,
        )
    normalized, replacement_events = coalesce_segments(segments, replacement_map)
    manual_corrections = apply_manual_corrections(normalized, correction_path)
    if correction_path:
        print(f"  manual corrections applied: {manual_corrections} from {correction_path}")
    cleaned, cleanup_edits = build_cleaned_segments(normalized, level=cleanup_level)
    save_segment_stage_artifact(
        output_dir, audio_path, "deterministic_cleanup", cleaned,
        {"replacement_events": replacement_events, "manual_corrections": manual_corrections,
         "cleanup_edits": cleanup_edits, "normalized_segments": [segment_to_payload(item) for item in normalized]},
        fingerprint, [speaker_attribution_fingerprint],
    )
    return normalized, cleaned, replacement_events, manual_corrections, cleanup_edits, fingerprint, False


def process_file(
    audio_path: Path,
    output_dir: Path,
    asr_provider,
    alignment_provider,
    diarization_pipeline: Pipeline,
    verifier: Any,
    speaker_embedding_identity: ProviderIdentity,
    language: str,
    beam_size: int,
    batch_size: int,
    initial_prompt: Optional[str],
    hotwords: Optional[str],
    replacement_map: Dict[str, List[str]],
    host_reference: Optional[str],
    host_profile_path: Optional[str],
    known_speaker_profiles: Dict[str, Dict[str, object]],
    host_threshold: float,
    assume_dominant: bool,
    max_embedding_seconds: float,
    min_host_seconds: float,
    num_speakers: Optional[int],
    cleanup_level: str = "normal",
    corrections_dir: Optional[str] = None,
    runtime_config: Optional[Dict[str, object]] = None,
    review_calibration_session: Optional[ReviewCalibrationSession] = None,
    historical_similarity_scores: Optional[Dict[str, List[float]]] = None,
    resume_intermediates: bool = True,
    archive_debug_artifacts: bool = False,
) -> Dict[str, object]:
    """Process one audio file through all model stages and write its output bundle."""

    file_started = time.perf_counter()
    RESOURCE_USAGE_TRACKER.clear()
    stage_timings: Dict[str, float] = {}
    speaker_telemetry = new_speaker_telemetry(
        output_dir / CHECKPOINT_DIRNAME / f"{audio_path.stem}_speaker_telemetry.json"
    )
    print(f"Processing {audio_path.name}")
    print_episode_mode("tier1+tier2" if resolve_review_runtime_config(runtime_config or {}).get("any_review_enabled") else "tier1-only")
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_checkpoint = load_processing_checkpoint(output_dir, audio_path)
    if prior_checkpoint:
        print(
            f"  resuming from durable checkpoint: "
            f"{str(prior_checkpoint.get('stage') or 'unknown').replace('_', ' ')}"
        )
    log_memory_usage("before_transcription")

    transcription_started = time.perf_counter()
    print_episode_stage(1, 6, "transcription")
    segments, info_payload, transcription_reused = run_transcription_stage(
        output_dir=output_dir,
        audio_path=audio_path,
        asr_provider=asr_provider,
        language=language,
        beam_size=beam_size,
        batch_size=batch_size,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        resume_intermediates=resume_intermediates,
    )
    print(
        f"  transcription complete: {len(segments)} raw segments "
        f"in {time.perf_counter() - transcription_started:.1f}s"
    )
    stage_timings["transcription"] = time.perf_counter() - transcription_started
    write_processing_checkpoint(
        output_dir,
        audio_path,
        "transcription_complete",
        {
            "segment_count": len(segments),
            "duration_seconds": info_payload.get("duration"),
        },
    )
    log_memory_usage("after_transcription")

    alignment_started = time.perf_counter()
    print_episode_stage(2, 6, "alignment")
    transcription_fingerprint = info_payload.get("stage_fingerprint")
    if not isinstance(transcription_fingerprint, dict) or not transcription_fingerprint:
        transcription_fingerprint = build_stage_fingerprint(
            "transcription",
            asr_provider.identity,
            {
                "language": language,
                "beam_size": beam_size,
                "batch_size": batch_size,
                "initial_prompt": initial_prompt or "",
                "hotwords": hotwords or "",
            },
        )
    segments, alignment_metadata, alignment_reused, alignment_fingerprint = run_alignment_stage(
        output_dir=output_dir,
        audio_path=audio_path,
        segments=segments,
        alignment_provider=alignment_provider,
        language=language,
        transcription_fingerprint=transcription_fingerprint,
        resume_intermediates=resume_intermediates,
    )
    stage_timings["alignment"] = time.perf_counter() - alignment_started
    print(
        f"  alignment complete: {sum(len(segment.words) for segment in segments)} words "
        f"in {stage_timings['alignment']:.1f}s"
    )
    write_processing_checkpoint(
        output_dir,
        audio_path,
        "alignment_complete",
        {"word_count": sum(len(segment.words) for segment in segments), "segment_count": len(segments)},
    )

    diarization_started = time.perf_counter()
    print_episode_stage(3, 6, "diarization")
    diarized_turns, diarization_reused, diarization_metadata = run_diarization_stage(
        output_dir=output_dir,
        audio_path=audio_path,
        diarization_pipeline=diarization_pipeline,
        diarization_model_id=str((runtime_config or {}).get("diarization_model") or ""),
        diarization_model_revision=str((runtime_config or {}).get("diarization_model_revision") or ""),
        verifier=verifier,
        num_speakers=num_speakers,
        max_embedding_seconds=max_embedding_seconds,
        resume_intermediates=resume_intermediates,
        speaker_telemetry=speaker_telemetry,
    )
    assign_speakers_to_segments(segments, diarized_turns)
    print(
        f"  diarization complete: {len(diarized_turns)} turns "
        f"in {time.perf_counter() - diarization_started:.1f}s"
    )
    stage_timings["diarization"] = time.perf_counter() - diarization_started
    write_processing_checkpoint(
        output_dir,
        audio_path,
        "diarization_complete",
        {
            "turn_count": len(diarized_turns),
            "segment_count": len(segments),
        },
    )
    log_memory_usage("after_diarization")

    anonymous_meeting = str((runtime_config or {}).get("workflow_profile") or "podcast") == "anonymous_meeting"
    print_episode_stage(4, 6, "speaker labeling" if anonymous_meeting else "speaker matching")
    matching_started = time.perf_counter()
    matching_component_timings: Dict[str, float] = {}
    print("  speaker telemetry: collecting audio spans and embeddings")
    if anonymous_meeting:
        with timed_component(matching_component_timings, "speaker label assignment"):
            host_speaker, updated_profile, durations, similarity_scores, known_assignments, speaker_mapping, drift_alerts, speaker_identity_evidence, speaker_attribution_fingerprint, speaker_attribution_reused = run_anonymous_speaker_attribution_stage(
                output_dir=output_dir, audio_path=audio_path, segments=segments, diarized_turns=diarized_turns,
                alignment_fingerprint=alignment_fingerprint,
                diarization_fingerprint=diarization_metadata.get("stage_fingerprint", {}), resume_intermediates=resume_intermediates,
                speaker_telemetry=speaker_telemetry,
            )
    else:
        host_speaker, updated_profile, durations, similarity_scores, known_assignments, speaker_mapping, drift_alerts, speaker_identity_evidence, speaker_attribution_fingerprint, speaker_attribution_reused = run_speaker_attribution_stage(
            output_dir=output_dir, audio_path=audio_path, segments=segments, diarized_turns=diarized_turns,
            verifier=verifier, speaker_embedding_identity=speaker_embedding_identity, host_reference=host_reference,
            host_profile_path=host_profile_path, known_speaker_profiles=known_speaker_profiles, host_threshold=host_threshold,
            assume_dominant=assume_dominant, max_embedding_seconds=max_embedding_seconds, min_host_seconds=min_host_seconds,
            historical_similarity_scores=historical_similarity_scores or {}, alignment_fingerprint=alignment_fingerprint,
            diarization_fingerprint=diarization_metadata.get("stage_fingerprint", {}), resume_intermediates=resume_intermediates,
            speaker_telemetry=speaker_telemetry,
            component_timings=matching_component_timings,
        )
    filename_date_config = (runtime_config or {}).get("filename_date", {})
    episode_metadata_for_review = build_episode_metadata(str(audio_path), filename_date_config)
    correction_path = correction_path_for_audio(corrections_dir, audio_path)
    with timed_component(matching_component_timings, "deterministic cleanup"):
        normalized_segments, cleaned_segments, replacement_events, manual_corrections, cleanup_edits, cleanup_fingerprint, cleanup_reused = run_deterministic_cleanup_stage(
            output_dir=output_dir, audio_path=audio_path, segments=segments, replacement_map=replacement_map,
            correction_path=correction_path, cleanup_level=cleanup_level,
            speaker_attribution_fingerprint=speaker_attribution_fingerprint, resume_intermediates=resume_intermediates,
        )
    resolved_host_label = speaker_mapping.get(host_speaker, "HOST") if host_speaker else "HOST"
    host_output_labels = {
        resolved_host_label,
        "HOST",
        *[
            speaker_mapping.get(speaker_id, speaker_id)
            for speaker_id, assignment in known_assignments.items()
            if assignment.get("is_host")
            or any(str(role).lower() in {"host", "co-host"} for role in assignment.get("roles") or [])
        ],
    }
    review_rows = collect_review_rows(
        source_file=str(audio_path),
        segments=normalized_segments,
        replacement_events=replacement_events,
        host_speaker=host_speaker,
        host_threshold=host_threshold,
        durations=durations,
        similarity_scores=similarity_scores,
        speaker_mapping=speaker_mapping,
        host_output_labels=host_output_labels,
        episode_metadata=episode_metadata_for_review,
    )
    for alert in drift_alerts:
        review_rows.append(
            {
                "issue_type": "speaker_similarity_drift",
                "speaker": alert["speaker"],
                "start": "",
                "end": "",
                "score": alert["current_similarity"],
                "details": alert["review_reason"],
                "text": (
                    f"Current similarity {alert['current_similarity']} is below historical "
                    f"average {alert['historical_average_similarity']} by {alert['drop']}."
                ),
                "source_file": str(audio_path),
                "episode_date": episode_metadata_for_review.get("episode_date", ""),
                "episode_date_compact": episode_metadata_for_review.get("episode_date_compact", ""),
                "episode_sort_key": episode_metadata_for_review.get("episode_sort_key", ""),
            }
        )
    print(
        f"  speaker matching complete: {len(speaker_mapping)} labeled speakers, "
        f"{len(review_rows)} review rows in {time.perf_counter() - matching_started:.1f}s"
    )
    stage_timings["speaker_matching"] = time.perf_counter() - matching_started
    print_timing_component_summary("speaker matching", matching_component_timings)
    write_processing_checkpoint(
        output_dir,
        audio_path,
        "speaker_matching_complete",
        {
            "labeled_speakers": len(speaker_mapping),
            "review_rows": len(review_rows),
        },
    )
    log_memory_usage("after_speaker_matching")

    runtime_review_config = resolve_review_runtime_config(runtime_config or {})
    review_started = time.perf_counter()
    review_component_timings: Dict[str, float] = {}
    print_episode_stage(5, 6, "review")
    review_debug_dir = review_debug_directory(
        runtime_review_config,
        {
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "review_input_source": "inline_cleaned_segments",
        },
    )
    if review_debug_dir is not None:
        print(f"  review debug output: {review_debug_dir}")
    review_result = review_segments(
        cleaned_segments,
        runtime_review_config,
        review_input_source="inline_cleaned_segments",
        calibration_session=review_calibration_session,
        progress_callback=make_checkpointed_review_progress_callback(
            output_dir, audio_path, review_component_timings
        ),
        debug_context={
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "review_input_source": "inline_cleaned_segments",
        },
    )
    stage_timings["review"] = time.perf_counter() - review_started
    print(f"  review complete in {stage_timings['review']:.1f}s")
    print_timing_component_summary("review", review_component_timings)
    write_processing_checkpoint(
        output_dir,
        audio_path,
        "review_complete",
        {
            "review_status": review_result.get("metadata", {}).get("review_status", ""),
            "candidate_count": review_result.get("metadata", {}).get("review_candidate_count", 0),
            "corrected_segment_count": review_result.get("metadata", {}).get("corrected_segment_count", 0),
        },
    )
    print_episode_stage(6, 6, "writing outputs")
    writing_started = time.perf_counter()
    writing_component_timings: Dict[str, float] = {}
    base_name = audio_path.stem
    filename_date_config = (runtime_config or {}).get("filename_date", {})
    stage_provenance = {
        "transcription": {"provider": asr_provider.identity.to_payload(), "fingerprint": transcription_fingerprint},
        "alignment": {
            "provider": alignment_provider.identity.to_payload(),
            "fingerprint": alignment_fingerprint,
            **alignment_metadata,
        },
        "diarization": {
            "provider": diarization_metadata.get("provider", {}),
            "fingerprint": diarization_metadata.get("stage_fingerprint", {}),
        },
        "speaker_embedding": {"provider": speaker_embedding_identity.to_payload()},
        "speaker_attribution": {"provider": speaker_embedding_identity.to_payload(), "fingerprint": speaker_attribution_fingerprint},
        "deterministic_cleanup": {"provider": {"stage": "deterministic_cleanup", "provider": "builtin", "model": "cleanup_rules", "version": "2"}, "fingerprint": cleanup_fingerprint},
    }
    episode_metadata = {
        **build_episode_metadata(str(audio_path), filename_date_config),
        "workflow_profile": str((runtime_config or {}).get("workflow_profile") or "podcast"),
        "episode_uid": stable_episode_uid(str(audio_path), audio_file_fingerprint(audio_path)),
        "stage_provenance": stage_provenance,
        "speaker_identity_evidence": speaker_identity_evidence,
        "correction_lineage": load_correction_lineage(output_dir, audio_path.stem),
    }
    cleaned_metadata = {**episode_metadata, "text_version": "cleaned"}
    with timed_component(writing_component_timings, "transcript text outputs"):
        output_write_text_transcript(
            output_dir / f"{base_name}_speaker_transcript.txt",
            normalized_segments,
            format_timestamp,
            host_only=False,
            metadata=episode_metadata,
        )
        output_write_text_transcript(
            output_dir / f"{base_name}_host_only.txt",
            normalized_segments,
            format_timestamp,
            host_only=True,
            host_labels=host_output_labels,
            metadata=episode_metadata,
        )
        output_write_text_transcript(
            output_dir / f"{base_name}_cleaned_speaker_transcript.txt",
            cleaned_segments,
            format_timestamp,
            host_only=False,
            metadata=cleaned_metadata,
        )
        output_write_text_transcript(
            output_dir / f"{base_name}_cleaned_host_only.txt",
            cleaned_segments,
            format_timestamp,
            host_only=True,
            host_labels=host_output_labels,
            metadata=cleaned_metadata,
        )
    reviewed_segments = review_result["segments"]
    review_metadata = review_result["metadata"]
    reviewed_output_written = bool(reviewed_segments)
    with timed_component(writing_component_timings, "reviewed output bundle"):
        reviewed_output_paths = write_reviewed_output_bundle(
            audio_path=audio_path,
            output_dir=output_dir,
            reviewed_segments=reviewed_segments,
            review_metadata=review_metadata,
            host_output_labels=host_output_labels,
            episode_metadata=episode_metadata,
            info_payload=info_payload,
            diarized_turns=diarized_turns,
            speaker_mapping=speaker_mapping,
            host_speaker=host_speaker,
            durations=durations,
            known_assignments=known_assignments,
            runtime_config=runtime_config,
        )
    with timed_component(writing_component_timings, "review CSV outputs"):
        output_write_review_csv(output_dir / f"{base_name}_review.csv", review_rows)
        speaker_review_path = output_dir / f"{base_name}_speaker_identity_review.csv"
        output_write_speaker_identity_review_csv(
            speaker_review_path,
            speaker_mapping=speaker_mapping,
            durations=durations,
            similarity_scores=similarity_scores,
            known_assignments=known_assignments,
            host_speaker=host_speaker,
        )
    with timed_component(writing_component_timings, "transcript JSON outputs"):
        output_write_json_output(
            output_dir / f"{base_name}_speaker_transcript.json",
            source_file=str(audio_path),
            info_payload=info_payload,
            diarized_turns=diarized_turns,
            segments=normalized_segments,
            speaker_mapping=speaker_mapping,
            host_speaker=host_speaker,
            durations=durations,
            known_assignments=known_assignments,
            metadata=episode_metadata,
            pipeline_version=runtime_config.get("model", "") if runtime_config else "",
        )
        output_write_json_output(
            output_dir / f"{base_name}_cleaned_speaker_transcript.json",
            source_file=str(audio_path),
            info_payload=info_payload,
            diarized_turns=diarized_turns,
            segments=cleaned_segments,
            speaker_mapping=speaker_mapping,
            host_speaker=host_speaker,
            durations=durations,
            known_assignments=known_assignments,
            metadata=cleaned_metadata,
            text_version="cleaned",
            pipeline_version=runtime_config.get("model", "") if runtime_config else "",
        )
    stage_timings["writing"] = time.perf_counter() - writing_started
    print(f"  writing complete in {stage_timings['writing']:.1f}s")
    print_timing_component_summary("writing outputs", writing_component_timings)
    log_memory_usage("after_writing")

    total_segments = len(normalized_segments)
    host_segments = sum(1 for segment in normalized_segments if segment.speaker in host_output_labels)
    print(f"  review rows: {len(review_rows)}")
    print(f"  speaker segments: {total_segments}")
    print(f"  host segments: {host_segments}")
    print(f"  cleaned text edits: {len(cleanup_edits)}")
    print(f"  manual corrections: {manual_corrections}")
    print(f"  host detected: {host_speaker is not None}")
    speaker_telemetry_payload = finalize_speaker_telemetry(speaker_telemetry)
    print(
        "  speaker telemetry summary: "
        f"audio_cache={speaker_telemetry_payload['audio_cache_mode']}, "
        f"audio_cache_wall={speaker_telemetry_payload['audio_cache_conversion_wall_seconds']:.1f}s, "
        f"audio_reads={speaker_telemetry_payload['audio_span_read_count']}, "
        f"audio_wall={speaker_telemetry_payload['audio_span_wall_seconds']:.1f}s, "
        f"embedding_calls={speaker_telemetry_payload['embedding_call_count']}, "
        f"embedding_wall={speaker_telemetry_payload['embedding_wall_seconds']:.1f}s, "
        f"embedding_input={speaker_telemetry_payload['embedding_input_seconds']:.1f}s"
    )
    finalization_started = time.perf_counter()
    summary_operation_started = start_finalization_operation("building episode summary")
    summary_row = build_episode_summary_row(
        audio_path=audio_path,
        normalized_segments=normalized_segments,
        review_rows=review_rows,
        host_speaker=host_speaker,
        durations=durations,
        similarity_scores=similarity_scores,
        speaker_mapping=speaker_mapping,
        known_assignments=known_assignments,
        episode_metadata=episode_metadata,
    )
    finish_finalization_operation("building episode summary", summary_operation_started)
    summary_row["cleanup_level"] = cleanup_level
    summary_row["cleanup_edit_count"] = len(cleanup_edits)
    summary_row["manual_correction_count"] = manual_corrections
    summary_row["processing_seconds"] = round(time.perf_counter() - file_started, 2)
    summary_row["transcription_artifact_reused"] = transcription_reused
    summary_row["alignment_artifact_reused"] = alignment_reused
    summary_row["asr_provider"] = asr_provider.identity.provider
    summary_row["alignment_provider"] = alignment_provider.identity.provider
    summary_row["speaker_embedding_provider"] = speaker_embedding_identity.provider
    summary_row["speaker_attribution_artifact_reused"] = speaker_attribution_reused
    summary_row["deterministic_cleanup_artifact_reused"] = cleanup_reused
    summary_row["diarization_artifact_reused"] = diarization_reused
    warnings_for_language = language_model_warnings(info_payload, language)
    summary_row["language_model_warnings"] = "; ".join(warnings_for_language)
    apply_review_metadata_to_summary(summary_row, review_result)
    apply_speaker_risk_flags_to_summary(summary_row, speaker_mapping)
    summary_row["processing_mode"] = "tier1+tier2" if resolve_review_runtime_config(runtime_config or {}).get("any_review_enabled") else "tier1-only"
    summary_row["tier1_reused_from_existing"] = False
    summary_row["review_backfilled_from_cleaned_json"] = False
    summary_row["episode_contract_version"] = EPISODE_CONTRACT_V2
    summary_row["contract_upgrade_method"] = "native_v2_processing"
    summary_row["diarization_mode"] = str(diarization_metadata.get("mode") or "")
    summary_row["diarization_probe_attempted"] = bool(diarization_metadata.get("probe"))
    summary_row["diarization_learned_route"] = bool(diarization_metadata.get("learned_route"))
    summary_row["diarization_route_reason"] = str(diarization_metadata.get("reason") or "")
    summary_row["diarization_failure_floor_seconds"] = int(float(diarization_metadata.get("failure_floor_seconds") or 0.0))
    summary_row["diarization_safe_success_ceiling_seconds"] = int(float(diarization_metadata.get("safe_success_ceiling_seconds") or 0.0))
    summary_row["diarization_chunk_count"] = int(diarization_metadata.get("chunk_count") or 0)
    summary_row["diarization_chunk_overlap_seconds"] = int(float(diarization_metadata.get("chunk_overlap_seconds") or 0.0))
    summary_row["diarization_reconciliation_merge_count"] = int(diarization_metadata.get("reconciliation_merge_count") or 0)
    summary_row["diarization_reconciliation_ambiguous_count"] = int(diarization_metadata.get("reconciliation_ambiguous_count") or 0)
    summary_row["speaker_audio_span_read_count"] = int(speaker_telemetry_payload.get("audio_span_read_count") or 0)
    summary_row["speaker_audio_span_wall_seconds"] = float(speaker_telemetry_payload.get("audio_span_wall_seconds") or 0.0)
    summary_row["speaker_embedding_call_count"] = int(speaker_telemetry_payload.get("embedding_call_count") or 0)
    summary_row["speaker_embedding_wall_seconds"] = float(speaker_telemetry_payload.get("embedding_wall_seconds") or 0.0)
    summary_row["speaker_embedding_input_seconds"] = float(speaker_telemetry_payload.get("embedding_input_seconds") or 0.0)
    summary_row["speaker_embedding_calls_by_kind"] = json.dumps(
        speaker_telemetry_payload.get("embedding_calls_by_kind") or {},
        sort_keys=True,
    )
    summary_row["speaker_reconciliation_boundary_count"] = int(
        speaker_telemetry_payload.get("chunk_reconciliation_boundary_count") or 0
    )
    stage_timings["total"] = time.perf_counter() - file_started
    outputs = [
        output_dir / f"{base_name}_speaker_transcript.txt",
        output_dir / f"{base_name}_host_only.txt",
        output_dir / f"{base_name}_cleaned_speaker_transcript.txt",
        output_dir / f"{base_name}_cleaned_host_only.txt",
        output_dir / f"{base_name}_review.csv",
        speaker_review_path,
        output_dir / f"{base_name}_speaker_transcript.json",
        output_dir / f"{base_name}_cleaned_speaker_transcript.json",
    ]
    outputs.extend(reviewed_output_paths)
    manifest_operation_started = start_finalization_operation(
        f"writing output manifest and hashing {len(outputs)} outputs"
    )
    output_write_output_manifest(
        output_dir / f"{base_name}_manifest.json",
        source_file=str(audio_path),
        source_fingerprint=audio_file_fingerprint(audio_path),
        config=runtime_config or {},
        outputs=outputs,
        timings=stage_timings,
        summary=summary_row,
        stage_provenance=stage_provenance,
        resource_usage=dict(RESOURCE_USAGE_TRACKER),
        speaker_telemetry=speaker_telemetry_payload,
        progress_callback=lambda message: print(f"    finalization: {message}"),
    )
    finish_finalization_operation("writing output manifest", manifest_operation_started)
    checkpoint_operation_started = start_finalization_operation("clearing processing checkpoint")
    clear_processing_checkpoint(output_dir, audio_path)
    finish_finalization_operation("clearing processing checkpoint", checkpoint_operation_started)
    if not archive_debug_artifacts:
        cleanup_label = "clearing debug artifacts" if resume_intermediates else "clearing stage artifacts"
        cleanup_operation_started = start_finalization_operation(cleanup_label)
        if resume_intermediates:
            state_clear_debug_artifacts(output_dir, audio_path)
        else:
            state_clear_stage_artifacts(output_dir, audio_path)
        finish_finalization_operation(cleanup_label, cleanup_operation_started)
    print(f"  finalization complete in {time.perf_counter() - finalization_started:.1f}s")
    return summary_row


def discover_audio_files(input_dir: Path, input_file: Optional[str]) -> List[Path]:
    if input_file:
        candidate = Path(input_file)
        if not candidate.is_absolute():
            candidate = input_dir / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Input file not found: {candidate}")
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise RuntimeError(f"Input file is not a supported audio file: {candidate}")
        return [candidate]

    return sorted(
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def disk_space_preflight(output_dir: Path, audio_files: List[Path]):
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    input_bytes = sum(path.stat().st_size for path in audio_files if path.exists())
    recommended_free = max(input_bytes * 2, 5 * 1024 * 1024 * 1024)
    if usage.free < recommended_free:
        print(
            "Disk space warning: output drive has "
            f"{usage.free / (1024 ** 3):.1f} GiB free; recommended at least "
            f"{recommended_free / (1024 ** 3):.1f} GiB for this batch."
        )


def audio_duration_map(audio_files: List[Path]) -> Dict[str, Optional[float]]:
    return {str(path): get_audio_duration_seconds(str(path)) for path in audio_files}


def print_benchmark_plan(args, audio_files: List[Path], durations: Dict[str, Optional[float]]):
    known_durations = [duration for duration in durations.values() if duration is not None]
    total_audio_seconds = sum(known_durations)
    print("Benchmark plan")
    print(f"  files: {len(audio_files)}")
    print(f"  known audio duration: {format_timestamp(total_audio_seconds)}")
    print(f"  model: {args.model}")
    print(f"  device: {args.device}")
    print(f"  compute_type: {args.compute_type}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  beam_size: {args.beam_size}")
    print(f"  isolate_files: {args.isolate_files}")
    print(f"  resume_intermediates: {args.resume_intermediates}")


def run_review_benchmark_mode(args, output_dir: Path):
    runtime_config = runtime_config_payload(args)
    print("Review benchmark mode")
    report = run_review_benchmark(runtime_config, output_dir)
    json_path, md_path = write_review_benchmark_reports(output_dir, report)
    print(f"  fixtures: {report['benchmark_metadata']['fixture_count']}")
    print(f"  backend: {report['backend_identity']['backend_name']}")
    print(f"  model: {report['backend_identity']['review_model_name']}")
    print(f"  average quality score: {report['quality']['average_fixture_quality_score']}")
    print(f"  average elapsed seconds: {report['speed']['average_elapsed_seconds']}")
    print(f"  quality per second: {report['derived_scores']['quality_per_second']}")
    print(f"  recommended for fast default: {report['production_recommendations']['recommended_for_fast_default']}")
    print(f"  recommended for long context qa: {report['production_recommendations']['recommended_for_long_context_qa']}")
    print(f"  report json: {json_path}")
    print(f"  report markdown: {md_path}")


def run_pipeline_benchmark_mode(args, output_dir: Path):
    configured_pack = str(getattr(args, "evaluation_pack_path", "") or "").strip()
    configured_gold = str(getattr(args, "gold_set_dir", "") or "").strip()
    gold_set_dir = Path(
        configured_pack
        or configured_gold
        or (Path(__file__).resolve().parents[2] / "benchmarks" / "pipeline_gold_set")
    ).resolve()
    candidate_dir = Path(args.benchmark_candidate_dir).resolve() if args.benchmark_candidate_dir else output_dir.resolve()
    print("Pipeline quality benchmark mode")
    print(f"  gold set: {gold_set_dir}")
    print(f"  candidate outputs: {candidate_dir}")
    baseline_dir = Path(args.benchmark_baseline_dir).resolve() if args.benchmark_baseline_dir else None
    if baseline_dir:
        print(f"  baseline outputs: {baseline_dir}")
    report = run_pipeline_benchmark(gold_set_dir, candidate_dir, baseline_dir)
    if args.speech_run_id:
        identities = selected_provider_identities(args)
        report["providers"] = [identity.to_payload() for identity in identities]
        report["provider_policies"] = {
            "license": sorted({identity.license for identity in identities}),
            "acquisition": sorted({identity.acquisition for identity in identities}),
            "privacy_boundary": sorted({identity.privacy_boundary for identity in identities}),
            "offline_behavior": "local after explicit acquisition",
        }
        execution = resolve_execution_profile(args.device, args.batch_size)
        speech_run = build_speech_provider_run(
            run_id=args.speech_run_id,
            evaluation_pack=report.get("evaluation_identity") or {},
            audio_identity=(report.get("evaluation_identity") or {}).get("source_identity") or {},
            preprocessing={"profile": "gold-set-manifest", "normalization": "identical candidate/baseline inputs"},
            providers=identities,
            execution=asdict(execution),
            outputs={"candidate_dir": str(candidate_dir), "baseline_dir": str(baseline_dir or ""), "raw_outputs_preserved": True},
            metrics={"aggregate": report.get("aggregate") or {}, "results": report.get("results") or []},
        )
        shadow_root = Path(args.speech_shadow_root).resolve() if args.speech_shadow_root else output_dir.resolve() / "speech-shadow-runs"
        report["speech_run_manifest"] = str(write_immutable_speech_run(shadow_root, speech_run))
    json_path, md_path = write_pipeline_benchmark_reports(output_dir, report)
    aggregate = report.get("aggregate") or {}
    print(f"  completed entries: {report.get('gold_set', {}).get('entry_count', 0)}")
    print(f"  WER: {(aggregate.get('wer') or {}).get('wer', 0.0):.4f}")
    print(
        "  speaker-attributed WER: "
        f"{(aggregate.get('speaker_attributed_wer') or {}).get('speaker_attributed_wer', 0.0):.4f}"
    )
    print(f"  report json: {json_path}")
    print(f"  report markdown: {md_path}")


def selected_provider_identities(args) -> List[ProviderIdentity]:
    asr_model_id = str(args.model_id or "").strip()
    if not asr_model_id:
        if args.asr_provider == "parakeet":
            asr_model_id = str(args.model)
        else:
            whisper_model_ids = {
                "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
            }
            asr_model_id = whisper_model_ids.get(str(args.model), f"Systran/faster-whisper-{args.model}")
    speaker_model_id = str(args.speaker_model)
    if args.speaker_embedding_provider == "speechbrain_xvector" and "xvect" not in speaker_model_id.lower():
        speaker_model_id = "speechbrain/spkrec-xvect-voxceleb"
    identities = [
        ProviderIdentity(
            stage="transcription",
            provider=args.asr_provider,
            model=asr_model_id,
            model_revision=str(args.model_revision or ""),
            capabilities={"timestamps": True, "word_alignment": True, "device_support": ["cpu", "cuda"]},
            confidence_semantics="provider-specific; cross-provider confidence comparison is prohibited",
            license="model-card terms apply",
        ),
        pyannote_provider_identity(args.diarization_model, str(args.diarization_model_revision or "")),
    ]
    if args.workflow_profile != "anonymous_meeting":
        identities.append(
            ProviderIdentity(
                stage="speaker_embedding",
                provider=args.speaker_embedding_provider,
                model=speaker_model_id,
                model_revision=str(args.speaker_model_revision or ""),
                capabilities={"sample_rate": 16000, "device_support": ["cpu", "cuda"]},
                confidence_semantics="cosine similarity within one embedding family only",
                license="model-card terms apply",
            )
        )
    if args.alignment_provider == "timestamp_passthrough":
        identities.append(
            ProviderIdentity(
                stage="alignment",
                provider="timestamp_passthrough",
                model="asr_native_word_timestamps",
                model_revision="builtin-v1",
                acquisition="bundled",
                license="project license",
                capabilities={"timestamps": True, "word_alignment": True, "device_support": ["cpu", "cuda"]},
            )
        )
    else:
        identities.append(
            ProviderIdentity(
                stage="alignment",
                provider="whisperx",
                model=str(args.alignment_model or ""),
                model_revision=str(args.alignment_model_revision or ""),
                capabilities={"timestamps": True, "word_alignment": True, "device_support": ["cpu", "cuda"]},
                confidence_semantics="provider-specific alignment score",
                license="model-card terms apply",
            )
        )
    return identities


def run_provider_artifact_mode(args) -> None:
    cache_root = Path(args.provider_cache_dir).resolve()
    profile = resolve_execution_profile(args.device, args.batch_size)
    reports = []
    for identity in selected_provider_identities(args):
        if args.download_provider_models and identity.acquisition != "bundled":
            reports.append(acquire_provider_artifact(cache_root, identity, token=args.hf_token))
        else:
            reports.append(provider_preflight(cache_root, identity))
    print(json.dumps({"execution": asdict(profile), "providers": reports}, indent=2))


def estimate_audio_eta(processed_audio_seconds: float, elapsed_seconds: float, remaining_audio_seconds: float) -> Optional[float]:
    if processed_audio_seconds <= 0 or elapsed_seconds <= 0:
        return None
    seconds_per_audio_second = elapsed_seconds / processed_audio_seconds
    return seconds_per_audio_second * remaining_audio_seconds


def is_current_review_bundle_skip(state_info: Mapping[str, object]) -> bool:
    """Return whether an episode is already complete with the current review bundle."""

    return (
        str(state_info.get("state") or "") == "complete"
        and str(state_info.get("review_bundle_status") or "") == "current_review_complete"
    )


def eta_excluded_file_names(
    audio_files: List[Path],
    episode_states: Mapping[str, Mapping[str, object]],
) -> set[str]:
    """Identify already-reviewed episodes that should not affect batch ETA statistics."""

    return {
        audio_path.name
        for audio_path in audio_files
        if is_current_review_bundle_skip(episode_states.get(audio_path.name, {}))
    }


def build_child_process_command(args, audio_path: Path, output_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "podcast_transcribe_host.py"),
        "--input-dir",
        str(Path(args.input_dir).resolve()),
        "--input-file",
        str(audio_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--workflow-profile",
        args.workflow_profile,
        "--model",
        args.model,
        "--model-id",
        args.model_id,
        "--model-revision",
        args.model_revision,
        "--asr-provider",
        args.asr_provider,
        "--alignment-provider",
        args.alignment_provider,
        "--speaker-embedding-provider",
        args.speaker_embedding_provider,
        "--language",
        args.language,
        "--device",
        args.device,
        "--compute-type",
        args.compute_type,
        "--beam-size",
        str(args.beam_size),
        "--batch-size",
        str(args.batch_size),
        "--cleanup-level",
        args.cleanup_level,
        "--runtime-profile",
        args.runtime_profile,
        "--backend",
        args.backend,
        "--review-base-url",
        args.review_base_url,
        "--review-model-name",
        args.review_model_name,
        "--review-reasoning-effort",
        args.review_reasoning_effort,
        "--review-batch-token-limit",
        str(args.review_batch_token_limit),
        "--diarization-model",
        args.diarization_model,
        "--diarization-model-revision",
        args.diarization_model_revision,
        "--speaker-model",
        args.speaker_model,
        "--speaker-model-revision",
        args.speaker_model_revision,
        "--provider-cache-dir",
        args.provider_cache_dir,
        "--host-profile-json",
        args.host_profile_json,
        "--host-threshold",
        str(args.host_threshold),
        "--min-host-seconds",
        str(args.min_host_seconds),
        "--max-embedding-seconds",
        str(args.max_embedding_seconds),
        "--no-isolate-files",
    ]

    if args.alignment_model:
        command.extend(["--alignment-model", args.alignment_model])

    if args.hf_token:
        command.extend(["--hf-token", args.hf_token])
    if args.host_reference:
        command.extend(["--host-reference", args.host_reference])
    if args.known_speakers_dir:
        command.extend(["--known-speakers-dir", args.known_speakers_dir])
    if args.preferred_terms_file:
        command.extend(["--preferred-terms-file", args.preferred_terms_file])
    for preferred_term in args.preferred_terms or []:
        command.extend(["--preferred-term", preferred_term])
    if args.replacement_map_json:
        command.extend(["--replacement-map-json", args.replacement_map_json])
    if args.filename_date_preset:
        command.extend(["--filename-date-preset", args.filename_date_preset])
    if args.filename_date_position:
        command.extend(["--filename-date-position", args.filename_date_position])
    if args.filename_date_formats:
        command.extend(["--filename-date-formats", *args.filename_date_formats])
    if args.corrections_dir:
        command.extend(["--corrections-dir", args.corrections_dir])
    if args.transcript_cleanup_review is True:
        command.append("--transcript-cleanup-review")
    elif args.transcript_cleanup_review is False:
        command.append("--no-transcript-cleanup-review")
    if args.glossary_correction_review is True:
        command.append("--glossary-correction-review")
    elif args.glossary_correction_review is False:
        command.append("--no-glossary-correction-review")
    if args.speaker_consistency_review is True:
        command.append("--speaker-consistency-review")
    elif args.speaker_consistency_review is False:
        command.append("--no-speaker-consistency-review")
    if args.episode_qa_review is True:
        command.append("--episode-qa-review")
    elif args.episode_qa_review is False:
        command.append("--no-episode-qa-review")
    if args.review_debug:
        command.append("--review-debug")
    if args.review_debug_dir:
        command.extend(["--review-debug-dir", args.review_debug_dir])
    if args.review_auto_calibrate is True:
        command.append("--review-auto-calibrate")
    elif args.review_auto_calibrate is False:
        command.append("--no-review-auto-calibrate")
    if args.review_auto_adapt_upward is True:
        command.append("--review-auto-adapt-upward")
    elif args.review_auto_adapt_upward is False:
        command.append("--no-review-auto-adapt-upward")
    if args.review_context_budget:
        command.extend(["--review-context-budget", str(args.review_context_budget)])
    if not args.review_candidate_filter:
        command.append("--review-all-segments")
    if args.review_structured_output_support:
        command.append("--review-structured-output-support")
    if args.review_transcript_qa_available:
        command.append("--review-transcript-qa-available")
    if args.review_episode_wide_correction_available:
        command.append("--review-episode-wide-correction-available")
    if not args.resume_intermediates:
        command.append("--no-resume-intermediates")
    if args.archive_debug_artifacts:
        command.append("--archive-debug-artifacts")
    if args.assume_dominant_speaker_is_host:
        command.append("--assume-dominant-speaker-is-host")
    if args.num_speakers:
        command.extend(["--num-speakers", str(args.num_speakers)])

    return command


def run_isolated_batch(args, input_dir: Path, output_dir: Path, audio_files: List[Path]):
    """Process each episode in a child Python process to reclaim native memory between files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    resume_state_path = output_dir / RESUME_STATE_FILENAME
    existing_summary_rows = state_load_episode_summary_rows(summary_path, normalize_episode_summary_row)
    processed_files = state_load_processed_files(resume_state_path)
    effective_runtime_config = runtime_config_payload(args)
    episode_states = {
        audio_path.name: classify_episode_processing_state(
            audio_path,
            output_dir,
            processed_files,
            existing_summary_rows,
            effective_runtime_config,
        )
        for audio_path in audio_files
    }
    pending_audio_files = [
        audio_path
        for audio_path in audio_files
        if episode_states[audio_path.name]["state"] != "complete"
    ]
    if any(state_requires_tier1(str(episode_states[audio_path.name]["state"])) for audio_path in pending_audio_files):
        load_replacement_map(args.replacement_map_json)
    total_files = len(audio_files)
    batch_started = time.perf_counter()
    durations = audio_duration_map(audio_files)
    eta_excluded_names = eta_excluded_file_names(audio_files, episode_states)
    processed_audio_seconds = 0.0

    print("Using isolated per-file processing to release native memory between episodes.")
    for index, audio_path in enumerate(audio_files, start=1):
        duration_seconds = durations.get(str(audio_path))
        if duration_seconds is not None and duration_seconds >= LONG_FILE_WARNING_HOURS * 3600:
            print(
                f"Long file notice: {audio_path.name} is {format_timestamp(duration_seconds)} long. "
                "This file will run in its own Python process so memory is reclaimed before the next episode."
            )

        elapsed = time.perf_counter() - batch_started
        excluded_before = sum(1 for path in audio_files[: index - 1] if path.name in eta_excluded_names)
        completed_file_count = index - 1 - excluded_before
        average_seconds = elapsed / completed_file_count if completed_file_count > 0 else None
        remaining_files = sum(1 for path in audio_files[index - 1 :] if path.name not in eta_excluded_names)
        remaining_audio_seconds = sum(
            duration or 0.0
            for path_text, duration in durations.items()
            if Path(path_text) in audio_files[index - 1 :]
            and Path(path_text).name not in eta_excluded_names
        )
        eta_seconds = estimate_audio_eta(processed_audio_seconds, elapsed, remaining_audio_seconds)
        if eta_seconds is None and average_seconds is not None:
            eta_seconds = average_seconds * remaining_files
        if eta_seconds is not None:
            print(
                f"Batch progress: file {index} of {total_files} "
                f"(estimated remaining {format_timestamp(eta_seconds)}, "
                f"processed_audio={format_timestamp(processed_audio_seconds)})"
            )
        else:
            print(f"Batch progress: file {index} of {total_files}")

        state_info = classify_episode_processing_state(
            audio_path,
            output_dir,
            processed_files,
            existing_summary_rows,
            effective_runtime_config,
        )
        if state_info["state"] == "complete":
            bundle_status = state_info.get("review_bundle_status")
            if bundle_status == "current_review_complete":
                print(
                    f"Skipping completed file: {audio_path.name} "
                    "(current reviewed bundle already present)"
                )
            else:
                print(f"Skipping completed file: {audio_path.name}")
            if not is_current_review_bundle_skip(state_info):
                processed_audio_seconds += duration_seconds or 0.0
            continue
        if state_info["state"] == "needs_tier2_only" and state_info.get("missing_review_stages"):
            print(
                f"Review shortfall for {audio_path.name}: "
                f"{', '.join(state_info['missing_review_stages'])}"
            )
        if state_info.get("provider_shortfalls"):
            print(
                f"Provider delta for {audio_path.name}: "
                f"{', '.join(state_info['provider_shortfalls'])}; compatible stage artifacts will be reused."
            )
        mode_labels = {
            "needs_tier2_only": "tier2-only backfill",
            "needs_v2_delta_upgrade": "v2 contract delta upgrade",
            "needs_v2_cached_rebuild": "v2 cached rebuild",
            "needs_v2_full_reprocess": "v2 full reprocess",
            "v2_upgrade_blocked": "v2 upgrade blocked",
        }
        print(f"Processing mode for {audio_path.name}: {mode_labels.get(str(state_info['state']), 'tier1+tier2')}")
        if state_info["state"] == "v2_upgrade_blocked":
            print(f"V2 upgrade blocked for {audio_path.name}: source audio or required artifacts are unavailable.")
            continue

        command = build_child_process_command(args, audio_path, output_dir)
        try:
            result = subprocess.run(
                command,
                timeout=args.child_timeout_seconds if args.child_timeout_seconds and args.child_timeout_seconds > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Child process timed out for {audio_path.name} after {args.child_timeout_seconds} seconds. "
                "Intermediate artifacts may allow the next run to resume inside the episode."
            ) from exc
        existing_summary_rows = state_load_episode_summary_rows(summary_path, normalize_episode_summary_row)
        processed_files = state_load_processed_files(resume_state_path)
        if result.returncode != 0:
            refreshed_state = classify_episode_processing_state(
                audio_path,
                output_dir,
                processed_files,
                existing_summary_rows,
                effective_runtime_config,
            )
            if refreshed_state["state"] in {"complete", "v2_complete"}:
                print(
                    f"Child process for {audio_path.name} exited with code {result.returncode} "
                    "after writing all expected outputs; continuing batch."
                )
                continue
            raise RuntimeError(f"Child process failed for {audio_path.name} with exit code {result.returncode}.")
        processed_audio_seconds += duration_seconds or 0.0

    existing_summary_rows = state_load_episode_summary_rows(summary_path, normalize_episode_summary_row)
    operation_started = start_console_operation("batch finalization", "writing run-level reports")
    write_run_reports(
        output_dir,
        list(existing_summary_rows.values()),
        elapsed_seconds=time.perf_counter() - batch_started,
        workflow_profile=str(effective_runtime_config.get("workflow_profile") or "podcast"),
    )
    finish_console_operation("batch finalization", "writing run-level reports", operation_started)
    print_final_review_summary(list(existing_summary_rows.values()))
    print(f"Wrote folder summary: {summary_path}")


def exit_isolated_worker_after_success(input_file: Optional[str], exit_fn=None) -> bool:
    """Exit a successful isolated worker before native model teardown can fault."""

    if not input_file:
        return False

    sys.stdout.flush()
    sys.stderr.flush()
    (exit_fn or os._exit)(0)
    return True


def load_models(args, device: str):
    cache_root = Path(args.provider_cache_dir).resolve()
    identities = {identity.stage: identity for identity in selected_provider_identities(args)}
    preflights = {stage: provider_preflight(cache_root, identity) for stage, identity in identities.items()}
    missing = [f"{stage}: {report['diagnostic']} ({report['artifact_path']})" for stage, report in preflights.items() if not report["available"]]
    if missing:
        raise RuntimeError(
            "Provider artifacts are not ready. No model downloads occur implicitly. Run --provider-preflight, then "
            "--download-provider-models with pinned revisions. Missing: " + "; ".join(missing)
        )
    asr_identity = identities["transcription"]
    asr_local_path = str(artifact_directory(cache_root, asr_identity) / "model")
    if args.asr_provider == "parakeet":
        asr_provider = ParakeetASRProvider(
            asr_identity.model,
            device=device,
            model_revision=asr_identity.model_revision,
            local_model_path=asr_local_path,
        )
    else:
        whisper_model = WhisperModel(asr_local_path, device=device, compute_type=args.compute_type, local_files_only=True)
        asr_provider = FasterWhisperASRProvider(
            whisper_model,
            asr_identity.model,
            transcribe_audio,
            model_revision=asr_identity.model_revision,
        )
    alignment_identity = identities["alignment"]
    alignment_local_path = "" if alignment_identity.acquisition == "bundled" else str(artifact_directory(cache_root, alignment_identity) / "model")
    alignment_provider = create_alignment_provider(
        args.alignment_provider,
        device,
        args.alignment_model,
        args.alignment_model_revision,
        alignment_local_path,
    )

    try:
        diarization_pipeline, resolved_diarization_model = load_diarization_pipeline(
            str(artifact_directory(cache_root, identities["diarization"]) / "model"), args.hf_token
        )
    except TypeError as exc:
        raise RuntimeError(
            "Failed to load the pyannote diarization model because this environment's pyannote.audio API "
            "does not match the loader call. The code now supports both 'token' and 'use_auth_token', so "
            "this likely indicates an unexpected pyannote.audio version or conflicting installation. "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        message = str(exc).lower()
        if any(token_hint in message for token_hint in ["401", "403", "unauthorized", "forbidden", "access denied"]):
            raise RuntimeError(
                "Failed to load the pyannote diarization model because Hugging Face rejected the token or model access. "
                "Confirm the token value and make sure you have accepted access terms for "
                "pyannote/speaker-diarization-community-1."
            ) from exc
        if "plda" in message and "unexpected keyword argument" in message:
            raise RuntimeError(
                "Failed to load the diarization pipeline because the installed pyannote.audio version is not "
                "compatible with 'pyannote/speaker-diarization-community-1'. Upgrade to pyannote.audio 4.x for "
                "community-1, or use the legacy 'pyannote/speaker-diarization-3.1' pipeline with pyannote.audio 3.x."
            ) from exc
        raise RuntimeError(
            f"Failed to load diarization model '{args.diarization_model}'. Original error: {exc}"
        ) from exc

    print(f"Using diarization model: {resolved_diarization_model}")
    if device == "cuda":
        diarization_pipeline.to(torch.device(normalize_runtime_device(device)))

    if args.workflow_profile == "anonymous_meeting":
        return (
            asr_provider,
            alignment_provider,
            diarization_pipeline,
            None,
            AnonymousSpeakerEmbeddingProvider(),
            {},
        )

    speaker_model = str(artifact_directory(cache_root, identities["speaker_embedding"]) / "model")
    if args.speaker_embedding_provider == "speechbrain_xvector" and "xvect" not in speaker_model.lower():
        speaker_model = str(artifact_directory(cache_root, identities["speaker_embedding"]) / "model")
    verifier = load_speaker_verifier(speaker_model, device, args.speaker_embedding_provider)
    speaker_embedding_provider = (
        SpeechBrainXVectorProvider(verifier, identities["speaker_embedding"].model, identities["speaker_embedding"].model_revision)
        if args.speaker_embedding_provider == "speechbrain_xvector"
        else SpeechBrainECAPAProvider(verifier, identities["speaker_embedding"].model, identities["speaker_embedding"].model_revision)
    )
    known_speaker_profiles = load_known_speaker_profiles(
        verifier=verifier,
        known_speakers_dir=args.known_speakers_dir,
    )
    return (
        asr_provider,
        alignment_provider,
        diarization_pipeline,
        verifier,
        speaker_embedding_provider,
        known_speaker_profiles,
    )


def process_audio_batch(args, input_dir: Path, output_dir: Path, audio_files: List[Path]):
    """Process a batch in the current interpreter while reusing loaded models."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    resume_state_path = output_dir / RESUME_STATE_FILENAME
    existing_summary_rows = state_load_episode_summary_rows(summary_path, normalize_episode_summary_row)
    processed_files = state_load_processed_files(resume_state_path)
    episode_summary_rows_by_name = dict(existing_summary_rows)
    historical_similarity_scores = build_historical_similarity_scores(list(existing_summary_rows.values()))
    effective_runtime_config = runtime_config_payload(args)
    episode_states = {
        audio_path.name: classify_episode_processing_state(
            audio_path,
            output_dir,
            processed_files,
            episode_summary_rows_by_name,
            effective_runtime_config,
        )
        for audio_path in audio_files
    }
    resolved_review_runtime = resolve_review_runtime_config(effective_runtime_config)
    resolved_review_backend_capabilities = enrich_backend_capabilities_with_identity(
        resolve_backend_capabilities(effective_runtime_config)
    )
    review_calibration_session = (
        load_review_calibration_session(output_dir, resolved_review_backend_capabilities)
        if resolved_review_runtime.get("any_review_enabled") and resolved_review_runtime.get("backend_ready")
        else None
    )
    needs_tier1 = any(state_requires_tier1(str(state["state"])) for state in episode_states.values())
    preferred_terms: List[str] = []
    initial_prompt = None
    hotwords = None
    replacement_map: Dict[str, List[str]] = {}
    device = None
    asr_provider = alignment_provider = diarization_pipeline = verifier = speaker_embedding_provider = known_speaker_profiles = None
    if needs_tier1:
        preferred_terms = list(dict.fromkeys(load_preferred_terms(args.preferred_terms_file) + list(args.preferred_terms or [])))
        initial_prompt, hotwords = build_prompt_bias(preferred_terms)
        replacement_map = load_replacement_map(args.replacement_map_json)
        device = get_device(args.device)
        print(f"Using device: {device}")
        (
            asr_provider,
            alignment_provider,
            diarization_pipeline,
            verifier,
            speaker_embedding_provider,
            known_speaker_profiles,
        ) = load_models(args, device)
    else:
        if resolved_review_runtime.get("any_review_enabled"):
            print("No tier-1 work required; running review backfill from existing cleaned JSON outputs only.")
        else:
            print("No processing work required; all selected episodes already satisfy the active contract.")

    total_files = len(audio_files)
    is_single_episode_worker = bool(args.input_file) or total_files == 1
    batch_started = time.perf_counter()
    durations = audio_duration_map(audio_files)
    eta_excluded_names = eta_excluded_file_names(audio_files, episode_states)
    processed_audio_seconds = 0.0
    for index, audio_path in enumerate(audio_files, start=1):
        duration_seconds = durations.get(str(audio_path))
        if duration_seconds is not None and duration_seconds >= LONG_FILE_WARNING_HOURS * 3600:
            print(
                f"Long file notice: {audio_path.name} is {format_timestamp(duration_seconds)} long. "
                "Speaker matching streams diarized spans, but diarization may still preload the full file (requiring significant system RAM) "
                "when pyannote's path decoder is unavailable in the local environment."
            )
        elapsed = time.perf_counter() - batch_started
        excluded_before = sum(1 for path in audio_files[: index - 1] if path.name in eta_excluded_names)
        completed_file_count = index - 1 - excluded_before
        average_seconds = elapsed / completed_file_count if completed_file_count > 0 else None
        remaining_files = sum(1 for path in audio_files[index - 1 :] if path.name not in eta_excluded_names)
        remaining_audio_seconds = sum(
            duration or 0.0
            for path_text, duration in durations.items()
            if Path(path_text) in audio_files[index - 1 :]
            and Path(path_text).name not in eta_excluded_names
        )
        eta_seconds = estimate_audio_eta(processed_audio_seconds, elapsed, remaining_audio_seconds)
        if eta_seconds is None and average_seconds is not None:
            eta_seconds = average_seconds * remaining_files
        if not is_single_episode_worker:
            if eta_seconds is not None:
                print(
                    f"Batch progress: file {index} of {total_files} "
                    f"(estimated remaining {format_timestamp(eta_seconds)}, "
                    f"processed_audio={format_timestamp(processed_audio_seconds)})"
                )
            else:
                print(f"Batch progress: file {index} of {total_files}")
        state_info = classify_episode_processing_state(
            audio_path,
            output_dir,
            processed_files,
            episode_summary_rows_by_name,
            effective_runtime_config,
        )
        if state_info["state"] == "complete":
            bundle_status = state_info.get("review_bundle_status")
            if bundle_status == "current_review_complete":
                print(
                    f"Skipping completed file: {audio_path.name} "
                    "(current reviewed bundle already present)"
                )
            else:
                print(f"Skipping completed file: {audio_path.name}")
            if not is_current_review_bundle_skip(state_info):
                processed_audio_seconds += duration_seconds or 0.0
            continue
        if not is_single_episode_worker:
            if state_info["state"] == "needs_tier2_only" and state_info.get("missing_review_stages"):
                print(
                    f"Review shortfall for {audio_path.name}: "
                    f"{', '.join(state_info['missing_review_stages'])}"
                )
            if state_info.get("provider_shortfalls"):
                print(
                    f"Provider delta for {audio_path.name}: "
                    f"{', '.join(state_info['provider_shortfalls'])}; compatible stage artifacts will be reused."
                )
            mode_labels = {
                "needs_tier2_only": "tier2-only backfill",
                "needs_v2_delta_upgrade": "v2 contract delta upgrade",
                "needs_v2_cached_rebuild": "v2 cached rebuild",
                "needs_v2_full_reprocess": "v2 full reprocess",
                "v2_upgrade_blocked": "v2 upgrade blocked",
            }
            print(f"Processing mode for {audio_path.name}: {mode_labels.get(str(state_info['state']), 'tier1+tier2')}")

        if state_info["state"] == "needs_v2_delta_upgrade":
            episode_summary = process_v2_delta_upgrade(
                audio_path,
                output_dir,
                episode_summary_rows_by_name.get(audio_path.name),
            )
        elif state_info["state"] == "v2_upgrade_blocked":
            print(f"V2 upgrade blocked for {audio_path.name}: source audio or required artifacts are unavailable.")
            continue
        elif state_info["state"] == "needs_tier2_only":
            episode_summary = process_review_backfill_from_cleaned_json(
                audio_path=audio_path,
                output_dir=output_dir,
                runtime_config=effective_runtime_config,
                review_calibration_session=review_calibration_session,
                existing_summary_row=episode_summary_rows_by_name.get(audio_path.name),
            )
        else:
            if state_info["state"] in {"needs_v2_cached_rebuild", "needs_v2_full_reprocess"}:
                archive_path = archive_legacy_episode_bundle(audio_path, output_dir)
                if archive_path is not None:
                    print(f"  archived legacy v1 contract artifacts: {archive_path}")
            episode_summary = process_file(
                audio_path=audio_path,
                output_dir=output_dir,
                asr_provider=asr_provider,
                alignment_provider=alignment_provider,
                diarization_pipeline=diarization_pipeline,
                verifier=verifier,
                speaker_embedding_identity=speaker_embedding_provider.identity,
                language=args.language,
                beam_size=args.beam_size,
                batch_size=args.batch_size,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                replacement_map=replacement_map,
                host_reference=args.host_reference,
                host_profile_path=args.host_profile_json,
                known_speaker_profiles=known_speaker_profiles,
                host_threshold=args.host_threshold,
                assume_dominant=args.assume_dominant_speaker_is_host,
                max_embedding_seconds=args.max_embedding_seconds,
                min_host_seconds=args.min_host_seconds,
                num_speakers=args.num_speakers,
                cleanup_level=args.cleanup_level,
                corrections_dir=args.corrections_dir,
                runtime_config=effective_runtime_config,
                review_calibration_session=review_calibration_session,
                historical_similarity_scores=historical_similarity_scores,
                resume_intermediates=args.resume_intermediates,
                archive_debug_artifacts=args.archive_debug_artifacts,
            )
        post_episode_started = time.perf_counter()
        operation_started = start_console_operation("post-episode", "updating in-memory batch state")
        episode_summary_rows_by_name[audio_path.name] = episode_summary
        historical_similarity_scores = build_historical_similarity_scores(list(episode_summary_rows_by_name.values()))
        finish_console_operation("post-episode", "updating in-memory batch state", operation_started)

        operation_started = start_console_operation("post-episode", "recording source fingerprint")
        processed_files[audio_path.name] = audio_file_fingerprint(audio_path)
        finish_console_operation("post-episode", "recording source fingerprint", operation_started)

        operation_started = start_console_operation("post-episode", "writing episode summary CSV")
        write_episode_summary_csv(summary_path, list(episode_summary_rows_by_name.values()))
        finish_console_operation("post-episode", "writing episode summary CSV", operation_started)

        operation_started = start_console_operation("post-episode", "saving processed-file state")
        state_save_processed_files(resume_state_path, processed_files)
        finish_console_operation("post-episode", "saving processed-file state", operation_started)

        operation_started = start_console_operation("post-episode", "saving review calibration state")
        save_review_calibration_session(output_dir, review_calibration_session)
        finish_console_operation("post-episode", "saving review calibration state", operation_started)
        processed_audio_seconds += duration_seconds or 0.0
        operation_started = start_console_operation("post-episode", "releasing Python objects")
        gc.collect()
        finish_console_operation("post-episode", "releasing Python objects", operation_started)
        if torch.cuda.is_available():
            operation_started = start_console_operation("post-episode", "releasing CUDA cache")
            torch.cuda.empty_cache()
            finish_console_operation("post-episode", "releasing CUDA cache", operation_started)
        print(f"  post-episode complete in {time.perf_counter() - post_episode_started:.1f}s")

    if args.input_file:
        print("Isolated worker: episode state saved; deferring run-level reports to the parent.")
        exit_isolated_worker_after_success(args.input_file)
        return

    operation_started = start_console_operation("batch finalization", "writing episode summary CSV")
    write_episode_summary_csv(summary_path, list(episode_summary_rows_by_name.values()))
    finish_console_operation("batch finalization", "writing episode summary CSV", operation_started)

    operation_started = start_console_operation("batch finalization", "saving processed-file state")
    state_save_processed_files(resume_state_path, processed_files)
    finish_console_operation("batch finalization", "saving processed-file state", operation_started)

    operation_started = start_console_operation("batch finalization", "saving review calibration state")
    save_review_calibration_session(output_dir, review_calibration_session)
    finish_console_operation("batch finalization", "saving review calibration state", operation_started)

    operation_started = start_console_operation("batch finalization", "writing run-level reports")
    write_run_reports(
        output_dir,
        list(episode_summary_rows_by_name.values()),
        elapsed_seconds=time.perf_counter() - batch_started,
        workflow_profile=str(effective_runtime_config.get("workflow_profile") or "podcast"),
    )
    finish_console_operation("batch finalization", "writing run-level reports", operation_started)
    print_final_review_summary(list(episode_summary_rows_by_name.values()))
    print(f"Wrote folder summary: {summary_path}")


def main():
    """CLI entry point used by the compatibility wrapper and package console script."""

    args = parse_args()

    if args.provider_preflight or args.download_provider_models:
        run_provider_artifact_mode(args)
        return

    if args.review_benchmark:
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        run_review_benchmark_mode(args, output_dir)
        return

    if args.pipeline_benchmark:
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "output"
        run_pipeline_benchmark_mode(args, output_dir)
        return

    if not args.input_dir:
        raise RuntimeError("--input-dir is required unless a benchmark mode is used.")

    if args.batch_size <= 0:
        execution = resolve_execution_profile(args.device, 0)
        args.batch_size = execution.batch_size
        print(f"Adaptive batch size: {args.batch_size} ({execution.resolved_device}/{execution.precision})")
        if execution.fallback_reason:
            print(f"Device diagnostic: {execution.fallback_reason}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    audio_files = discover_audio_files(input_dir, args.input_file)
    if not audio_files:
        raise RuntimeError(f"No supported audio files found in {input_dir}")
    disk_space_preflight(output_dir, audio_files)
    durations = audio_duration_map(audio_files)
    if args.benchmark_only:
        print_benchmark_plan(args, audio_files, durations)
        return

    effective_runtime_config = runtime_config_payload(args)
    existing_summary_rows = state_load_episode_summary_rows(output_dir / SUMMARY_FILENAME, normalize_episode_summary_row)
    processed_files = state_load_processed_files(output_dir / RESUME_STATE_FILENAME)
    requires_tier1 = any(
        classify_episode_processing_state(
            audio_path,
            output_dir,
            processed_files,
            existing_summary_rows,
            effective_runtime_config,
        )["state"]
        in {"needs_tier1", "needs_v2_cached_rebuild", "needs_v2_full_reprocess"}
        for audio_path in audio_files
    )

    if requires_tier1 and not args.hf_token:
        raise RuntimeError(
            "A Hugging Face token is required for pyannote diarization. "
            "Set HF_TOKEN or pass --hf-token."
        )

    if args.isolate_files and args.input_file is None:
        run_isolated_batch(args, input_dir, output_dir, audio_files)
    else:
        process_audio_batch(args, input_dir, output_dir, audio_files)


if __name__ == "__main__":
    main()
