import argparse
import csv
import gc
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
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def configure_ffmpeg_dll_directory():
    ffmpeg_bin_dir = os.getenv("PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR") or os.getenv("FFMPEG_BIN_DIR")
    if os.name != "nt" or not ffmpeg_bin_dir or not hasattr(os, "add_dll_directory"):
        return

    if os.path.isdir(ffmpeg_bin_dir):
        os.add_dll_directory(ffmpeg_bin_dir)


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

import numpy as np
import torch
import torchaudio
import huggingface_hub
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
    DEFAULT_REVIEW_BACKEND,
    DEFAULT_RUNTIME_PROFILE,
    REVIEW_BACKENDS,
    RUNTIME_PROFILES,
    load_replacement_map as config_load_replacement_map,
    resolve_review_runtime_config,
)
from podcast_transcribe.contract import (
    validate_reviewed_transcript_payload,
    validate_transcript_payload,
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
    RESUME_STATE_FILENAME,
    REVIEW_CALIBRATION_FILENAME,
    SUMMARY_FILENAME,
    audio_file_fingerprint,
    clear_stage_artifacts as state_clear_stage_artifacts,
    expected_output_paths as state_expected_output_paths,
    is_file_already_processed as state_is_file_already_processed,
    load_review_calibration_state as state_load_review_calibration_state,
    load_stage_artifact as state_load_stage_artifact,
    load_episode_summary_rows as state_load_episode_summary_rows,
    load_processed_files as state_load_processed_files,
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


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
LONG_FILE_WARNING_HOURS = 4.0


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


def create_stage_progress(transient: bool = False) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
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


def make_review_progress_callback():
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
            print(f"  {str(event.get('summary') or 'Review calibration complete.')}")
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


def review_calibration_state_path(output_dir: Path) -> Path:
    return output_dir / REVIEW_CALIBRATION_FILENAME


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
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name.")
    parser.add_argument("--language", default="en", help="Language code.")
    parser.add_argument("--device", default="auto", help="Whisper device: auto, cpu, or cuda.")
    # "auto" can pick CPU paths or unsupported configs. 5070 Ti → float16 is correct and fastest
    # parser.add_argument("--compute-type", default="auto", help="faster-whisper compute type.")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute type.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for faster-whisper.")
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
        help="Disable reuse of per-episode transcription and diarization artifacts.",
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
        help="Keep intermediate stage artifacts after successful output writing for debugging.",
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

    return shutil.which("ffprobe")


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


def log_memory_usage(stage_label: str):
    process_memory = get_process_memory_mb()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(
            f"  memory [{stage_label}]: cpu_working_set={format_memory_mb(process_memory)}, "
            f"gpu_allocated={allocated:.0f} MiB, gpu_reserved={reserved:.0f} MiB"
        )
    else:
        print(f"  memory [{stage_label}]: cpu_working_set={format_memory_mb(process_memory)}")


def load_preferred_terms(path: Optional[str]) -> List[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_speaker_verifier(model_id: str, device: str):
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

    waveform, _ = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if resampler is not None:
        waveform = resampler(waveform)
    elif sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.squeeze(0).contiguous()


def load_host_profile(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
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


def save_host_profile(path: Optional[str], embedding: Optional[np.ndarray], source: str):
    if not path or embedding is None:
        return
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "updated_from": source,
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


def save_transcription_artifact(output_dir: Path, audio_path: Path, segments: List[SegmentItem], info_payload: Dict[str, object]):
    state_save_stage_artifact(
        output_dir,
        audio_path,
        "transcription",
        {
            "segments": [segment_to_payload(segment) for segment in segments],
            "info_payload": info_payload,
        },
    )


def load_transcription_artifact(output_dir: Path, audio_path: Path) -> Optional[Tuple[List[SegmentItem], Dict[str, object]]]:
    payload = state_load_stage_artifact(output_dir, audio_path, "transcription")
    if not payload:
        return None
    raw_segments = payload.get("segments")
    info_payload = payload.get("info_payload")
    if not isinstance(raw_segments, list) or not isinstance(info_payload, dict):
        return None
    return [segment_from_payload(segment) for segment in raw_segments if isinstance(segment, dict)], info_payload


def save_diarization_artifact(output_dir: Path, audio_path: Path, diarized_turns: List[Dict[str, object]]):
    state_save_stage_artifact(output_dir, audio_path, "diarization", {"diarized_turns": diarized_turns})


def load_diarization_artifact(output_dir: Path, audio_path: Path) -> Optional[List[Dict[str, object]]]:
    payload = state_load_stage_artifact(output_dir, audio_path, "diarization")
    if not payload or not isinstance(payload.get("diarized_turns"), list):
        return None
    return [turn for turn in payload["diarized_turns"] if isinstance(turn, dict)]


def run_transcription_stage(
    output_dir: Path,
    audio_path: Path,
    whisper_model: WhisperModel,
    language: str,
    beam_size: int,
    batch_size: int,
    initial_prompt: Optional[str],
    hotwords: Optional[str],
    resume_intermediates: bool,
) -> Tuple[List[SegmentItem], Dict[str, object], bool]:
    if resume_intermediates:
        cached = load_transcription_artifact(output_dir, audio_path)
        if cached:
            print("  stage: transcription (reused cached artifact)")
            return cached[0], cached[1], True

    print("  stage: transcription")
    segments, info_payload = transcribe_audio(
        model=whisper_model,
        audio_path=str(audio_path),
        language=language,
        beam_size=beam_size,
        batch_size=batch_size,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )
    save_transcription_artifact(output_dir, audio_path, segments, info_payload)
    return segments, info_payload, False


def run_diarization_stage(
    output_dir: Path,
    audio_path: Path,
    diarization_pipeline: Pipeline,
    num_speakers: Optional[int],
    resume_intermediates: bool,
) -> Tuple[List[Dict[str, object]], bool]:
    if resume_intermediates:
        cached = load_diarization_artifact(output_dir, audio_path)
        if cached is not None:
            print("  stage: diarization (reused cached artifact)")
            return cached, True

    print("  stage: diarization")
    diarized_turns = diarize_audio(diarization_pipeline, str(audio_path), num_speakers=num_speakers)
    save_diarization_artifact(output_dir, audio_path, diarized_turns)
    return diarized_turns, False


def pyannote_path_input_available() -> bool:
    try:
        import pyannote.audio.core.io as pyannote_io
    except Exception:
        return False

    return hasattr(pyannote_io, "AudioDecoder")


def diarize_audio(pipeline: Pipeline, audio_path: str, num_speakers: Optional[int]) -> List[Dict[str, object]]:
    """Run pyannote diarization and return plain speaker-turn dictionaries."""

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    if pyannote_path_input_available():
        try:
            with ProgressHook() as hook:
                diarization = pipeline(audio_path, hook=hook, **kwargs)
        except Exception as path_exc:
            print(
                "  diarization path input failed unexpectedly; falling back to preloaded audio. "
                f"Path-input error: {path_exc}"
            )
        else:
            return diarization_to_turns(diarization)

    waveform, sample_rate = torchaudio.load(audio_path)
    diarization_input = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }
    with ProgressHook() as hook:
        diarization = pipeline(diarization_input, hook=hook, **kwargs)

    return diarization_to_turns(diarization)


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

    for segment in segments:
        overlap_by_speaker = defaultdict(float)
        for turn in diarized_turns:
            overlap = overlap_seconds(segment.start, segment.end, turn["start"], turn["end"])
            if overlap > 0:
                overlap_by_speaker[turn["speaker"]] += overlap

        if overlap_by_speaker:
            segment.speaker = max(overlap_by_speaker.items(), key=lambda item: item[1])[0]
        elif not segment.speaker:
            segment.speaker = "UNKNOWN"

        for word in segment.words:
            if word.start is None or word.end is None:
                word.speaker = segment.speaker
                continue

            word_overlap = defaultdict(float)
            for turn in diarized_turns:
                overlap = overlap_seconds(word.start, word.end, turn["start"], turn["end"])
                if overlap > 0:
                    word_overlap[turn["speaker"]] += overlap

            if word_overlap:
                word.speaker = max(word_overlap.items(), key=lambda item: item[1])[0]
            else:
                word.speaker = segment.speaker
            if not word.speaker:
                word.speaker = "UNKNOWN"


def speaker_durations(diarized_turns: List[Dict[str, object]]) -> Dict[str, float]:
    totals = defaultdict(float)
    for turn in diarized_turns:
        totals[turn["speaker"]] += max(0.0, turn["end"] - turn["start"])
    return dict(totals)


def build_speaker_audio_samples(
    audio_path: str,
    diarized_turns: List[Dict[str, object]],
    max_seconds: float,
) -> Dict[str, torch.Tensor]:
    clips = defaultdict(list)
    durations = defaultdict(float)
    sample_rate, _, _ = get_audio_metadata(audio_path)
    if sample_rate is None or sample_rate <= 0:
        sample_rate = 16000
    resampler = (
        torchaudio.transforms.Resample(sample_rate, 16000)
        if sample_rate != 16000
        else None
    )
    for turn in diarized_turns:
        speaker = turn["speaker"]
        if durations[speaker] >= max_seconds:
            continue

        remaining = max_seconds - durations[speaker]
        clipped_end = min(float(turn["end"]), float(turn["start"]) + remaining)
        if clipped_end <= float(turn["start"]):
            continue

        segment = load_audio_span_mono_16k(
            audio_path,
            start_seconds=float(turn["start"]),
            end_seconds=clipped_end,
            sample_rate=sample_rate,
            resampler=resampler,
        )
        if segment.numel() == 0:
            continue

        clips[speaker].append(segment)
        durations[speaker] += segment.shape[0] / 16000.0

    merged = {}
    for speaker, chunks in clips.items():
        merged[speaker] = torch.cat(chunks)
    return merged


def compute_embedding(verifier: Any, waveform_16k: torch.Tensor) -> np.ndarray:
    signal = waveform_16k.unsqueeze(0)
    with torch.no_grad():
        embedding = verifier.encode_batch(signal)
    vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def load_known_speaker_profiles(
    verifier: Any,
    known_speakers_dir: Optional[str],
) -> Dict[str, Dict[str, object]]:
    config_entries = load_known_speakers_config(known_speakers_dir)
    if not config_entries:
        return {}

    base_dir = Path(known_speakers_dir)
    profiles = {}

    for entry in config_entries:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name", "")).strip()
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
            embeddings.append(compute_embedding(verifier, waveform))
            resolved_files.append(str(sample_path))

        averaged = average_embeddings(embeddings)
        if averaged is None:
            continue

        profiles[name] = {
            "name": name,
            "embedding": averaged,
            "is_host": bool(entry.get("is_host", False)) or name.upper() == "HOST",
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
) -> Tuple[Optional[str], Dict[str, np.ndarray], Optional[np.ndarray], Dict[str, float], Dict[str, float]]:
    durations = speaker_durations(diarized_turns)
    if not durations:
        return None, {}, existing_profile, {}, {}

    speaker_audio = build_speaker_audio_samples(audio_path, diarized_turns, max_embedding_seconds)
    speaker_embeddings = {}
    for speaker, clip in speaker_audio.items():
        if durations.get(speaker, 0.0) >= min_host_seconds:
            speaker_embeddings[speaker] = compute_embedding(verifier, clip)
        del clip
    speaker_audio.clear()
    gc.collect()

    reference_embedding = existing_profile
    if host_reference_path:
        ref_waveform = load_audio_mono_16k(host_reference_path)
        reference_embedding = compute_embedding(verifier, ref_waveform)
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
        "diarization_artifact_reused": "",
        "review_attempted": False,
        "review_status": "",
        "review_skip_reason": "",
        "review_runtime_profile": "",
        "review_backend": "",
        "review_model_name": "",
        "reviewed_segment_count": 0,
        "review_corrected_segment_count": 0,
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
        "tier1_reused_from_existing": False,
        "review_backfilled_from_cleaned_json": False,
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
        "diarization_artifact_reused",
        "review_attempted",
        "review_status",
        "review_skip_reason",
        "review_runtime_profile",
        "review_backend",
        "review_model_name",
        "reviewed_segment_count",
        "review_corrected_segment_count",
        "reviewed_output_written",
        "review_pipeline_version",
        "review_enabled_stages",
        "review_completed_stages",
        "review_skipped_stages",
        "review_input_source",
        "review_episode_qa_mode",
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
        "tier1_reused_from_existing",
        "review_backfilled_from_cleaned_json",
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
    }
    bool_fields = {
        "host_detected",
        "transcription_artifact_reused",
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
    summary_row["reviewed_output_written"] = bool(review_result["segments"])
    summary_row["review_pipeline_version"] = str(review_metadata.get("review_pipeline_version") or "")
    summary_row["review_enabled_stages"] = ";".join(str(item) for item in review_metadata.get("review_enabled_stages") or [])
    summary_row["review_completed_stages"] = ";".join(str(item) for item in review_metadata.get("review_completed_stages") or [])
    summary_row["review_skipped_stages"] = ";".join(str(item) for item in review_metadata.get("review_skipped_stages") or [])
    summary_row["review_input_source"] = str(review_metadata.get("review_input_source") or "")
    summary_row["review_episode_qa_mode"] = str(review_metadata.get("episode_qa_mode") or "")
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
        "audio_file": audio_path.name,
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if details:
        payload["details"] = details
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    return all(path.exists() for path in expected_paths)


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
    if cleaned_json_path.exists():
        try:
            load_cleaned_transcript_payload(cleaned_json_path)
            cleaned_json_usable = True
        except RuntimeError as exc:
            cleaned_json_error = str(exc)
    reviewed_bundle = reviewed_output_bundle_status(audio_path, output_dir, resolved_review)
    baseline_complete = baseline_resume_complete
    if resolved_review["any_review_enabled"]:
        baseline_complete = baseline_bundle_complete and cleaned_json_usable

    if not resolved_review["any_review_enabled"]:
        state = "complete" if baseline_resume_complete and baseline_bundle_complete else "needs_tier1"
    elif baseline_complete and cleaned_json_usable:
        if reviewed_bundle["status"] == "current_review_complete":
            state = "complete"
        else:
            state = "needs_tier2_only"
    else:
        state = "needs_tier1"

    return {
        "state": state,
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
    return state["state"] == "complete"


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
    print(f"Processing {audio_path.name}")
    print_episode_mode("tier2-only backfill")
    output_dir.mkdir(parents=True, exist_ok=True)
    print_episode_stage(1, 3, "load cleaned transcript")
    cleaned_path = cleaned_json_output_path(audio_path, output_dir)
    cleaned_payload = load_cleaned_transcript_payload(cleaned_path)
    cleaned_segments = segment_items_from_cleaned_payload(cleaned_payload)
    runtime_review_config = resolve_review_runtime_config(runtime_config or {})
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
        progress_callback=make_review_progress_callback(),
        debug_context={
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "review_input_source": "cleaned_json_backfill",
        },
    )
    review_metadata = review_result["metadata"]

    host_speaker = cleaned_payload.get("host_original_speaker_id")
    speaker_mapping = {
        str(key): str(value)
        for key, value in (cleaned_payload.get("speaker_mapping") or {}).items()
        if value not in ("", None)
    }
    resolved_host_label = speaker_mapping.get(str(host_speaker), "HOST") if host_speaker else "HOST"
    host_output_labels = {resolved_host_label, "HOST"}
    episode_metadata = cleaned_payload.get("metadata") if isinstance(cleaned_payload.get("metadata"), dict) else {}
    durations = {
        str(key): float(value)
        for key, value in (cleaned_payload.get("speaker_durations_seconds") or {}).items()
        if value not in ("", None)
    }
    known_assignments = {
        str(key): value
        for key, value in (cleaned_payload.get("known_speaker_assignments") or {}).items()
        if isinstance(value, dict)
    }
    diarized_turns = [
        turn for turn in (cleaned_payload.get("diarization_turns") or []) if isinstance(turn, dict)
    ]
    info_payload = cleaned_payload.get("transcription") if isinstance(cleaned_payload.get("transcription"), dict) else {}
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
    print_episode_stage(3, 3, "writing reviewed outputs")
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
        "model",
        "language",
        "device",
        "compute_type",
        "beam_size",
        "batch_size",
        "diarization_model",
        "speaker_model",
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
    payload["preferred_terms"] = load_preferred_terms(payload.get("preferred_terms_file"))
    payload["filename_date"] = {
        "preset": getattr(args, "filename_date_preset", "strict_iso"),
        "position": getattr(args, "filename_date_position", "last"),
        "formats": getattr(args, "filename_date_formats", None),
    }
    payload["review_runtime"] = resolve_review_runtime_config(payload)
    return payload


def write_run_reports(output_dir: Path, rows: List[Dict[str, object]], elapsed_seconds: Optional[float] = None):
    output_write_batch_report_md(
        output_dir / "_batch_report.md",
        rows,
        elapsed_seconds=elapsed_seconds,
    )
    output_write_review_run_report(output_dir, rows, elapsed_seconds=elapsed_seconds)
    output_write_speaker_workflow_report(output_dir, rows)


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


def process_file(
    audio_path: Path,
    output_dir: Path,
    whisper_model: WhisperModel,
    diarization_pipeline: Pipeline,
    verifier: Any,
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
    stage_timings: Dict[str, float] = {}
    print(f"Processing {audio_path.name}")
    print_episode_mode("tier1+tier2" if resolve_review_runtime_config(runtime_config or {}).get("any_review_enabled") else "tier1-only")
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_processing_checkpoint(output_dir, audio_path)
    log_memory_usage("before_transcription")

    transcription_started = time.perf_counter()
    print_episode_stage(1, 5, "transcription")
    segments, info_payload, transcription_reused = run_transcription_stage(
        output_dir=output_dir,
        audio_path=audio_path,
        whisper_model=whisper_model,
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

    diarization_started = time.perf_counter()
    print_episode_stage(2, 5, "diarization")
    diarized_turns, diarization_reused = run_diarization_stage(
        output_dir=output_dir,
        audio_path=audio_path,
        diarization_pipeline=diarization_pipeline,
        num_speakers=num_speakers,
        resume_intermediates=resume_intermediates,
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

    print_episode_stage(3, 5, "speaker matching")
    matching_started = time.perf_counter()
    existing_profile = load_host_profile(host_profile_path)
    host_speaker, speaker_embeddings, updated_profile, durations, similarity_scores = choose_host_speaker(
        verifier=verifier,
        audio_path=str(audio_path),
        diarized_turns=diarized_turns,
        host_reference_path=host_reference,
        existing_profile=existing_profile,
        host_threshold=host_threshold,
        assume_dominant=assume_dominant,
        max_embedding_seconds=max_embedding_seconds,
        min_host_seconds=min_host_seconds,
    )

    known_assignments = match_known_speakers(
        speaker_embeddings=speaker_embeddings,
        known_profiles=known_speaker_profiles,
        threshold=host_threshold,
    )
    known_host_speaker = next(
        (speaker_id for speaker_id, assignment in known_assignments.items() if assignment.get("is_host")),
        None,
    )
    if known_host_speaker:
        host_speaker = known_host_speaker
    updated_profile = final_host_profile_update(
        existing_profile,
        speaker_embeddings,
        host_speaker,
        updated_profile,
    )

    speaker_mapping = rename_speakers(
        segments,
        diarized_turns,
        host_speaker,
        durations,
        known_assignments=known_assignments,
    )
    current_label_scores = {
        speaker_mapping.get(speaker_id, speaker_id): score
        for speaker_id, score in similarity_scores.items()
    }
    drift_alerts = detect_speaker_similarity_drift(
        current_label_scores,
        historical_similarity_scores or {},
    )
    filename_date_config = (runtime_config or {}).get("filename_date", {})
    episode_metadata_for_review = build_episode_metadata(str(audio_path), filename_date_config)
    normalized_segments, replacement_events = coalesce_segments(segments, replacement_map)
    correction_path = correction_path_for_audio(corrections_dir, audio_path)
    manual_corrections = apply_manual_corrections(normalized_segments, correction_path)
    if correction_path:
        print(f"  manual corrections applied: {manual_corrections} from {correction_path}")
    cleaned_segments, cleanup_edits = build_cleaned_segments(normalized_segments, level=cleanup_level)
    runtime_review_config = resolve_review_runtime_config(runtime_config or {})
    print_episode_stage(4, 5, "review")
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
        progress_callback=make_review_progress_callback(),
        debug_context={
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "review_input_source": "inline_cleaned_segments",
        },
    )
    resolved_host_label = speaker_mapping.get(host_speaker, "HOST") if host_speaker else "HOST"
    host_output_labels = {resolved_host_label, "HOST"}
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

    print_episode_stage(5, 5, "writing outputs")
    writing_started = time.perf_counter()
    base_name = audio_path.stem
    filename_date_config = (runtime_config or {}).get("filename_date", {})
    episode_metadata = build_episode_metadata(str(audio_path), filename_date_config)
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
    cleaned_metadata = {**episode_metadata, "text_version": "cleaned"}
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
    if updated_profile is not None and host_speaker is not None:
        save_host_profile(host_profile_path, updated_profile, str(audio_path))
    stage_timings["writing"] = time.perf_counter() - writing_started
    print(f"  writing complete in {stage_timings['writing']:.1f}s")
    log_memory_usage("after_writing")

    total_segments = len(normalized_segments)
    host_segments = sum(1 for segment in normalized_segments if segment.speaker in host_output_labels)
    print(f"  review rows: {len(review_rows)}")
    print(f"  speaker segments: {total_segments}")
    print(f"  host segments: {host_segments}")
    print(f"  cleaned text edits: {len(cleanup_edits)}")
    print(f"  manual corrections: {manual_corrections}")
    print(f"  host detected: {host_speaker is not None}")
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
    summary_row["cleanup_level"] = cleanup_level
    summary_row["cleanup_edit_count"] = len(cleanup_edits)
    summary_row["manual_correction_count"] = manual_corrections
    summary_row["processing_seconds"] = round(time.perf_counter() - file_started, 2)
    summary_row["transcription_artifact_reused"] = transcription_reused
    summary_row["diarization_artifact_reused"] = diarization_reused
    warnings_for_language = language_model_warnings(info_payload, language)
    summary_row["language_model_warnings"] = "; ".join(warnings_for_language)
    apply_review_metadata_to_summary(summary_row, review_result)
    apply_speaker_risk_flags_to_summary(summary_row, speaker_mapping)
    summary_row["processing_mode"] = "tier1+tier2" if resolve_review_runtime_config(runtime_config or {}).get("any_review_enabled") else "tier1-only"
    summary_row["tier1_reused_from_existing"] = False
    summary_row["review_backfilled_from_cleaned_json"] = False
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
    output_write_output_manifest(
        output_dir / f"{base_name}_manifest.json",
        source_file=str(audio_path),
        source_fingerprint=audio_file_fingerprint(audio_path),
        config=runtime_config or {},
        outputs=outputs,
        timings=stage_timings,
        summary=summary_row,
    )
    clear_processing_checkpoint(output_dir, audio_path)
    if not archive_debug_artifacts:
        state_clear_stage_artifacts(output_dir, audio_path)
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


def estimate_audio_eta(processed_audio_seconds: float, elapsed_seconds: float, remaining_audio_seconds: float) -> Optional[float]:
    if processed_audio_seconds <= 0 or elapsed_seconds <= 0:
        return None
    seconds_per_audio_second = elapsed_seconds / processed_audio_seconds
    return seconds_per_audio_second * remaining_audio_seconds


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
        "--model",
        args.model,
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
        "--diarization-model",
        args.diarization_model,
        "--speaker-model",
        args.speaker_model,
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

    if args.hf_token:
        command.extend(["--hf-token", args.hf_token])
    if args.host_reference:
        command.extend(["--host-reference", args.host_reference])
    if args.known_speakers_dir:
        command.extend(["--known-speakers-dir", args.known_speakers_dir])
    if args.preferred_terms_file:
        command.extend(["--preferred-terms-file", args.preferred_terms_file])
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
    if any(episode_states[audio_path.name]["state"] == "needs_tier1" for audio_path in pending_audio_files):
        load_replacement_map(args.replacement_map_json)
    total_files = len(audio_files)
    batch_started = time.perf_counter()
    durations = audio_duration_map(audio_files)
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
        average_seconds = elapsed / (index - 1) if index > 1 else None
        remaining_files = total_files - index + 1
        remaining_audio_seconds = sum(
            duration or 0.0
            for path_text, duration in durations.items()
            if Path(path_text) in audio_files[index - 1 :]
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
            processed_audio_seconds += duration_seconds or 0.0
            continue
        if state_info["state"] == "needs_tier2_only" and state_info.get("missing_review_stages"):
            print(
                f"Review shortfall for {audio_path.name}: "
                f"{', '.join(state_info['missing_review_stages'])}"
            )
        print(
            f"Processing mode for {audio_path.name}: "
            f"{'tier2-only backfill' if state_info['state'] == 'needs_tier2_only' else 'tier1+tier2'}"
        )

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
            if refreshed_state["state"] == "complete":
                print(
                    f"Child process for {audio_path.name} exited with code {result.returncode} "
                    "after writing all expected outputs; continuing batch."
                )
                continue
            raise RuntimeError(f"Child process failed for {audio_path.name} with exit code {result.returncode}.")
        processed_audio_seconds += duration_seconds or 0.0

    existing_summary_rows = state_load_episode_summary_rows(summary_path, normalize_episode_summary_row)
    write_run_reports(
        output_dir,
        list(existing_summary_rows.values()),
        elapsed_seconds=time.perf_counter() - batch_started,
    )
    print_final_review_summary(list(existing_summary_rows.values()))
    print(f"Wrote folder summary: {summary_path}")


def load_models(args, device: str):
    whisper_model = WhisperModel(args.model, device=device, compute_type=args.compute_type)

    try:
        diarization_pipeline, resolved_diarization_model = load_diarization_pipeline(
            args.diarization_model, args.hf_token
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

    verifier = load_speaker_verifier(args.speaker_model, device)
    known_speaker_profiles = load_known_speaker_profiles(
        verifier=verifier,
        known_speakers_dir=args.known_speakers_dir,
    )
    return whisper_model, diarization_pipeline, verifier, known_speaker_profiles


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
    needs_tier1 = any(state["state"] == "needs_tier1" for state in episode_states.values())
    preferred_terms: List[str] = []
    initial_prompt = None
    hotwords = None
    replacement_map: Dict[str, List[str]] = {}
    device = None
    whisper_model = diarization_pipeline = verifier = known_speaker_profiles = None
    if needs_tier1:
        preferred_terms = load_preferred_terms(args.preferred_terms_file)
        initial_prompt, hotwords = build_prompt_bias(preferred_terms)
        replacement_map = load_replacement_map(args.replacement_map_json)
        device = get_device(args.device)
        print(f"Using device: {device}")
        whisper_model, diarization_pipeline, verifier, known_speaker_profiles = load_models(args, device)
    else:
        print("No tier-1 work required; running review backfill from existing cleaned JSON outputs only.")

    total_files = len(audio_files)
    is_single_episode_worker = bool(args.input_file) or total_files == 1
    batch_started = time.perf_counter()
    durations = audio_duration_map(audio_files)
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
        average_seconds = elapsed / (index - 1) if index > 1 else None
        remaining_files = total_files - index + 1
        remaining_audio_seconds = sum(
            duration or 0.0
            for path_text, duration in durations.items()
            if Path(path_text) in audio_files[index - 1 :]
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
            processed_audio_seconds += duration_seconds or 0.0
            continue
        if not is_single_episode_worker:
            if state_info["state"] == "needs_tier2_only" and state_info.get("missing_review_stages"):
                print(
                    f"Review shortfall for {audio_path.name}: "
                    f"{', '.join(state_info['missing_review_stages'])}"
                )
            print(
                f"Processing mode for {audio_path.name}: "
                f"{'tier2-only backfill' if state_info['state'] == 'needs_tier2_only' else 'tier1+tier2'}"
            )

        if state_info["state"] == "needs_tier2_only":
            episode_summary = process_review_backfill_from_cleaned_json(
                audio_path=audio_path,
                output_dir=output_dir,
                runtime_config=effective_runtime_config,
                review_calibration_session=review_calibration_session,
                existing_summary_row=episode_summary_rows_by_name.get(audio_path.name),
            )
        else:
            episode_summary = process_file(
                audio_path=audio_path,
                output_dir=output_dir,
                whisper_model=whisper_model,
                diarization_pipeline=diarization_pipeline,
                verifier=verifier,
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
        episode_summary_rows_by_name[audio_path.name] = episode_summary
        historical_similarity_scores = build_historical_similarity_scores(list(episode_summary_rows_by_name.values()))
        processed_files[audio_path.name] = audio_file_fingerprint(audio_path)
        write_episode_summary_csv(summary_path, list(episode_summary_rows_by_name.values()))
        state_save_processed_files(resume_state_path, processed_files)
        save_review_calibration_session(output_dir, review_calibration_session)
        processed_audio_seconds += duration_seconds or 0.0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_episode_summary_csv(summary_path, list(episode_summary_rows_by_name.values()))
    state_save_processed_files(resume_state_path, processed_files)
    save_review_calibration_session(output_dir, review_calibration_session)
    write_run_reports(
        output_dir,
        list(episode_summary_rows_by_name.values()),
        elapsed_seconds=time.perf_counter() - batch_started,
    )
    print_final_review_summary(list(episode_summary_rows_by_name.values()))
    print(f"Wrote folder summary: {summary_path}")


def main():
    """CLI entry point used by the compatibility wrapper and package console script."""

    args = parse_args()

    if args.review_benchmark:
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        run_review_benchmark_mode(args, output_dir)
        return

    if not args.input_dir:
        raise RuntimeError("--input-dir is required unless --review-benchmark is used.")

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
        == "needs_tier1"
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
        if args.input_file:
            # Isolated workers are short-lived by design; skip native-library teardown that can fault after outputs are complete.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
