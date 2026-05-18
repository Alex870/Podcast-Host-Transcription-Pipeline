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
from podcast_transcribe.outputs import (
    build_episode_metadata,
    write_batch_report_md as output_write_batch_report_md,
    write_json_output as output_write_json_output,
    write_output_manifest as output_write_output_manifest,
    write_review_csv as output_write_review_csv,
    write_speaker_identity_review_csv as output_write_speaker_identity_review_csv,
    write_text_transcript as output_write_text_transcript,
)
from podcast_transcribe.config import load_replacement_map as config_load_replacement_map
from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.quality import language_model_warnings
from podcast_transcribe.state import (
    ARTIFACT_DIRNAME,
    CHECKPOINT_DIRNAME,
    RESUME_STATE_FILENAME,
    SUMMARY_FILENAME,
    audio_file_fingerprint,
    clear_stage_artifacts as state_clear_stage_artifacts,
    expected_output_paths as state_expected_output_paths,
    is_file_already_processed as state_is_file_already_processed,
    load_stage_artifact as state_load_stage_artifact,
    load_episode_summary_rows as state_load_episode_summary_rows,
    load_processed_files as state_load_processed_files,
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


def parse_args():
    """Parse CLI options for parent batch runs and isolated child workers."""

    parser = argparse.ArgumentParser(description="Transcribe podcasts with diarization and host labeling.")
    parser.add_argument("--input-dir", required=True, help="Directory containing audio files to process.")
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
    }
    bool_fields = {
        "host_detected",
        "transcription_artifact_reused",
        "diarization_artifact_reused",
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


def is_file_already_processed(
    audio_path: Path,
    output_dir: Path,
    processed_files: Dict[str, Dict[str, object]],
    existing_summary_rows: Dict[str, Dict[str, object]],
) -> bool:
    return state_is_file_already_processed(
        audio_path,
        output_dir,
        processed_files,
        existing_summary_rows,
    )


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
    ]
    payload = {key: getattr(args, key, None) for key in keys}
    payload["filename_date"] = {
        "preset": getattr(args, "filename_date_preset", "strict_iso"),
        "position": getattr(args, "filename_date_position", "last"),
        "formats": getattr(args, "filename_date_formats", None),
    }
    return payload


def correction_path_for_audio(corrections_dir: Optional[str], audio_path: Path) -> Optional[Path]:
    if not corrections_dir:
        return None
    path = Path(corrections_dir) / f"{audio_path.stem}_corrections.csv"
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
    historical_similarity_scores: Optional[Dict[str, List[float]]] = None,
    resume_intermediates: bool = True,
    archive_debug_artifacts: bool = False,
) -> Dict[str, object]:
    """Process one audio file through all model stages and write its output bundle."""

    file_started = time.perf_counter()
    stage_timings: Dict[str, float] = {}
    print(f"Processing {audio_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_processing_checkpoint(output_dir, audio_path)
    log_memory_usage("before_transcription")

    transcription_started = time.perf_counter()
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

    print("  stage: speaker matching")
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

    print("  stage: writing outputs")
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
    pending_audio_files = [
        audio_path
        for audio_path in audio_files
        if not state_is_file_already_processed(audio_path, output_dir, processed_files, existing_summary_rows)
    ]
    if pending_audio_files:
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

        if state_is_file_already_processed(audio_path, output_dir, processed_files, existing_summary_rows):
            print(f"Skipping completed file: {audio_path.name}")
            processed_audio_seconds += duration_seconds or 0.0
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
            if state_is_file_already_processed(audio_path, output_dir, processed_files, existing_summary_rows):
                print(
                    f"Child process for {audio_path.name} exited with code {result.returncode} "
                    "after writing all expected outputs; continuing batch."
                )
                continue
            raise RuntimeError(f"Child process failed for {audio_path.name} with exit code {result.returncode}.")
        processed_audio_seconds += duration_seconds or 0.0

    existing_summary_rows = state_load_episode_summary_rows(summary_path, normalize_episode_summary_row)
    output_write_batch_report_md(
        output_dir / "_batch_report.md",
        list(existing_summary_rows.values()),
        elapsed_seconds=time.perf_counter() - batch_started,
    )
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

    preferred_terms = load_preferred_terms(args.preferred_terms_file)
    initial_prompt, hotwords = build_prompt_bias(preferred_terms)
    replacement_map = load_replacement_map(args.replacement_map_json)

    device = get_device(args.device)
    print(f"Using device: {device}")
    whisper_model, diarization_pipeline, verifier, known_speaker_profiles = load_models(args, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    resume_state_path = output_dir / RESUME_STATE_FILENAME
    existing_summary_rows = state_load_episode_summary_rows(summary_path, normalize_episode_summary_row)
    processed_files = state_load_processed_files(resume_state_path)
    episode_summary_rows_by_name = dict(existing_summary_rows)
    historical_similarity_scores = build_historical_similarity_scores(list(existing_summary_rows.values()))
    total_files = len(audio_files)
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
        if eta_seconds is not None:
            print(
                f"Batch progress: file {index} of {total_files} "
                f"(estimated remaining {format_timestamp(eta_seconds)}, "
                f"processed_audio={format_timestamp(processed_audio_seconds)})"
            )
        else:
            print(f"Batch progress: file {index} of {total_files}")
        if is_file_already_processed(audio_path, output_dir, processed_files, episode_summary_rows_by_name):
            print(f"Skipping completed file: {audio_path.name}")
            processed_audio_seconds += duration_seconds or 0.0
            continue

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
            runtime_config=runtime_config_payload(args),
            historical_similarity_scores=historical_similarity_scores,
            resume_intermediates=args.resume_intermediates,
            archive_debug_artifacts=args.archive_debug_artifacts,
        )
        episode_summary_rows_by_name[audio_path.name] = episode_summary
        historical_similarity_scores = build_historical_similarity_scores(list(episode_summary_rows_by_name.values()))
        processed_files[audio_path.name] = audio_file_fingerprint(audio_path)
        write_episode_summary_csv(summary_path, list(episode_summary_rows_by_name.values()))
        state_save_processed_files(resume_state_path, processed_files)
        processed_audio_seconds += duration_seconds or 0.0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_episode_summary_csv(summary_path, list(episode_summary_rows_by_name.values()))
    state_save_processed_files(resume_state_path, processed_files)
    output_write_batch_report_md(
        output_dir / "_batch_report.md",
        list(episode_summary_rows_by_name.values()),
        elapsed_seconds=time.perf_counter() - batch_started,
    )
    print(f"Wrote folder summary: {summary_path}")


def main():
    """CLI entry point used by the compatibility wrapper and package console script."""

    args = parse_args()

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

    if not args.hf_token:
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

