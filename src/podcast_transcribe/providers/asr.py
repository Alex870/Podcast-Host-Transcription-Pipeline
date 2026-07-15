"""ASR provider wrappers.

The faster-whisper provider intentionally accepts the existing transcription
callable so the provider boundary can be introduced without changing baseline
decoding or progress behavior.
"""

from __future__ import annotations

import importlib
import platform
from importlib import metadata
from typing import Callable, Dict, List, Optional, Tuple

from podcast_transcribe.models import SegmentItem
from podcast_transcribe.providers.contracts import ProviderIdentity, StageResult


class FasterWhisperASRProvider:
    def __init__(self, model, model_name: str, transcribe_callable: Callable[..., Tuple[List[SegmentItem], Dict[str, object]]]):
        self.model = model
        self.transcribe_callable = transcribe_callable
        self._identity = ProviderIdentity(
            stage="transcription",
            provider="faster_whisper",
            model=model_name,
            capabilities={"word_timestamps": True, "hotwords": True, "batched_decode": True},
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def transcribe(
        self,
        audio_path: str,
        language: str,
        beam_size: int,
        batch_size: int,
        initial_prompt: Optional[str],
        hotwords: Optional[str],
    ) -> StageResult[List[SegmentItem]]:
        segments, metadata = self.transcribe_callable(
            model=self.model,
            audio_path=audio_path,
            language=language,
            beam_size=beam_size,
            batch_size=batch_size,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        return StageResult(value=segments, provider=self.identity, metadata=metadata)


def parakeet_environment_diagnostics() -> Dict[str, object]:
    """Describe optional Parakeet support without importing the provider runtime."""

    packages = {}
    for name in ("nemo", "torch", "cuda-python", "soundfile"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(getattr(torch.version, "cuda", "") or "")
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        cuda_available = False
        cuda_version = ""
        packages["torch_error"] = str(exc)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "supported": bool(packages.get("nemo") != "not-installed"),
    }


class ParakeetASRProvider:
    """Optional NeMo Parakeet adapter with lazy model import/loading.

    The loader is injectable for tests and for future backend-specific model
    factories. No NeMo or CUDA module is imported at application startup.
    """

    def __init__(self, model_name: str, device: str = "auto", model_loader: Optional[Callable[[], object]] = None):
        self.model_name = model_name
        self.device = device
        self._model_loader = model_loader
        self._model = None
        version = ""
        try:
            version = metadata.version("nemo_toolkit")
        except metadata.PackageNotFoundError:
            pass
        self._identity = ProviderIdentity(
            stage="transcription",
            provider="parakeet",
            model=model_name,
            version=version,
            capabilities={"word_timestamps": True, "batched_decode": True, "lazy_optional_runtime": True},
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def _load(self):
        if self._model is not None:
            return self._model
        if self._model_loader is not None:
            self._model = self._model_loader()
            return self._model
        try:
            asr_models = importlib.import_module("nemo.collections.asr.models")
        except ImportError as exc:
            diagnostics = parakeet_environment_diagnostics()
            raise RuntimeError(
                "asr_provider=parakeet requires optional NVIDIA NeMo. "
                f"Install a compatible NeMo/CUDA environment; diagnostics={diagnostics}"
            ) from exc
        asr_model = getattr(asr_models, "ASRModel", None)
        if asr_model is None or not hasattr(asr_model, "from_pretrained"):
            raise RuntimeError("The installed NeMo package does not expose ASRModel.from_pretrained for Parakeet.")
        self._model = asr_model.from_pretrained(model_name=self.model_name)
        if self.device == "cuda" and hasattr(self._model, "cuda"):
            self._model.cuda()
        elif self.device == "cpu" and hasattr(self._model, "cpu"):
            self._model.cpu()
        return self._model

    @staticmethod
    def _segment_payloads(output) -> List[Dict[str, object]]:
        if isinstance(output, dict):
            if isinstance(output.get("segments"), list):
                return [item for item in output["segments"] if isinstance(item, dict)]
            return [output]
        if isinstance(output, (list, tuple)):
            flattened = []
            for item in output:
                if isinstance(item, dict):
                    flattened.extend(ParakeetASRProvider._segment_payloads(item))
                elif hasattr(item, "text"):
                    flattened.append({"text": getattr(item, "text", ""), "timestamp": getattr(item, "timestamp", None)})
            return flattened
        if hasattr(output, "text"):
            return [{"text": getattr(output, "text", ""), "timestamp": getattr(output, "timestamp", None)}]
        return []

    def transcribe(
        self,
        audio_path: str,
        language: str,
        beam_size: int,
        batch_size: int,
        initial_prompt: Optional[str],
        hotwords: Optional[str],
    ) -> StageResult[List[SegmentItem]]:
        model = self._load()
        try:
            raw = model.transcribe([audio_path], batch_size=batch_size)
        except TypeError:
            raw = model.transcribe([audio_path])
        if isinstance(raw, tuple):
            raw = raw[0]
        raw_items = self._segment_payloads(raw)
        segments: List[SegmentItem] = []
        for index, item in enumerate(raw_items, start=1):
            timestamp = item.get("timestamp") if isinstance(item.get("timestamp"), dict) else {}
            start = item.get("start", timestamp.get("start", (index - 1) * 1.0))
            end = item.get("end", timestamp.get("end", float(start) + 1.0))
            try:
                start_value = float(start)
                end_value = max(start_value + 0.001, float(end))
            except (TypeError, ValueError):
                start_value, end_value = (index - 1) * 1.0, index * 1.0
            segments.append(
                SegmentItem(
                    id=index,
                    start=start_value,
                    end=end_value,
                    text=str(item.get("text") or "").strip(),
                    speaker=None,
                    avg_logprob=None,
                    no_speech_prob=None,
                    words=[],
                )
            )
        return StageResult(
            value=segments,
            provider=self.identity,
            metadata={
                "language": language,
                "mode": "parakeet_optional",
                "diagnostics": parakeet_environment_diagnostics(),
                "prompt_forwarded": bool(initial_prompt or hotwords),
                "beam_size": beam_size,
            },
        )
