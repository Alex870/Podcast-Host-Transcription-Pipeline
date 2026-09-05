"""Word-alignment providers with a dependency-free baseline default."""

from __future__ import annotations

from copy import deepcopy
from importlib import metadata
import time
from typing import Dict, List, Optional

from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.providers.contracts import ProviderIdentity, StageResult
from podcast_transcribe.providers.governance import invocation_metadata


ALIGNMENT_PROVIDERS = {"timestamp_passthrough", "whisperx"}


class TimestampPassthroughAlignmentProvider:
    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            stage="alignment",
            provider="timestamp_passthrough",
            model="asr_native_word_timestamps",
            capabilities={"language": True, "timestamps": True, "word_alignment": True, "streaming": False, "speaker_attribution": False, "overlap": False, "device_support": ["cpu", "cuda"], "forced_alignment": False},
            model_revision="builtin-v1",
            confidence_semantics="inherited from the selected ASR provider",
            license="project license",
            acquisition="bundled",
        )

    def align(self, audio_path: str, segments: List[SegmentItem], language: str) -> StageResult[List[SegmentItem]]:
        started_at = time.perf_counter()
        invocation = invocation_metadata(
            audio_path=audio_path,
            preprocessing={"language": language, "input_segment_count": len(segments), "mode": "passthrough"},
            execution={"device": "inherited", "precision": "inherited", "batch_size": 1},
            started_at=started_at,
        )
        return StageResult(
            value=deepcopy(segments),
            provider=self.identity,
            metadata={"mode": "passthrough", "aligned_word_count": sum(len(segment.words) for segment in segments), **invocation},
        )


class WhisperXAlignmentProvider:
    def __init__(self, device: str, model_name: str = "", model_revision: str = "", local_model_path: str = ""):
        self.device = device
        self.model_name = model_name
        self.model_revision = model_revision
        self.local_model_path = local_model_path
        self._models: Dict[str, object] = {}
        try:
            version = metadata.version("whisperx")
        except metadata.PackageNotFoundError:
            version = "not-installed"
        self._identity = ProviderIdentity(
            stage="alignment",
            provider="whisperx",
            model=model_name or "language_default",
            version=version,
            capabilities={"language": True, "timestamps": True, "word_alignment": True, "streaming": False, "speaker_attribution": False, "overlap": False, "device_support": ["cpu", "cuda"], "forced_alignment": True},
            model_revision=model_revision,
            confidence_semantics="alignment score is provider-specific and not cross-provider calibrated",
            license="model-specific; inspect acquisition receipt",
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def _load(self, language: str):
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "alignment_provider=whisperx requires the optional WhisperX environment. "
                "Install podcast_transcribe_alignment_requirements.txt in a compatible environment."
            ) from exc
        if language not in self._models:
            if not self.local_model_path:
                raise RuntimeError(
                    "WhisperX alignment model acquisition is explicit. Select a pinned model, download it, and pass its local path."
                )
            model, model_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_name=self.local_model_path,
            )
            self._models[language] = (model, model_metadata)
        return whisperx, self._models[language]

    def align(self, audio_path: str, segments: List[SegmentItem], language: str) -> StageResult[List[SegmentItem]]:
        started_at = time.perf_counter()
        whisperx, (align_model, align_metadata) = self._load(language)
        audio = whisperx.load_audio(audio_path)
        input_segments = [
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text}
            for segment in segments
        ]
        aligned = whisperx.align(
            input_segments,
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        aligned_segments = aligned.get("segments") or []
        rebuilt: List[SegmentItem] = []
        for index, source in enumerate(segments):
            item = aligned_segments[index] if index < len(aligned_segments) and isinstance(aligned_segments[index], dict) else {}
            words = []
            for raw_word in item.get("words") or []:
                if not isinstance(raw_word, dict) or not str(raw_word.get("word") or "").strip():
                    continue
                words.append(
                    WordItem(
                        start=raw_word.get("start"),
                        end=raw_word.get("end"),
                        word=str(raw_word.get("word") or ""),
                        speaker=None,
                    )
                )
            rebuilt.append(
                SegmentItem(
                    id=source.id,
                    start=float(item.get("start", source.start)),
                    end=float(item.get("end", source.end)),
                    text=str(item.get("text") or source.text).strip(),
                    speaker=source.speaker,
                    avg_logprob=source.avg_logprob,
                    no_speech_prob=source.no_speech_prob,
                    words=words or deepcopy(source.words),
                )
            )
        invocation = invocation_metadata(
            audio_path=audio_path,
            preprocessing={"language": language, "input_segment_count": len(segments), "mode": "forced_alignment"},
            execution={"device": self.device, "precision": "provider_default", "batch_size": 1},
            started_at=started_at,
        )
        return StageResult(
            value=rebuilt,
            provider=self.identity,
            metadata={
                "mode": "forced_alignment",
                "aligned_word_count": sum(len(segment.words) for segment in rebuilt),
                "language": language,
                "raw_provider_output": aligned_segments,
                "normalization_applied_after_provider_return": True,
                **invocation,
            },
        )


def create_alignment_provider(provider_name: str, device: str, model_name: str = "", model_revision: str = "", local_model_path: str = ""):
    normalized = str(provider_name or "timestamp_passthrough").strip().lower()
    if normalized == "timestamp_passthrough":
        return TimestampPassthroughAlignmentProvider()
    if normalized == "whisperx":
        return WhisperXAlignmentProvider(device=device, model_name=model_name, model_revision=model_revision, local_model_path=local_model_path)
    raise ValueError(f"Unsupported alignment provider: {provider_name}")
