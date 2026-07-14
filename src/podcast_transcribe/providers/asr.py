"""ASR provider wrappers.

The faster-whisper provider intentionally accepts the existing transcription
callable so the provider boundary can be introduced without changing baseline
decoding or progress behavior.
"""

from __future__ import annotations

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
