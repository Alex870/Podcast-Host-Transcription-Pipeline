"""Stable internal contracts for interchangeable speech pipeline providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Generic, List, Optional, Protocol, TypeVar

from podcast_transcribe.models import SegmentItem


@dataclass(frozen=True)
class ProviderIdentity:
    stage: str
    provider: str
    model: str
    version: str = ""
    implementation_version: int = 1
    capabilities: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, object]:
        return asdict(self)


T = TypeVar("T")


@dataclass
class StageResult(Generic[T]):
    value: T
    provider: ProviderIdentity
    metadata: Dict[str, object] = field(default_factory=dict)


class ASRProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    def transcribe(
        self,
        audio_path: str,
        language: str,
        beam_size: int,
        batch_size: int,
        initial_prompt: Optional[str],
        hotwords: Optional[str],
    ) -> StageResult[List[SegmentItem]]: ...


class AlignmentProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    def align(
        self,
        audio_path: str,
        segments: List[SegmentItem],
        language: str,
    ) -> StageResult[List[SegmentItem]]: ...


class DiarizationProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> StageResult[List[Dict[str, object]]]: ...


class SpeakerEmbeddingProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    def encode(self, waveform) -> object: ...
