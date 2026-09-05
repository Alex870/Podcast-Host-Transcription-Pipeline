"""Stable internal contracts for interchangeable speech pipeline providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
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
    model_revision: str = ""
    confidence_semantics: str = "unavailable"
    license: str = "unknown"
    acquisition: str = "explicit_download"
    network_boundary: str = "local_inference"
    privacy_boundary: str = "audio_remains_local"
    cache_policy: str = "provider_model_revision"

    def to_payload(self) -> Dict[str, object]:
        return asdict(self)

    def validate_for_experiment(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError(f"Provider {self.stage} requires canonical provider and model IDs.")
        if not self.model_revision.strip():
            raise ValueError(
                f"Provider {self.stage}/{self.provider}/{self.model} has no immutable model_revision; "
                "pin a resolved revision before creating a shadow experiment."
            )
        if self.acquisition != "bundled" and not re.fullmatch(r"[0-9a-fA-F]{7,64}", self.model_revision.strip()):
            raise ValueError(
                f"Provider {self.stage}/{self.provider}/{self.model} revision {self.model_revision!r} is not an immutable commit hash."
            )
        if self.acquisition not in {"bundled", "explicit_download", "local_path"}:
            raise ValueError(f"Unsupported provider acquisition policy: {self.acquisition}")


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
