"""Diarization provider identity for the established pyannote orchestration."""

from __future__ import annotations

from importlib import metadata
import time

from podcast_transcribe.providers.contracts import ProviderIdentity
from podcast_transcribe.providers.governance import invocation_metadata


def pyannote_provider_identity(model_name: str, model_revision: str = "") -> ProviderIdentity:
    try:
        version = metadata.version("pyannote.audio")
    except metadata.PackageNotFoundError:
        version = ""
    return ProviderIdentity(
        stage="diarization",
        provider="pyannote",
        model=model_name,
        version=version,
        model_revision=model_revision,
        confidence_semantics="turn boundaries and labels are categorical; no cross-provider confidence comparison",
        license="model-specific gated Hugging Face terms",
        capabilities={
            "exclusive_diarization": True,
            "chunked_fallback": True,
            "learned_long_file_routing": True,
            "language": False,
            "timestamps": True,
            "word_alignment": False,
            "streaming": False,
            "speaker_attribution": True,
            "overlap": True,
            "device_support": ["cpu", "cuda"],
        },
    )


class PyannoteDiarizationProvider:
    """Provider boundary for callers that do not need the CLI's fallback router."""

    def __init__(self, pipeline, model_name: str):
        self.pipeline = pipeline
        self._identity = pyannote_provider_identity(model_name)

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def diarize(self, audio_path: str, num_speakers=None):
        from podcast_transcribe.providers.contracts import StageResult

        started_at = time.perf_counter()
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        result = self.pipeline(audio_path, **kwargs)
        annotation = getattr(result, "speaker_diarization", result)
        turns = [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        invocation = invocation_metadata(
            audio_path=audio_path,
            preprocessing={"num_speakers": num_speakers, "mode": "global"},
            execution={"device": "pipeline_configured", "precision": "provider_default", "batch_size": 1},
            started_at=started_at,
        )
        return StageResult(turns, self.identity, {"mode": "global", **invocation})
