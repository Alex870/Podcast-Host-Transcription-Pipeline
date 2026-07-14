"""Diarization provider identity for the established pyannote orchestration."""

from __future__ import annotations

from importlib import metadata

from podcast_transcribe.providers.contracts import ProviderIdentity


def pyannote_provider_identity(model_name: str) -> ProviderIdentity:
    try:
        version = metadata.version("pyannote.audio")
    except metadata.PackageNotFoundError:
        version = ""
    return ProviderIdentity(
        stage="diarization",
        provider="pyannote",
        model=model_name,
        version=version,
        capabilities={
            "exclusive_diarization": True,
            "chunked_fallback": True,
            "learned_long_file_routing": True,
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

        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        result = self.pipeline(audio_path, **kwargs)
        annotation = getattr(result, "speaker_diarization", result)
        turns = [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        return StageResult(turns, self.identity, {"mode": "global"})
