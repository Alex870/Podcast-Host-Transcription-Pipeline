"""Speaker-embedding provider identities and baseline adapter."""

from __future__ import annotations

from importlib import metadata

from podcast_transcribe.providers.contracts import ProviderIdentity


class SpeechBrainECAPAProvider:
    def __init__(self, verifier, model_name: str):
        self.verifier = verifier
        try:
            version = metadata.version("speechbrain")
        except metadata.PackageNotFoundError:
            version = ""
        self._identity = ProviderIdentity(
            stage="speaker_embedding",
            provider="speechbrain_ecapa",
            model=model_name,
            version=version,
            capabilities={"sample_rate": 16000, "normalized_embeddings": True},
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def encode(self, waveform):
        return self.verifier.encode_batch(waveform)
