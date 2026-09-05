"""Speaker-embedding provider identities and baseline adapter."""

from __future__ import annotations

from importlib import metadata

from podcast_transcribe.providers.contracts import ProviderIdentity


class SpeechBrainECAPAProvider:
    def __init__(self, verifier, model_name: str, model_revision: str = ""):
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
            model_revision=model_revision,
            confidence_semantics="cosine similarity within one embedding family only",
            license="Apache-2.0 code; model-card data/model terms apply",
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def encode(self, waveform):
        return self.verifier.encode_batch(waveform)


class SpeechBrainXVectorProvider:
    """Optional candidate speaker embedder using SpeechBrain x-vector."""

    def __init__(self, verifier, model_name: str, model_revision: str = ""):
        self.verifier = verifier
        try:
            version = metadata.version("speechbrain")
        except metadata.PackageNotFoundError:
            version = ""
        self._identity = ProviderIdentity(
            stage="speaker_embedding",
            provider="speechbrain_xvector",
            model=model_name,
            version=version,
            capabilities={"sample_rate": 16000, "normalized_embeddings": True, "candidate": True},
            model_revision=model_revision,
            confidence_semantics="cosine similarity within one embedding family only",
            license="Apache-2.0 code; model-card data/model terms apply",
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def encode(self, waveform):
        return self.verifier.encode_batch(waveform)


def speaker_embedding_provider_names():
    return ("speechbrain_ecapa", "speechbrain_xvector")
