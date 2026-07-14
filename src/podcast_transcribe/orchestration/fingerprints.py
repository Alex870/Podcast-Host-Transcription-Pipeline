"""Stable stage fingerprints used for selective intermediate reuse."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Optional

from podcast_transcribe.providers.contracts import ProviderIdentity


STAGE_FINGERPRINT_VERSION = 1


def stable_payload_hash(payload: Dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_stage_fingerprint(
    stage: str,
    provider: ProviderIdentity,
    config: Optional[Dict[str, object]] = None,
    dependency_fingerprints: Optional[Iterable[Dict[str, object]]] = None,
) -> Dict[str, object]:
    dependencies = list(dependency_fingerprints or [])
    payload = {
        "fingerprint_version": STAGE_FINGERPRINT_VERSION,
        "stage": stage,
        "provider": provider.to_payload(),
        "config": dict(config or {}),
        "dependency_hashes": [str(item.get("hash") or stable_payload_hash(item)) for item in dependencies],
    }
    return {**payload, "hash": stable_payload_hash(payload)}
