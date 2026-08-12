"""Operational governance for optional speech-provider artifacts and runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from podcast_transcribe.providers.contracts import ProviderIdentity


SPEECH_PROVIDER_RUN_VERSION = "1.0"


@dataclass(frozen=True)
class ExecutionProfile:
    requested_device: str
    resolved_device: str
    precision: str
    batch_size: int
    fallback_reason: str = ""


def resolve_execution_profile(
    requested_device: str = "auto",
    requested_batch_size: int = 0,
    *,
    cuda_available: Optional[bool] = None,
    cpu_count: Optional[int] = None,
) -> ExecutionProfile:
    requested = str(requested_device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device override: {requested_device}")
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    fallback = ""
    if requested == "cuda" and not cuda_available:
        resolved, fallback = "cpu", "CUDA was requested but unavailable; using CPU."
    elif requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
        fallback = "" if cuda_available else "CUDA unavailable; automatic CPU fallback."
    else:
        resolved = requested
    if requested_batch_size > 0:
        batch = requested_batch_size
    elif resolved == "cuda":
        batch = 8
    else:
        cores = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
        batch = 2 if cores >= 8 else 1
    return ExecutionProfile(requested, resolved, "float16" if resolved == "cuda" else "float32", batch, fallback)


def artifact_directory(cache_root: Path, identity: ProviderIdentity) -> Path:
    digest = hashlib.sha256(
        f"{identity.provider}\0{identity.model}\0{identity.model_revision}".encode("utf-8")
    ).hexdigest()[:16]
    return Path(cache_root) / identity.stage / f"{identity.provider}-{digest}"


def provider_preflight(cache_root: Path, identity: ProviderIdentity) -> Dict[str, object]:
    identity.validate_for_experiment()
    location = artifact_directory(cache_root, identity)
    if identity.acquisition == "bundled":
        return {
            "contract_version": "provider-preflight-1.0",
            "identity": identity.to_payload(),
            "artifact_path": "",
            "available": True,
            "revision_matches": True,
            "interrupted": False,
            "diagnostic": "bundled",
        }
    receipt_path = location / "acquisition.json"
    receipt = {}
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipt = {}
    revision_matches = str(receipt.get("model_revision") or "") == identity.model_revision
    complete = bool(receipt.get("complete")) and revision_matches
    partials = list(location.parent.glob(location.name + ".partial-*")) if location.parent.exists() else []
    return {
        "contract_version": "provider-preflight-1.0",
        "identity": identity.to_payload(),
        "artifact_path": str(location),
        "available": complete,
        "revision_matches": revision_matches,
        "interrupted": (location.exists() and not complete) or bool(partials),
        "diagnostic": "ready" if complete else "explicit download required",
    }


def acquire_provider_artifact(
    cache_root: Path,
    identity: ProviderIdentity,
    *,
    token: Optional[str] = None,
    downloader: Optional[Callable[..., str]] = None,
) -> Dict[str, object]:
    """Acquire an artifact only from an explicit caller action, then publish atomically."""

    identity.validate_for_experiment()
    destination = artifact_directory(cache_root, identity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    try:
        if downloader is None:
            from huggingface_hub import snapshot_download

            downloader = snapshot_download
        downloaded = downloader(
            repo_id=identity.model,
            revision=identity.model_revision,
            token=token,
            local_dir=str(staging / "model"),
        )
        receipt = {
            "contract_version": "provider-acquisition-1.0",
            "complete": True,
            "provider": identity.provider,
            "model": identity.model,
            "model_revision": identity.model_revision,
            "downloaded_path": str(downloaded),
        }
        (staging / "acquisition.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        if destination.exists():
            existing = provider_preflight(cache_root, identity)
            if existing["available"]:
                shutil.rmtree(staging)
                return existing
            shutil.rmtree(destination)
        staging.replace(destination)
    except Exception:
        # Keep a visible partial directory for diagnostics/resume, never mark it available.
        raise
    return provider_preflight(cache_root, identity)


def input_audio_identity(path: Path) -> Dict[str, object]:
    if not Path(path).is_file():
        return {"path": str(Path(path).resolve()), "missing": True}
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest(), "size_bytes": stat.st_size}


def invocation_metadata(
    *,
    audio_path: str,
    preprocessing: Dict[str, object],
    execution: Dict[str, object],
    started_at: float,
    warnings: Optional[Iterable[str]] = None,
    failures: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    preprocessing_fingerprint = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    peak_memory = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_memory = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return {
        "input_audio_identity": input_audio_identity(Path(audio_path)),
        "preprocessing": {**preprocessing, "fingerprint": preprocessing_fingerprint},
        "execution": {
            **execution,
            "runtime_seconds": max(0.0, time.perf_counter() - started_at),
            "peak_memory_bytes": peak_memory,
            "warnings": list(warnings or []),
            "failures": list(failures or []),
        },
    }


def build_speech_provider_run(
    *,
    run_id: str,
    evaluation_pack: Dict[str, object],
    audio_identity: Dict[str, object],
    preprocessing: Dict[str, object],
    providers: Iterable[ProviderIdentity],
    execution: Dict[str, object],
    outputs: Dict[str, object],
    metrics: Dict[str, object],
    parent_run_id: str = "",
) -> Dict[str, object]:
    identities = list(providers)
    for identity in identities:
        identity.validate_for_experiment()
    if not evaluation_pack.get("pack_id") or not evaluation_pack.get("source_identity"):
        raise ValueError("Speech experiments require exact pack_id and source_identity values.")
    preprocess_fingerprint = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "contract_version": SPEECH_PROVIDER_RUN_VERSION,
        "run_id": str(run_id),
        "parent_run_id": str(parent_run_id),
        "immutable": True,
        "shadow_only": True,
        "evaluation_pack": evaluation_pack,
        "audio_identity": audio_identity,
        "preprocessing": {**preprocessing, "fingerprint": preprocess_fingerprint},
        "providers": [identity.to_payload() for identity in identities],
        "execution": execution,
        "outputs": outputs,
        "metrics": metrics,
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "confidence_comparison_allowed": False,
        "human_corrections_authoritative": True,
    }


def write_immutable_speech_run(root: Path, payload: Dict[str, object]) -> Path:
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Speech provider run_id is required.")
    target = Path(root) / run_id
    if target.exists():
        raise FileExistsError(f"Immutable speech run already exists: {target}")
    target.mkdir(parents=True)
    path = target / "speech-provider-run.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
