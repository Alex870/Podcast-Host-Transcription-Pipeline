"""Redacted clean-machine diagnostics for M6 operations."""

from __future__ import annotations
import ctypes, hashlib, importlib.util, json, os, platform, shutil, socket, sys, tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

CONTRACT_VERSION = "runtime-preflight-1.0"
DEPENDENCY_COMMAND = "python -m pip install -r podcast_transcribe_requirements.txt"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _tool(name: str, required: bool) -> dict[str, Any]:
    found = shutil.which(name)
    commands = {
        "ffmpeg": "winget install --id Gyan.FFmpeg -e",
        "node": "winget install --id OpenJS.NodeJS.LTS -e",
    }
    return {
        "capability": name,
        "kind": "executable",
        "requirement": "required" if required else "optional",
        "status": "available" if found else "unavailable",
        "version_probe": "operator-run",
        "remediation": None if found else f"Install {name} and add it to PATH.",
        "remediation_command": None if found else commands.get(name),
    }


def _module(name: str, required: bool) -> dict[str, Any]:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        found = False
    return {
        "capability": name,
        "kind": "python_module",
        "requirement": "required" if required else "optional",
        "status": "available" if found else "unavailable",
        "remediation": (
            None if found else f"Install the pinned dependency providing {name}."
        ),
        "remediation_command": None if found else DEPENDENCY_COMMAND,
    }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _total_memory_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None


def build_preflight(
    component: str,
    workspace: Path,
    *,
    required_tools: Iterable[str] = (),
    optional_tools: Iterable[str] = (),
    required_modules: Iterable[str] = (),
    optional_modules: Iterable[str] = (),
    service_ports: Mapping[str, int] | None = None,
    minimum_free_bytes: int = 2_000_000_000,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(workspace)
    writable = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=workspace, prefix=".m6-write-", delete=True
        ):
            writable = True
    except OSError:
        pass
    capabilities = [
        {
            "capability": "python",
            "kind": "runtime",
            "requirement": "required",
            "status": "available" if sys.version_info >= (3, 10) else "incompatible",
            "version": platform.python_version(),
            "remediation": (
                None if sys.version_info >= (3, 10) else "Install Python 3.10 or newer."
            ),
            "remediation_command": (
                None
                if sys.version_info >= (3, 10)
                else "winget install --id Python.Python.3.12 -e"
            ),
        },
        *(_tool(name, True) for name in required_tools),
        *(_tool(name, False) for name in optional_tools),
        *(_module(name, True) for name in required_modules),
        *(_module(name, False) for name in optional_modules),
    ]
    for name, port in sorted((service_ports or {}).items()):
        sock = socket.socket()
        sock.settimeout(0.15)
        try:
            reachable = sock.connect_ex(("127.0.0.1", int(port))) == 0
        finally:
            sock.close()
        capabilities.append(
            {
                "capability": name,
                "kind": "local_service",
                "requirement": "optional",
                "status": "available" if reachable else "user_action",
                "port": int(port),
                "remediation": (
                    None
                    if reachable
                    else f"Start {name} on localhost:{port} when model-backed operation is required."
                ),
            }
        )
    blockers = []
    if not writable:
        blockers.append("workspace_not_writable")
    if disk.free < minimum_free_bytes:
        blockers.append("insufficient_free_disk")
    blockers.extend(
        f"missing_required:{item['capability']}"
        for item in capabilities
        if item["requirement"] == "required" and item["status"] != "available"
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "component": component,
        "profile": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "total_memory_bytes": _total_memory_bytes(),
            "cuda": (
                "available"
                if _module_available("torch") and _cuda_available()
                else "unavailable_or_unverified"
            ),
        },
        "workspace": {
            "label": workspace.name,
            "writable": writable,
            "free_bytes": disk.free,
            "minimum_free_bytes": minimum_free_bytes,
        },
        "capabilities": capabilities,
        "offline": {"network_probe_performed": False, "implicit_downloads": False},
        "blockers": blockers,
        "supported": not blockers,
        "redaction": {
            "absolute_paths_omitted": True,
            "environment_values_omitted": True,
            "credentials_omitted": True,
        },
    }
    report["report_id"] = "preflight_" + _hash(report)
    return report


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def write_report(path: Path, report: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path
