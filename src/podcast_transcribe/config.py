import json
from pathlib import Path
from typing import Dict, List, Optional


def load_replacement_map(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}

    raw_text = file_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        lines = raw_text.splitlines()
        bad_line = lines[exc.lineno - 1] if 0 <= exc.lineno - 1 < len(lines) else ""
        pointer = " " * max(exc.colno - 1, 0) + "^"
        raise RuntimeError(
            f"Invalid JSON in replacement map file: {file_path}\n"
            f"JSON error at line {exc.lineno}, column {exc.colno}: {exc.msg}\n"
            f"{bad_line}\n"
            f"{pointer}\n"
            "Replacement maps must be strict JSON: use double quotes, no comments, and no trailing commas."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Replacement map file must contain a JSON object: {file_path}")

    normalized = {}
    for preferred, aliases in payload.items():
        if isinstance(aliases, list):
            normalized[preferred] = [alias for alias in aliases if isinstance(alias, str) and alias.strip()]
    return normalized

