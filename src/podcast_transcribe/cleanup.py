import copy
import re
from typing import List, Tuple


CleanupEdit = Tuple[str, str]


def _preserve_case(replacement: str, original: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_repeated_word(match: re.Match) -> str:
    first = match.group("first")
    tail = match.group("tail") or ""
    return f"{first}{tail}"


def clean_speech_text(text: str, level: str = "normal") -> str:
    """Lightly remove repeated words and small speech restarts while preserving meaning."""

    level = (level or "normal").strip().lower()
    if level == "disabled":
        return text
    if level not in {"conservative", "normal", "aggressive"}:
        raise ValueError(f"Unknown cleanup level: {level}")

    cleaned = " ".join(text.split())
    if not cleaned:
        return cleaned

    repeated_word_pattern = re.compile(
        r"\b(?P<first>[A-Za-z][A-Za-z']*)\b\s*,\s*(?P=first)\b(?P<tail>\s+|[,.;:!?])",
        re.IGNORECASE,
    )
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = repeated_word_pattern.sub(_replace_repeated_word, cleaned)

    replacements = []
    if level in {"normal", "aggressive"}:
        replacements.extend(
            [
                (re.compile(r"(?i)\bAnd so that's,\s*now,\s*"), "Now, "),
                (re.compile(r"(?i)\bbut I,\s*if\b"), "but if"),
                (re.compile(r"(?i)\bbasically,\s*you know,\s*(he|she|they)\s+"), "basically "),
                (re.compile(r"(?i),\s*like,\s*"), " "),
                (re.compile(r"(?i)\bwe,\s*we\s+"), "we "),
            ]
        )
    if level == "aggressive":
        replacements.extend(
            [
                (re.compile(r"(?i)\b(?:um|uh|er|ah),?\s+"), ""),
                (re.compile(r"(?i),\s*you know,\s*"), ", "),
                (re.compile(r"(?i),\s*I mean,\s*"), ", "),
            ]
        )

    for pattern, replacement in replacements:
        cleaned = pattern.sub(lambda match: _preserve_case(replacement, match.group(0)), cleaned)

    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def build_cleaned_segments(segments, level: str = "normal") -> Tuple[List[object], List[CleanupEdit]]:
    cleaned_segments = []
    edits = []
    for segment in segments:
        cleaned_segment = copy.deepcopy(segment)
        original_text = cleaned_segment.text
        cleaned_text = clean_speech_text(original_text, level=level)
        cleaned_segment.text = cleaned_text
        cleaned_segment.original_text = original_text
        cleaned_segment.cleanup_applied = cleaned_text != original_text
        cleaned_segment.cleanup_level = level
        if cleaned_segment.cleanup_applied:
            edits.append((original_text, cleaned_text))
        cleaned_segments.append(cleaned_segment)
    return cleaned_segments, edits

