import re
from collections import Counter
from typing import Dict, Iterable, List


SPONSOR_TERMS = {
    "sponsor",
    "sponsors",
    "promo code",
    "discount code",
    "check out",
    "use code",
    "support the show",
}
MUSIC_TERMS = {"music", "intro music", "outro music", "theme song", "song"}
BOILERPLATE_TERMS = {"welcome everyone", "like and subscribe", "links are in the description"}


def classify_segment_text(text: str, duration_seconds: float = 0.0) -> Dict[str, object]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    lower = normalized.lower()
    tags: List[str] = []

    if not normalized:
        tags.append("empty")
    if duration_seconds >= 20 and len(normalized) < 12:
        tags.append("possible_silence_or_non_speech")
    if any(term in lower for term in SPONSOR_TERMS):
        tags.append("possible_sponsor_block")
    if any(term in lower for term in MUSIC_TERMS):
        tags.append("possible_music_or_transition")
    if any(term in lower for term in BOILERPLATE_TERMS):
        tags.append("possible_boilerplate")
    if _has_repeated_phrase(lower):
        tags.append("possible_repetition")

    return {
        "tags": sorted(set(tags)),
        "character_count": len(normalized),
        "duration_seconds": round(float(duration_seconds or 0.0), 3),
    }


def _has_repeated_phrase(text: str) -> bool:
    words = re.findall(r"[a-z0-9']+", text)
    if len(words) < 8:
        return False
    trigrams = [" ".join(words[index : index + 3]) for index in range(len(words) - 2)]
    counts = Counter(trigrams)
    return any(count >= 3 for count in counts.values())


def summarize_content_quality(segment_quality: Iterable[Dict[str, object]]) -> Dict[str, object]:
    tag_counts = Counter()
    total = 0
    for item in segment_quality:
        total += 1
        for tag in item.get("tags", []) or []:
            tag_counts[str(tag)] += 1
    return {
        "segment_count": total,
        "tag_counts": dict(sorted(tag_counts.items())),
    }


def language_model_warnings(info_payload: Dict[str, object], expected_language: str) -> List[str]:
    warnings = []
    expected = (expected_language or "").strip().lower()
    detected = str(info_payload.get("language") or info_payload.get("detected_language") or "").strip().lower()
    probability = info_payload.get("language_probability")

    if expected and detected and detected != expected:
        warnings.append(f"detected language '{detected}' does not match configured language '{expected}'")
    try:
        if probability is not None and float(probability) < 0.7:
            warnings.append(f"low language detection probability ({float(probability):.2f})")
    except (TypeError, ValueError):
        warnings.append("invalid language_probability value")
    return warnings

