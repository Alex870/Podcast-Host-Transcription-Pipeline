"""Core transcript data models shared across pipeline stages."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class WordItem:
    """Word-level transcript token with timing and speaker attribution."""

    start: Optional[float]
    end: Optional[float]
    word: str
    speaker: Optional[str]


@dataclass
class SegmentItem:
    """Segment-level transcript span passed between transcription, diarization, and output writers."""

    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str]
    avg_logprob: Optional[float]
    no_speech_prob: Optional[float]
    words: List[WordItem]
    original_text: Optional[str] = None
    cleanup_applied: bool = False
    cleanup_level: str = ""
    manual_correction_applied: bool = False
    original_speaker: Optional[str] = None
    llm_reviewed_text: Optional[str] = None
    review_runtime_profile: Optional[str] = None
    review_backend: Optional[str] = None
    review_model_name: Optional[str] = None
    review_stage_flags: Optional[Dict[str, bool]] = None
