"""Core transcript data models shared across pipeline stages."""

from dataclasses import dataclass
from typing import List, Optional


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
