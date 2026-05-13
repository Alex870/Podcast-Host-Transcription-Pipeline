import unittest
from collections import defaultdict
from types import SimpleNamespace


def apply_missing_speaker_fallback(segments):
    for segment in segments:
        overlap_by_speaker = defaultdict(float)
        if overlap_by_speaker:
            segment.speaker = max(overlap_by_speaker.items(), key=lambda item: item[1])[0]
        elif not segment.speaker:
            segment.speaker = "UNKNOWN"

        for word in segment.words:
            if word.start is None or word.end is None:
                word.speaker = segment.speaker
                continue
            word_overlap = defaultdict(float)
            if word_overlap:
                word.speaker = max(word_overlap.items(), key=lambda item: item[1])[0]
            else:
                word.speaker = segment.speaker
            if not word.speaker:
                word.speaker = "UNKNOWN"


class MissingSpeakerFallbackTests(unittest.TestCase):
    def test_segment_and_words_fall_back_to_unknown(self):
        segment = SimpleNamespace(
            speaker=None,
            words=[
                SimpleNamespace(start=None, end=None, speaker=None),
                SimpleNamespace(start=1.0, end=1.5, speaker=None),
            ],
        )

        apply_missing_speaker_fallback([segment])

        self.assertEqual(segment.speaker, "UNKNOWN")
        self.assertEqual(segment.words[0].speaker, "UNKNOWN")
        self.assertEqual(segment.words[1].speaker, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
