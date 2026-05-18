import unittest
from types import SimpleNamespace

from podcast_transcribe.cleanup import build_cleaned_segments, clean_speech_text


class SpeechCleanupTests(unittest.TestCase):
    def test_removes_immediate_repeated_words(self):
        self.assertEqual(
            clean_speech_text("Anyway, so you can get the, the doll with the removal..."),
            "Anyway, so you can get the doll with the removal...",
        )
        self.assertEqual(
            clean_speech_text("Otherwise, otherwise, all that production goes to shit."),
            "Otherwise, all that production goes to shit.",
        )

    def test_removes_small_dead_end_fragments(self):
        self.assertEqual(
            clean_speech_text(
                "They talked for a couple hours, and Putin, and we, We have, like, the transcripts from the Russian side."
            ),
            "They talked for a couple hours, and Putin, and we have the transcripts from the Russian side.",
        )
        self.assertEqual(
            clean_speech_text(
                "And Putin basically, you know, he thanked Trump and congratulated him on this, that, and the other, you know, very diplomatic."
            ),
            "And Putin basically thanked Trump and congratulated him on this, that, and the other, you know, very diplomatic.",
        )
        self.assertEqual(
            clean_speech_text(
                "And so that's, now, I don't have any proof of this, but I, if you notice in 2025, things were quiet."
            ),
            "Now, I don't have any proof of this, but if you notice in 2025, things were quiet.",
        )

    def test_build_cleaned_segments_preserves_original_text(self):
        segments = [SimpleNamespace(text="Otherwise, otherwise, this changed.", speaker="HOST")]

        cleaned_segments, edits = build_cleaned_segments(segments)

        self.assertEqual(cleaned_segments[0].text, "Otherwise, this changed.")
        self.assertEqual(cleaned_segments[0].original_text, "Otherwise, otherwise, this changed.")
        self.assertTrue(cleaned_segments[0].cleanup_applied)
        self.assertEqual(edits, [("Otherwise, otherwise, this changed.", "Otherwise, this changed.")])

    def test_cleanup_levels_can_be_disabled_or_conservative(self):
        self.assertEqual(
            clean_speech_text("Otherwise, otherwise, this changed.", level="disabled"),
            "Otherwise, otherwise, this changed.",
        )
        self.assertEqual(
            clean_speech_text("And so that's, now, this changed.", level="conservative"),
            "And so that's, now, this changed.",
        )
        self.assertEqual(
            clean_speech_text("And so that's, now, this changed.", level="normal"),
            "Now, this changed.",
        )


if __name__ == "__main__":
    unittest.main()

