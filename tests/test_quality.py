import unittest

from podcast_transcribe.quality import classify_segment_text, language_model_warnings, summarize_content_quality


class QualityTests(unittest.TestCase):
    def test_classifies_sponsor_and_repetition(self):
        quality = classify_segment_text(
            "Check out our sponsor and use code TEST. repeat that phrase repeat that phrase repeat that phrase"
        )

        self.assertIn("possible_sponsor_block", quality["tags"])
        self.assertIn("possible_repetition", quality["tags"])

    def test_summarizes_quality_tags(self):
        summary = summarize_content_quality(
            [
                {"tags": ["possible_sponsor_block"]},
                {"tags": ["possible_sponsor_block", "possible_boilerplate"]},
            ]
        )

        self.assertEqual(summary["segment_count"], 2)
        self.assertEqual(summary["tag_counts"]["possible_sponsor_block"], 2)

    def test_language_model_warnings(self):
        warnings = language_model_warnings({"language": "es", "language_probability": 0.45}, "en")

        self.assertTrue(any("does not match" in warning for warning in warnings))
        self.assertTrue(any("low language detection probability" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()

