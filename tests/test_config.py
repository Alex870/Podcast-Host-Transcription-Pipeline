import tempfile
import unittest
from pathlib import Path

from podcast_transcribe.config import load_replacement_map, resolve_review_runtime_config


class ConfigTests(unittest.TestCase):
    def test_replacement_map_reports_json_line_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = Path(tmp) / "preferred_replacements.json"
            map_path.write_text('{\n  "Federal Reserve": ["fed"],\n  "Bad": \n}', encoding="utf-8")

            with self.assertRaises(RuntimeError) as context:
                load_replacement_map(str(map_path))

            message = str(context.exception)
            self.assertIn("Invalid JSON in replacement map file", message)
            self.assertIn("line 4, column 1", message)
            self.assertIn("Replacement maps must be strict JSON", message)

    def test_replacement_map_normalizes_alias_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            map_path = Path(tmp) / "preferred_replacements.json"
            map_path.write_text(
                '{"Federal Reserve": ["fed", "", 123], "Ignored": "not a list"}',
                encoding="utf-8",
            )

            self.assertEqual(load_replacement_map(str(map_path)), {"Federal Reserve": ["fed"]})

    def test_high_context_profile_enables_review_defaults(self):
        resolved = resolve_review_runtime_config(
            {
                "runtime_profile": "high_context_5090",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
            }
        )

        self.assertTrue(resolved["transcript_cleanup_review"])
        self.assertTrue(resolved["glossary_correction_review"])
        self.assertTrue(resolved["speaker_consistency_review"])
        self.assertTrue(resolved["episode_qa_review"])
        self.assertTrue(resolved["backend_ready"])

    def test_custom_profile_respects_explicit_flags(self):
        resolved = resolve_review_runtime_config(
            {
                "runtime_profile": "custom",
                "backend": "lm_studio",
                "review_base_url": "http://127.0.0.1:1234",
                "review_model_name": "mistral-review",
                "transcript_cleanup_review": True,
                "glossary_correction_review": False,
                "speaker_consistency_review": True,
                "episode_qa_review": False,
                "review_context_budget": 64000,
            }
        )

        self.assertTrue(resolved["transcript_cleanup_review"])
        self.assertFalse(resolved["glossary_correction_review"])
        self.assertTrue(resolved["speaker_consistency_review"])
        self.assertFalse(resolved["episode_qa_review"])
        self.assertEqual(resolved["max_context_budget"], 64000)

    def test_review_debug_settings_are_preserved(self):
        resolved = resolve_review_runtime_config(
            {
                "runtime_profile": "custom",
                "backend": "vllm",
                "review_debug": True,
                "review_debug_dir": "review-debug",
            }
        )

        self.assertTrue(resolved["review_debug"])
        self.assertEqual(resolved["review_debug_dir"], "review-debug")

    def test_review_calibration_defaults_follow_enabled_review(self):
        resolved = resolve_review_runtime_config(
            {
                "runtime_profile": "high_context_5090",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
            }
        )

        self.assertTrue(resolved["review_auto_calibrate"])
        self.assertTrue(resolved["review_auto_adapt_upward"])


if __name__ == "__main__":
    unittest.main()
