import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_transcribe.review_benchmark import (
    default_fixture_dir,
    load_benchmark_fixtures,
    run_review_benchmark,
    write_review_benchmark_reports,
)


class ReviewBenchmarkTests(unittest.TestCase):
    def _runtime_config(self):
        return {
            "runtime_profile": "high_context_5090",
            "backend": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "transcript_cleanup_review": True,
            "glossary_correction_review": True,
            "speaker_consistency_review": True,
            "episode_qa_review": True,
        }

    def _fake_stage_call(self, window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
        stage_name = stage_definition["name"]
        texts = " ".join(str(segment.text) for segment in window).lower()
        if stage_name == "transcript_cleanup_review" and "chromadb remains the storage layer" in texts:
            return {
                "reviewed_segments": [{"id": 1, "text": "Chroma DB remains the storage layer for this workflow."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        if stage_name == "transcript_cleanup_review" and "we we tested the fallback once" in texts:
            return {
                "reviewed_segments": [{"id": 2, "text": "We tested the fallback once and moved on."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        if stage_name == "glossary_correction_review" and "chroma db import performance" in texts:
            return {
                "reviewed_segments": [{"id": 1, "text": "Today we are talking about ChromaDB import performance."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        if stage_name == "glossary_correction_review" and "later i casually call it chroma db" in texts:
            return {
                "reviewed_segments": [{"id": 2, "text": "Later I casually call it ChromaDB in conversation."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        if stage_name == "glossary_correction_review" and "lm studio on the desktop box" in texts:
            return {
                "reviewed_segments": [{"id": 1, "text": "We benchmarked LM Studio on the desktop box."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        if stage_name == "speaker_consistency_review" and "thanks for having me on" in texts:
            return {
                "reviewed_segments": [{"id": 2, "text": "Thanks for having me on.", "speaker": "GUEST"}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        if stage_name == "episode_qa_review" and "later on i shorten that to chroma db" in texts:
            return {
                "reviewed_segments": [{"id": 2, "text": "Later on I shorten that to ChromaDB once or twice."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        if stage_name == "episode_qa_review" and "much later i lazily shorten that to chroma db" in texts:
            return {
                "reviewed_segments": [{"id": 3, "text": "Much later I lazily shorten that to ChromaDB during the recap."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }
        return {
            "reviewed_segments": [],
            "corrected_segment_count": 0,
            "episode_notes": [],
        }

    def test_default_fixture_dir_contains_repo_fixtures(self):
        fixtures = load_benchmark_fixtures(default_fixture_dir())
        self.assertGreaterEqual(len(fixtures), 9)

    def test_run_review_benchmark_generates_reports_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=self._fake_stage_call):
                report = run_review_benchmark(self._runtime_config(), output_dir)
                json_path, md_path = write_review_benchmark_reports(output_dir, report)

            self.assertEqual(report["speed"]["fixture_count"], len(report["fixtures"]))
            self.assertGreater(report["quality"]["average_fixture_quality_score"], 80.0)
            self.assertIn("usable_capacity", report)
            self.assertIn("transcript_cleanup_review", report["usable_capacity"])
            self.assertEqual(report["quality"]["protected_term_violation_count"], 0)
            self.assertIn("average_patch_compactness", report["quality"])
            self.assertIn("model_verdict", report)
            self.assertIn("derived_scores", report)
            self.assertIn("production_recommendations", report)
            self.assertIn("stage_usefulness", report)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("focus_stage_scores", report["quality"])


if __name__ == "__main__":
    unittest.main()
