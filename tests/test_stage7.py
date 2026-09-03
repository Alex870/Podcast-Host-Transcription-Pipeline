import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_transcribe.evaluation.pipeline_benchmark import run_pipeline_benchmark, write_pipeline_benchmark_reports
from podcast_transcribe.evaluation.stage7 import (
    condition_report,
    gold_set_readiness,
    provider_promotion_report,
)
from podcast_transcribe.models import SegmentItem
from podcast_transcribe.providers.asr import ParakeetASRProvider
from podcast_transcribe.speaker_workflow import (
    assert_write_revision,
    build_cross_episode_speaker_view,
    file_revision,
)
import podcast_transcribe.speaker_workflow as speaker_workflow
from podcast_transcribe.speakers import (
    approve_speaker_profile_promotion,
    calibrate_speaker_thresholds,
    rollback_speaker_profile_promotion,
    stage_speaker_profile_promotion,
)


class _FakeParakeet:
    def transcribe(self, paths, batch_size=1):
        return [{"segments": [{"text": "hello host", "start": 1.0, "end": 2.5}]}]


class Stage7Tests(unittest.TestCase):
    def test_gold_readiness_and_condition_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference.json").write_text("{}", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "gold_set_version": 2,
                        "entries": [
                            {
                                "id": "episode",
                                "reference": "reference.json",
                                "segment_ids": [1],
                                "tags": ["crosstalk", "short_turn"],
                                "approval_status": "human_approved",
                                "reviewer_id": "reviewer-1",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            readiness = gold_set_readiness(root)
            self.assertEqual(readiness["ready_count"], 1)
            conditions = condition_report(
                {"results": [{"error_taxonomy": ["crosstalk"], "wer": {"errors": 1, "reference_words": 10}, "timestamp_error": {"mean_absolute_error_seconds": 0.2}}]}
            )
            self.assertEqual(conditions["crosstalk"]["wer"], 0.1)

    def test_provider_promotion_report_has_condition_guardrail(self):
        baseline = {"aggregate": {"wer": {"wer": 0.2}, "speaker_attributed_wer": {"speaker_attributed_wer": 0.2}, "mean_timestamp_error_seconds": 0.3}}
        candidate = {"aggregate": {"wer": {"wer": 0.1}, "speaker_attributed_wer": {"speaker_attributed_wer": 0.1}, "mean_timestamp_error_seconds": 0.2}}
        report = provider_promotion_report(baseline, candidate, provider_stage="alignment")
        self.assertTrue(report["passed"])
        self.assertIn("alignment", report["provider_stage"])

    def test_parakeet_adapter_is_lazy_and_normalizes_segments(self):
        provider = ParakeetASRProvider("fake-parakeet", model_loader=lambda: _FakeParakeet())
        result = provider.transcribe("episode.wav", "en", 5, 2, None, None)
        self.assertEqual(provider.identity.provider, "parakeet")
        self.assertEqual(result.value[0].text, "hello host")
        self.assertEqual(result.value[0].start, 1.0)
        self.assertTrue(result.metadata["input_audio_identity"]["missing"])
        self.assertIn("fingerprint", result.metadata["preprocessing"])
        self.assertIn("runtime_seconds", result.metadata["execution"])

    def test_calibration_and_reversible_profile_promotion(self):
        calibration = calibrate_speaker_thresholds(
            [{"similarity": 0.9}, {"similarity": 0.8}],
            [{"similarity": 0.2}, {"similarity": 0.4}],
            [{"similarity": 0.85}],
        )
        self.assertGreaterEqual(calibration["threshold"], 0.4)
        self.assertEqual(calibration["short_turn_pass_rate"], 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "host_profile.json"
            profile_path.write_text(
                json.dumps({"profile_schema_version": 2, "embedding_provider": {"provider": "ecapa", "model": "m"}, "embedding": [1.0, 0.0], "embedding_dimension": 2}),
                encoding="utf-8",
            )
            candidate = {"profile_schema_version": 2, "embedding_provider": {"provider": "ecapa", "model": "m"}, "embedding": [0.0, 1.0], "embedding_dimension": 2}
            report = {"passed": True, "threshold": calibration["threshold"]}
            stage_speaker_profile_promotion(profile_path, candidate, report)
            approve_speaker_profile_promotion(profile_path, "reviewer-1")
            self.assertEqual(json.loads(profile_path.read_text(encoding="utf-8"))["promotion"]["status"], "approved")
            rollback_speaker_profile_promotion(profile_path, "reviewer-1")
            self.assertEqual(json.loads(profile_path.read_text(encoding="utf-8"))["embedding"], [1.0, 0.0])

    def test_cross_episode_speaker_view_and_write_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for episode in ("EpisodeA", "EpisodeB"):
                (output / f"{episode}_cleaned_speaker_transcript.json").write_text(
                    json.dumps(
                        {
                            "segments": [
                                {
                                    "id": 1,
                                    "start": 2,
                                    "end": 3,
                                    "speaker": "SPEAKER_01",
                                    "original_speaker": "SPEAKER_01",
                                    "text": episode,
                                }
                            ],
                            "speaker_identity_evidence": [
                                {
                                    "evidence_id": f"evidence-{episode}",
                                    "episode_id": episode,
                                    "local_speaker": "SPEAKER_01",
                                    "source_audio": f"{episode}.mp3",
                                    "embedding_family": "ecapa:model",
                                    "embedding": [1.0, 0.0] if episode == "EpisodeA" else [0.99, 0.01],
                                    "duration_seconds": 400,
                                    "quality_score": 1.0,
                                    "spans": [{"start": 2, "end": 3}],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            with patch.object(speaker_workflow, "file_revision", wraps=speaker_workflow.file_revision) as revision:
                view = build_cross_episode_speaker_view(output, "speaker")
            self.assertEqual(view["row_count"], 2)
            self.assertEqual(revision.call_count, 2)
            self.assertEqual(view["recurring_unknown_speakers"][0]["episode_count"], 2)
            self.assertEqual(
                view["recurring_unknown_speakers"][0]["evidence_clips"][0]["evidence_id"],
                "evidence-EpisodeA",
            )
            source = output / "EpisodeA_cleaned_speaker_transcript.json"
            revision = file_revision(source)
            assert_write_revision(source, revision)
            source.write_text("changed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                assert_write_revision(source, revision)
