import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from podcast_transcribe.outputs import (
    REVIEWED_TRANSCRIPT_SCHEMA_VERSION,
    TRANSCRIPT_SCHEMA_VERSION,
    build_episode_metadata,
    segment_confidence,
    validate_reviewed_transcript_payload,
    validate_transcript_payload,
    write_batch_report_md,
    write_json_output,
    write_output_manifest,
    write_speaker_identity_review_csv,
    write_text_transcript,
)

class OutputTests(unittest.TestCase):
    def test_episode_metadata_uses_last_valid_yyyymmdd_in_filename(self):
        metadata = build_episode_metadata(r"D:\Podcasts\TFM 20250101 final 20260204.mp3")

        self.assertEqual(metadata["source_filename"], "TFM 20250101 final 20260204.mp3")
        self.assertEqual(metadata["episode_date"], "2026-02-04")
        self.assertEqual(metadata["episode_date_compact"], "20260204")
        self.assertEqual(metadata["episode_year"], 2026)
        self.assertEqual(metadata["episode_month"], 2)
        self.assertEqual(metadata["episode_day"], 4)
        self.assertEqual(metadata["episode_sort_key"], 20260204)

    def test_episode_metadata_supports_configured_dashed_iso_format(self):
        metadata = build_episode_metadata(
            "TFM 2026-02-04.mp3",
            {"formats": ["YYYY-MM-DD"]},
        )

        self.assertEqual(metadata["episode_date"], "2026-02-04")
        self.assertEqual(metadata["episode_date_compact"], "20260204")
        self.assertEqual(metadata["episode_sort_key"], 20260204)

    def test_episode_metadata_respects_format_priority_for_ambiguous_dates(self):
        metadata = build_episode_metadata(
            "Episode 03-04-2026.mp3",
            {"formats": ["DD-MM-YYYY", "MM-DD-YYYY"]},
        )

        self.assertEqual(metadata["episode_date"], "2026-04-03")
        self.assertEqual(metadata["episode_date_compact"], "20260403")

    def test_episode_metadata_can_use_first_match_when_configured(self):
        metadata = build_episode_metadata(
            "TFM 20260101 final 20260204.mp3",
            {"formats": ["YYYYMMDD"], "position": "first"},
        )

        self.assertEqual(metadata["episode_date"], "2026-01-01")
        self.assertEqual(metadata["episode_sort_key"], 20260101)

    def test_episode_metadata_can_be_disabled(self):
        metadata = build_episode_metadata(
            "TFM 20260204.mp3",
            {"enabled": False},
        )

        self.assertEqual(metadata["episode_date"], "")
        self.assertEqual(metadata["episode_date_compact"], "")
        self.assertEqual(metadata["episode_sort_key"], "")

    def test_episode_metadata_ignores_invalid_dates(self):
        metadata = build_episode_metadata("TFM 20261340.mp3")

        self.assertEqual(metadata["episode_date"], "")
        self.assertEqual(metadata["episode_date_compact"], "")
        self.assertEqual(metadata["episode_sort_key"], "")

    def test_host_only_transcript_accepts_named_host_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "host_only.txt"
            segments = [
                SimpleNamespace(start=0, speaker="Alex", text="host line"),
                SimpleNamespace(start=1, speaker="SPEAKER_01", text="guest line"),
            ]

            write_text_transcript(
                output_path,
                segments,
                lambda seconds: f"{int(seconds):02d}",
                host_only=True,
                host_labels={"HOST", "Alex"},
            )

            self.assertEqual(output_path.read_text(encoding="utf-8"), "[00][Alex] host line")

    def test_json_output_accepts_dict_words_for_review_backfill_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "speaker_transcript.json"
            metadata = build_episode_metadata("TFM 20260204.mp3")
            segment = SimpleNamespace(
                id=1,
                start=0.0,
                end=1.0,
                speaker="HOST",
                text="hello",
                avg_logprob=-0.1,
                no_speech_prob=0.01,
                words=[{"start": 0.0, "end": 0.5, "word": "hello", "speaker": "HOST"}],
            )

            write_json_output(
                output_path,
                source_file="TFM 20260204.mp3",
                info_payload={"duration": 1.0},
                diarized_turns=[],
                segments=[segment],
                speaker_mapping={"SPEAKER_00": "HOST"},
                host_speaker="SPEAKER_00",
                durations={"SPEAKER_00": 1.0},
                known_assignments={},
                metadata=metadata,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["segments"][0]["words"][0]["word"], "hello")

    def test_text_transcript_writes_episode_metadata_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "speaker_transcript.txt"
            segments = [SimpleNamespace(start=0, speaker="HOST", text="opening line")]
            metadata = build_episode_metadata("TFM 20260204.mp3")

            write_text_transcript(
                output_path,
                segments,
                lambda seconds: f"{int(seconds):02d}",
                metadata=metadata,
            )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "\n".join(
                    [
                        "# source_file: TFM 20260204.mp3",
                        "# source_filename: TFM 20260204.mp3",
                        "# episode_date: 2026-02-04",
                        "# episode_date_compact: 20260204",
                        "# episode_sort_key: 20260204",
                        "",
                        "[00][HOST] opening line",
                    ]
                ),
            )

    def test_json_output_repeats_episode_metadata_on_segments_for_vector_ingest(self):
        @dataclass
        class Word:
            start: float
            end: float
            word: str
            speaker: str

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "speaker_transcript.json"
            segment = SimpleNamespace(
                id=1,
                start=12.0,
                end=14.0,
                speaker="HOST",
                text="host viewpoint",
                avg_logprob=-0.1,
                no_speech_prob=0.01,
                words=[Word(12.0, 12.5, "host", "HOST")],
            )

            write_json_output(
                output_path,
                source_file="TFM 20260204.mp3",
                info_payload={"duration": 120.0},
                diarized_turns=[],
                segments=[segment],
                speaker_mapping={"SPEAKER_00": "HOST"},
                host_speaker="SPEAKER_00",
                durations={"SPEAKER_00": 60.0},
                known_assignments={},
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], TRANSCRIPT_SCHEMA_VERSION)
            self.assertEqual(payload["pipeline"], "podcast-host-transcription-pipeline")
            self.assertEqual(payload["metadata"]["episode_date"], "2026-02-04")
            self.assertEqual(payload["episode_date"], "2026-02-04")
            self.assertEqual(payload["segments"][0]["episode_date"], "2026-02-04")
            self.assertEqual(payload["segments"][0]["episode_sort_key"], 20260204)
            self.assertEqual(payload["segments"][0]["transcription_confidence"]["quality"], "high")
            self.assertIn("content_quality", payload["segments"][0])
            self.assertEqual(payload["content_quality_summary"]["segment_count"], 1)
            self.assertEqual(validate_transcript_payload(payload), [])

    def test_unknown_segment_satisfies_json_contract(self):
        @dataclass
        class Word:
            start: float
            end: float
            word: str
            speaker: str = "UNKNOWN"

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "speaker_transcript.json"
            segment = SimpleNamespace(
                id=17,
                start=12.0,
                end=14.0,
                speaker="UNKNOWN",
                text="Gap between diarization turns.",
                avg_logprob=-0.1,
                no_speech_prob=0.01,
                words=[Word(12.0, 12.5, "Gap")],
            )

            write_json_output(
                output_path,
                source_file="TFM 20260204.mp3",
                info_payload={"duration": 120.0},
                diarized_turns=[],
                segments=[segment],
                speaker_mapping={},
                host_speaker=None,
                durations={},
                known_assignments={},
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["segments"][0]["speaker"], "UNKNOWN")
            self.assertEqual(payload["segments"][0]["words"][0]["speaker"], "UNKNOWN")
            self.assertEqual(validate_transcript_payload(payload), [])

    def test_cleaned_json_preserves_original_segment_text(self):
        @dataclass
        class Word:
            start: float
            end: float
            word: str
            speaker: str

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "cleaned_speaker_transcript.json"
            segment = SimpleNamespace(
                id=1,
                start=12.0,
                end=14.0,
                speaker="HOST",
                text="Otherwise, this changed.",
                original_text="Otherwise, otherwise, this changed.",
                cleanup_applied=True,
                avg_logprob=-0.1,
                no_speech_prob=0.01,
                words=[Word(12.0, 12.5, "Otherwise", "HOST")],
            )

            write_json_output(
                output_path,
                source_file="TFM 20260204.mp3",
                info_payload={"duration": 120.0},
                diarized_turns=[],
                segments=[segment],
                speaker_mapping={"SPEAKER_00": "HOST"},
                host_speaker="SPEAKER_00",
                durations={"SPEAKER_00": 60.0},
                known_assignments={},
                text_version="cleaned",
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["text_version"], "cleaned")
            self.assertEqual(payload["segments"][0]["text"], "Otherwise, this changed.")
            self.assertEqual(payload["segments"][0]["original_text"], "Otherwise, otherwise, this changed.")
            self.assertTrue(payload["segments"][0]["cleanup_applied"])

    def test_reviewed_json_emits_additive_review_metadata(self):
        @dataclass
        class Word:
            start: float
            end: float
            word: str
            speaker: str

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "reviewed_speaker_transcript.json"
            segment = SimpleNamespace(
                id=1,
                start=12.0,
                end=14.0,
                speaker="HOST",
                text="Reviewed wording.",
                original_text="Original wording.",
                cleanup_applied=False,
                cleanup_level="",
                manual_correction_applied=False,
                original_speaker=None,
                llm_reviewed_text="Reviewed wording.",
                review_runtime_profile="high_context_5090",
                review_backend="vllm",
                review_model_name="qwen-review",
                review_stage_flags={
                    "transcript_cleanup_review": True,
                    "glossary_correction_review": True,
                    "speaker_consistency_review": True,
                    "episode_qa_review": True,
                },
                avg_logprob=-0.1,
                no_speech_prob=0.01,
                words=[Word(12.0, 12.5, "Reviewed", "HOST")],
            )

            write_json_output(
                output_path,
                source_file="TFM 20260204.mp3",
                info_payload={"duration": 120.0},
                diarized_turns=[],
                segments=[segment],
                speaker_mapping={"SPEAKER_00": "HOST"},
                host_speaker="SPEAKER_00",
                durations={"SPEAKER_00": 60.0},
                known_assignments={},
                text_version="reviewed_llm_high_context",
                review_metadata={
                    "review_pipeline_version": 2,
                    "review_runtime_profile": "high_context_5090",
                    "review_backend": "vllm",
                    "review_model_name": "qwen-review",
                    "review_stage_flags": {
                        "transcript_cleanup_review": True,
                        "glossary_correction_review": True,
                        "speaker_consistency_review": True,
                        "episode_qa_review": True,
                    },
                    "review_status": "completed",
                    "review_skip_reason": "",
                    "reviewed_segment_count": 1,
                    "corrected_segment_count": 1,
                    "episode_notes": ["One correction applied."],
                    "review_stage_results": {
                        "transcript_cleanup_review": {
                            "attempted": True,
                            "status": "completed",
                            "skip_reason": "",
                            "corrected_segment_count": 1,
                            "edit_scope": "text_only",
                        }
                    },
                    "review_enabled_stages": ["transcript_cleanup_review"],
                    "review_completed_stages": ["transcript_cleanup_review"],
                    "review_skipped_stages": [],
                    "episode_qa_mode": "disabled",
                    "review_input_source": "inline_cleaned_segments",
                },
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["review_schema_version"], REVIEWED_TRANSCRIPT_SCHEMA_VERSION)
            self.assertEqual(payload["review_metadata"]["review_backend"], "vllm")
            self.assertEqual(payload["segments"][0]["original_text"], "Original wording.")
            self.assertEqual(payload["segments"][0]["llm_reviewed_text"], "Reviewed wording.")
            self.assertEqual(validate_reviewed_transcript_payload(payload), [])

    def test_segment_confidence_flags_low_quality(self):
        confidence = segment_confidence(avg_logprob=-1.2, no_speech_prob=0.7)

        self.assertEqual(confidence["quality"], "low")
        self.assertIn("low_avg_logprob", confidence["warnings"])
        self.assertIn("high_no_speech_prob", confidence["warnings"])

    def test_speaker_identity_review_csv_highlights_uncertain_speakers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "speaker_identity_review.csv"

            write_speaker_identity_review_csv(
                output_path,
                speaker_mapping={"SPEAKER_00": "HOST", "SPEAKER_01": "Guest"},
                durations={"SPEAKER_00": 120.0, "SPEAKER_01": 8.0},
                similarity_scores={"SPEAKER_00": 0.62, "SPEAKER_01": 0.6},
                known_assignments={},
                host_speaker="SPEAKER_00",
            )

            text = output_path.read_text(encoding="utf-8")
            self.assertIn("review_recommended", text)
            self.assertIn("short speaker duration", text)

    def test_output_manifest_records_outputs_and_config_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_file = tmp_path / "episode_speaker_transcript.json"
            output_file.write_text("{}", encoding="utf-8")
            manifest_path = tmp_path / "episode_manifest.json"

            write_output_manifest(
                manifest_path,
                source_file="episode.mp3",
                source_fingerprint={"size_bytes": 12, "mtime_ns": 34},
                config={"model": "large-v3", "cleanup_level": "normal"},
                outputs=[output_file],
                timings={"total": 1.2},
                summary={"episode": "episode.mp3", "review_priority_score": 0},
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest_version"], 1)
            self.assertEqual(payload["outputs"][0]["filename"], "episode_speaker_transcript.json")
            self.assertTrue(payload["config_fingerprint"])

    def test_batch_report_summarizes_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "_batch_report.md"

            write_batch_report_md(
                output_path,
                [
                    {
                        "episode": "episode.mp3",
                        "episode_date": "2026-01-01",
                        "host_detected": True,
                        "transcript_segments": 10,
                        "review_row_count": 2,
                        "review_priority_score": 12.5,
                        "review_priority_reason": "needs review",
                        "review_attempted": True,
                        "reviewed_output_written": True,
                        "review_corrected_segment_count": 1,
                        "cleanup_review_corrected_count": 1,
                        "glossary_review_corrected_count": 0,
                        "speaker_consistency_review_corrected_count": 0,
                        "episode_qa_review_corrected_count": 0,
                    }
                ],
                elapsed_seconds=90,
            )

            text = output_path.read_text(encoding="utf-8")
            self.assertIn("# Podcast Transcription Batch Report", text)
            self.assertIn("Episodes: 1", text)
            self.assertIn("episode.mp3", text)
            self.assertIn("Review attempted: 1/1", text)
            self.assertIn("Cleanup-review corrections: 1", text)


if __name__ == "__main__":
    unittest.main()
