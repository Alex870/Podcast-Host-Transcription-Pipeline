import json
import tempfile
import unittest
from pathlib import Path

from podcast_transcribe.evaluation.metrics import diarization_error_rate, speaker_attributed_wer, word_error_rate
from podcast_transcribe.evaluation.pipeline_benchmark import run_pipeline_benchmark, write_pipeline_benchmark_reports
from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.providers.alignment import TimestampPassthroughAlignmentProvider


def transcript_payload(text: str, speaker: str = "HOST"):
    return {
        "schema_version": 2,
        "pipeline": "podcast-host-transcription-pipeline",
        "pipeline_version": "test",
        "source_file": "Episode.mp3",
        "metadata": {"episode_date": "2026-07-01"},
        "episode_date": "2026-07-01",
        "episode_date_compact": "20260701",
        "episode_sort_key": 20260701,
        "text_version": "cleaned",
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 2.0,
                "speaker": speaker,
                "text": text,
                "episode_date": "2026-07-01",
                "episode_sort_key": 20260701,
                "transcription_confidence": {},
                "words": [],
            }
        ],
    }


class PipelineEvaluationTests(unittest.TestCase):
    def test_text_and_speaker_metrics_distinguish_identity_errors(self):
        self.assertEqual(word_error_rate("hello world", "hello world")["wer"], 0.0)
        reference = transcript_payload("hello world")["segments"]
        wrong_speaker = transcript_payload("hello world", speaker="GUEST")["segments"]
        self.assertGreater(speaker_attributed_wer(reference, wrong_speaker)["speaker_attributed_wer"], 0.0)

    def test_pipeline_benchmark_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold"
            candidates = root / "candidates"
            reports = root / "reports"
            gold.mkdir()
            candidates.mkdir()
            (gold / "reference.json").write_text(json.dumps(transcript_payload("hello ChromaDB")), encoding="utf-8")
            (candidates / "Episode_cleaned_speaker_transcript.json").write_text(
                json.dumps(transcript_payload("hello ChromaDB")), encoding="utf-8"
            )
            (gold / "manifest.json").write_text(
                json.dumps(
                    {
                        "gold_set_version": 1,
                        "name": "test",
                        "entries": [
                            {
                                "id": "Episode",
                                "audio_stem": "Episode",
                                "reference": "reference.json",
                                "preferred_terms": ["ChromaDB"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = run_pipeline_benchmark(gold, candidates)
            json_path, markdown_path = write_pipeline_benchmark_reports(reports, report)
            self.assertEqual(report["aggregate"]["wer"]["wer"], 0.0)
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())

    def test_passthrough_alignment_preserves_segments(self):
        segment = SegmentItem(
            id=1,
            start=0.0,
            end=1.0,
            text="hello",
            speaker=None,
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            words=[WordItem(start=0.0, end=0.8, word="hello", speaker=None)],
        )
        result = TimestampPassthroughAlignmentProvider().align("unused.wav", [segment], "en")
        self.assertEqual(result.value[0].words[0].word, "hello")
        self.assertEqual(result.provider.provider, "timestamp_passthrough")

    def test_der_maps_anonymous_speaker_labels(self):
        reference = [{"start": 0, "end": 2, "speaker": "HOST"}, {"start": 2, "end": 4, "speaker": "GUEST"}]
        hypothesis = [{"start": 0, "end": 2, "speaker": "SPEAKER_01"}, {"start": 2, "end": 4, "speaker": "SPEAKER_00"}]
        self.assertEqual(diarization_error_rate(reference, hypothesis)["diarization_error_rate"], 0.0)

    def test_pipeline_benchmark_compares_baseline_and_applies_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold, candidate, baseline = root / "gold", root / "candidate", root / "baseline"
            for path in (gold, candidate, baseline):
                path.mkdir()
            (gold / "reference.json").write_text(json.dumps(transcript_payload("hello world")), encoding="utf-8")
            (candidate / "Episode_cleaned_speaker_transcript.json").write_text(json.dumps(transcript_payload("hello world")), encoding="utf-8")
            (baseline / "Episode_cleaned_speaker_transcript.json").write_text(json.dumps(transcript_payload("hello there")), encoding="utf-8")
            (gold / "manifest.json").write_text(json.dumps({
                "gold_set_version": 1,
                "entries": [{"id": "Episode", "audio_stem": "Episode", "reference": "reference.json", "error_taxonomy": ["conversation"]}],
                "promotion_thresholds": {"max_wer_regression": 0.0},
            }), encoding="utf-8")
            report = run_pipeline_benchmark(gold, candidate, baseline)
            self.assertTrue(report["promotion"]["passed"])
            self.assertLess(report["comparison"]["wer_delta"], 0.0)
            self.assertIn("conversation", report["error_taxonomy"])


if __name__ == "__main__":
    unittest.main()
