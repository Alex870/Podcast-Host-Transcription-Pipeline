import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from podcast_transcribe.contract import validate_transcript_payload
from podcast_transcribe.outputs import (
    build_episode_metadata,
    write_batch_report_md,
    write_json_output,
    write_output_manifest,
    write_review_csv,
    write_speaker_identity_review_csv,
    write_text_transcript,
)


TEST_TMP = Path(__file__).resolve().parents[1] / "test_tmp"


@dataclass
class Word:
    start: float
    end: float
    word: str
    speaker: str


class IntegrationOutputTests(unittest.TestCase):
    def test_tiny_episode_output_set_matches_contract(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp)
            source_file = "Tiny 20260512.mp3"
            metadata = build_episode_metadata(source_file)
            segment = SimpleNamespace(
                id=0,
                start=1.0,
                end=3.0,
                speaker="HOST",
                text="Welcome everyone, support the show.",
                avg_logprob=-0.2,
                no_speech_prob=0.02,
                words=[Word(1.0, 1.5, "Welcome", "HOST")],
            )
            outputs = []

            text_path = root / "Tiny 20260512_speaker_transcript.txt"
            json_path = root / "Tiny 20260512_speaker_transcript.json"
            review_path = root / "Tiny 20260512_review.csv"
            speaker_review_path = root / "Tiny 20260512_speaker_identity_review.csv"
            manifest_path = root / "Tiny 20260512_manifest.json"
            report_path = root / "_batch_report.md"

            write_text_transcript(text_path, [segment], lambda seconds: f"{seconds:.0f}", metadata=metadata)
            write_review_csv(review_path, [])
            write_speaker_identity_review_csv(
                speaker_review_path,
                speaker_mapping={"SPEAKER_00": "HOST"},
                durations={"SPEAKER_00": 2.0},
                similarity_scores={"SPEAKER_00": 0.9},
                known_assignments={},
                host_speaker="SPEAKER_00",
            )
            write_json_output(
                json_path,
                source_file=source_file,
                info_payload={"language": "en", "duration": 4.0},
                diarized_turns=[{"start": 1.0, "end": 3.0, "speaker": "SPEAKER_00"}],
                segments=[segment],
                speaker_mapping={"SPEAKER_00": "HOST"},
                host_speaker="SPEAKER_00",
                durations={"SPEAKER_00": 2.0},
                known_assignments={},
                metadata=metadata,
            )
            outputs.extend([text_path, json_path, review_path, speaker_review_path])
            write_output_manifest(
                manifest_path,
                source_file=source_file,
                source_fingerprint={"size_bytes": 123, "mtime_ns": 456},
                config={"model": "test"},
                outputs=outputs,
                timings={"total": 1.0},
                summary={"episode": source_file, "review_priority_score": 0},
            )
            write_batch_report_md(
                report_path,
                [{"episode": source_file, "host_detected": True, "transcript_segments": 1, "review_row_count": 0}],
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_transcript_payload(payload), [])
            self.assertIn("possible_sponsor_block", payload["segments"][0]["content_quality"]["tags"])
            self.assertTrue(manifest_path.exists())
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()

