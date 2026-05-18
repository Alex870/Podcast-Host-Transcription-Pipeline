import json
import tempfile
import unittest
from pathlib import Path

from podcast_transcribe.state import (
    audio_file_fingerprint,
    clear_stage_artifacts,
    expected_output_paths,
    is_file_already_processed,
    load_processed_files,
    load_stage_artifact,
    save_stage_artifact,
    save_processed_files,
)


TEST_TMP = Path(__file__).resolve().parents[1] / "test_tmp"


class ResumeStateTests(unittest.TestCase):
    def test_processed_file_requires_outputs_and_matching_fingerprint(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp)
            audio = root / "episode.mp3"
            output_dir = root / "output"
            output_dir.mkdir()
            audio.write_bytes(b"first")
            record = audio_file_fingerprint(audio)
            processed = {audio.name: record}
            summary_rows = {audio.name: {"episode": audio.name}}

            self.assertFalse(is_file_already_processed(audio, output_dir, processed, summary_rows))

            for output_path in expected_output_paths(audio, output_dir):
                output_path.write_text("ok", encoding="utf-8")
            self.assertTrue(is_file_already_processed(audio, output_dir, processed, summary_rows))

            audio.write_bytes(b"changed")
            self.assertFalse(is_file_already_processed(audio, output_dir, processed, summary_rows))

    def test_legacy_processed_list_is_loaded_but_still_requires_summary(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            state_path = Path(tmp) / "_processed_files.json"
            state_path.write_text(json.dumps({"processed_files": ["episode.mp3"]}), encoding="utf-8")
            self.assertEqual(load_processed_files(state_path), {"episode.mp3": {}})

    def test_processed_state_round_trips_fingerprints(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            state_path = Path(tmp) / "_processed_files.json"
            records = {"b.mp3": {"size_bytes": 2, "mtime_ns": 20}, "a.mp3": {"size_bytes": 1, "mtime_ns": 10}}
            save_processed_files(state_path, records)
            self.assertEqual(load_processed_files(state_path), records)

    def test_stage_artifact_round_trip_and_fingerprint_validation(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp)
            audio = root / "episode.mp3"
            audio.write_bytes(b"abc")

            save_stage_artifact(root, audio, "transcription", {"value": 1})
            self.assertEqual(load_stage_artifact(root, audio, "transcription"), {"value": 1})

            audio.write_bytes(b"changed")
            self.assertIsNone(load_stage_artifact(root, audio, "transcription"))

    def test_clear_stage_artifacts_removes_episode_artifact_dir(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp)
            audio = root / "episode.mp3"
            audio.write_bytes(b"abc")

            save_stage_artifact(root, audio, "diarization", {"turns": []})
            clear_stage_artifacts(root, audio)

            self.assertIsNone(load_stage_artifact(root, audio, "diarization"))


if __name__ == "__main__":
    unittest.main()

