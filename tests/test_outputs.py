import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from podcast_transcribe_outputs import write_text_transcript


TEST_TMP = Path(__file__).resolve().parents[1] / "test_tmp"


class OutputTests(unittest.TestCase):
    def test_host_only_transcript_accepts_named_host_label(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
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


if __name__ == "__main__":
    unittest.main()
