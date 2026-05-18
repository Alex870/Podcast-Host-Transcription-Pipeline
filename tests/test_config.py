import tempfile
import unittest
from pathlib import Path

from podcast_transcribe.config import load_replacement_map


TEST_TMP = Path(__file__).resolve().parents[1] / "test_tmp"


class ConfigTests(unittest.TestCase):
    def test_replacement_map_reports_json_line_context(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            map_path = Path(tmp) / "preferred_replacements.json"
            map_path.write_text('{\n  "Federal Reserve": ["fed"],\n  "Bad": \n}', encoding="utf-8")

            with self.assertRaises(RuntimeError) as context:
                load_replacement_map(str(map_path))

            message = str(context.exception)
            self.assertIn("Invalid JSON in replacement map file", message)
            self.assertIn("line 4, column 1", message)
            self.assertIn("Replacement maps must be strict JSON", message)

    def test_replacement_map_normalizes_alias_lists(self):
        TEST_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            map_path = Path(tmp) / "preferred_replacements.json"
            map_path.write_text(
                '{"Federal Reserve": ["fed", "", 123], "Ignored": "not a list"}',
                encoding="utf-8",
            )

            self.assertEqual(load_replacement_map(str(map_path)), {"Federal Reserve": ["fed"]})


if __name__ == "__main__":
    unittest.main()

