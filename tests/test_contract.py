import unittest

from podcast_transcribe.contract import transcript_contract_summary, validate_transcript_payload


class ContractTests(unittest.TestCase):
    def test_contract_summary_lists_required_fields(self):
        summary = transcript_contract_summary()

        self.assertEqual(summary["schema_version"], 2)
        self.assertIn("segments", summary["required_top_level_fields"])
        self.assertIn("transcription_confidence", summary["required_segment_fields"])

    def test_contract_validator_reports_missing_fields(self):
        errors = validate_transcript_payload({"schema_version": 2, "pipeline": "podcast-host-transcription-pipeline"})

        self.assertTrue(any("missing top-level fields" in error for error in errors))
        self.assertIn("missing source_file", errors)


if __name__ == "__main__":
    unittest.main()

