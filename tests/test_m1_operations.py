import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_transcribe.operations import apply_retention, campaign_preflight, retry_downstream_delivery
from podcast_transcribe.workbench_core import _write_authoritative_pair_atomic


class MilestoneOneOperationsTests(unittest.TestCase):
    def test_campaign_preflight_orders_risk_and_reports_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            pending = output / "_downstream_corrections"
            pending.mkdir()
            (pending / "c.json").write_text(json.dumps({"correction_set_id": "c", "status": "downstream_pending"}), encoding="utf-8")
            episodes = [
                {"episode_id": "safe", "human_approved": True, "duration_seconds": 60},
                {"episode_id": "risky", "human_approved": False, "duration_seconds": 4000, "issue_count": 2},
            ]
            with patch("podcast_transcribe.workbench_core.discover_episode_bundles", return_value=episodes), patch(
                "podcast_transcribe.workbench_core.evaluation_queues", return_value={"pending": ["risky"]}
            ):
                result = campaign_preflight(root, output)
            self.assertEqual("risky", result["risk_order"][0]["episode_id"])
            self.assertEqual(1, result["downstream"]["pending_count"])
            self.assertTrue(result["preflight_id"].startswith("transcription_preflight_"))

    def test_authoritative_pair_rolls_back_if_second_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "transcript.json", root / "manifest.json"
            first.write_text('{"old":1}', encoding="utf-8")
            second.write_text('{"old":2}', encoding="utf-8")
            real_replace = __import__("os").replace
            calls = 0

            def fail_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated failure")
                return real_replace(source, target)

            with patch("podcast_transcribe.workbench_core.os.replace", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "simulated"):
                    _write_authoritative_pair_atomic(first, {"new": 1}, second, {"new": 2})
            self.assertEqual({"old": 1}, json.loads(first.read_text(encoding="utf-8")))
            self.assertEqual({"old": 2}, json.loads(second.read_text(encoding="utf-8")))

    def test_retry_delivery_and_retention_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            consumer = root / "consumer"
            consumer.mkdir()
            (root / "podcast_transcribe_config.json").write_text(json.dumps({"podcast_rag_project_dir": str(consumer)}), encoding="utf-8")
            pending = output / "_downstream_corrections" / "c.json"
            pending.parent.mkdir()
            pending.write_text(json.dumps({"correction_set_id": "c", "episode_id": "e"}), encoding="utf-8")
            delivered = retry_downstream_delivery(root, output, "c")
            self.assertEqual("downstream_pending", delivered["status"])
            self.assertTrue((consumer / "state/transcription_corrections/c.json").exists())

            logs = output / "logs"
            logs.mkdir()
            old = logs / "old.log"
            old.write_text("x", encoding="utf-8")
            preview = apply_retention(output, {"categories": ["logs"], "older_than_days": 0}, dry_run=True)
            self.assertEqual(1, len(preview["candidates"]))
            self.assertTrue(old.exists())
            apply_retention(output, {"categories": ["logs"], "older_than_days": 0}, dry_run=False)
            self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
