import json, tempfile, unittest
from pathlib import Path
from podcast_transcribe.hardening import create_backup, inspect_backup, restore_backup
from podcast_transcribe.m6_preflight import build_preflight


class M6HardeningTests(unittest.TestCase):
    def test_preflight_is_redacted_and_never_probes_network(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as raw:
            value = build_preflight(
                "tx",
                Path(raw),
                required_modules=("m6_missing_dependency",),
                minimum_free_bytes=1,
            )
            self.assertTrue(value["workspace"]["writable"])
            self.assertFalse(value["offline"]["network_probe_performed"])
            self.assertNotIn(str(Path(raw).resolve()), json.dumps(value))
            self.assertTrue(value["redaction"]["credentials_omitted"])
            self.assertGreater(value["profile"]["total_memory_bytes"], 0)
            missing = next(
                item
                for item in value["capabilities"]
                if item["capability"] == "m6_missing_dependency"
            )
            self.assertEqual(
                "python -m pip install -r podcast_transcribe_requirements.txt",
                missing["remediation_command"],
            )

    def test_review_state_backup_is_checksum_bound_and_restore_requires_approval(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as raw:
            root = Path(raw)
            project = root / "project"
            output = root / "output"
            project.mkdir()
            output.mkdir()
            (project / "config.json").write_text('{"model":"pinned"}')
            (output / "resume_state.json").write_text('{"done":true}')
            (output / "audio.wav").write_bytes(b"private-media")
            backup = create_backup(project, output, root / "state.zip")
            self.assertTrue(inspect_backup(root / "state.zip")["valid"])
            self.assertNotIn("audio.wav", [item["path"] for item in backup["entries"]])
            (project / "config.json").write_text("changed")
            with self.assertRaises(PermissionError):
                restore_backup(
                    root / "state.zip", project, output, approved_backup_id="wrong"
                )
            restored = restore_backup(
                root / "state.zip",
                project,
                output,
                approved_backup_id=backup["backup_id"],
            )
            self.assertGreater(restored["restored_count"], 0)
            self.assertEqual(
                '{"model":"pinned"}', (project / "config.json").read_text()
            )


if __name__ == "__main__":
    unittest.main()
