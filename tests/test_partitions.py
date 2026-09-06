import tempfile
import unittest
from pathlib import Path

from podcast_transcribe.partitions import PartitionError, PartitionRegistry, ensure_legacy_partition, resolve_partition_context


class PartitionRegistryTests(unittest.TestCase):
    def test_create_applies_context_defaults_and_creates_managed_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = PartitionRegistry(root)
            record = registry.create("Work Meetings", context_type="meeting")

            self.assertEqual("anonymous_meeting", record.workflow_profile)
            self.assertEqual("work-meetings", record.slug)
            self.assertTrue(record.intake_dir.is_dir())
            self.assertTrue(record.output_dir.is_dir())
            self.assertTrue(record.state_dir.is_dir())
            self.assertIsNone(record.speaker_reference_dir)
            self.assertTrue(record.corrections_dir.is_dir())

    def test_scan_tracks_new_files_and_reuses_completed_status_until_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = PartitionRegistry(root)
            record = registry.create("Podcast")
            audio = record.intake_dir / "episode.mp3"
            audio.write_bytes(b"first")

            first = registry.scan(record.partition_id)
            self.assertEqual(1, first["counts"]["ready"])
            registry.mark_file(record.partition_id, audio, "completed", output_valid=True)

            second = registry.scan(record.partition_id)
            self.assertEqual(1, second["counts"]["completed"])

            audio.write_bytes(b"changed")
            third = registry.scan(record.partition_id)
            self.assertEqual(1, third["counts"]["ready"])

    def test_overlapping_active_intake_folders_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = PartitionRegistry(root)
            shared = root / "shared-intake"
            registry.create("First", intake_dir=shared)
            with self.assertRaises(PartitionError):
                registry.create("Second", intake_dir=shared / "nested")

    def test_context_merges_partition_paths_and_metadata_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = PartitionRegistry(root)
            record = registry.create(
                "Research Interviews",
                context_type="custom",
                workflow_profile="podcast",
                config_overrides={"review_reasoning_effort": "none", "custom_flag": True},
            )
            context = resolve_partition_context(root, record.partition_id, {"hf_token": "secret", "backend": "vllm"})

            self.assertEqual(str(record.intake_dir), context.effective_config["default_source_dir"])
            self.assertEqual("none", context.effective_config["review_reasoning_effort"])
            self.assertEqual(record.partition_id, context.metadata()["partition_id"])
            self.assertEqual("secret", context.effective_config["hf_token"])
            self.assertNotIn("hf_token", context.metadata())

    def test_registry_backup_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = PartitionRegistry(root)
            registry.create("Podcast")
            backup = registry.backup()
            self.assertTrue(backup.exists())
            self.assertNotEqual(registry.path, backup)

    def test_partition_rejects_concurrent_runs_until_active_run_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = PartitionRegistry(Path(tmp))
            record = registry.create("Podcast")
            run_id = registry.start_run(record.partition_id)

            with self.assertRaisesRegex(PartitionError, "already has an active run"):
                registry.start_run(record.partition_id)

            registry.finish_run(run_id)
            self.assertTrue(registry.start_run(record.partition_id))

    def test_archived_space_can_still_be_inspected(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = PartitionRegistry(Path(tmp))
            record = registry.create("Archived")
            registry.update(record.partition_id, archived=True)

            summary = registry.summary(record.partition_id, include_archived=True)

            self.assertEqual(record.partition_id, summary["partition"]["partition_id"])

    def test_partition_settings_reject_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = PartitionRegistry(Path(tmp))
            with self.assertRaises(PartitionError):
                registry.create("Unsafe", config_overrides={"hf_token": "do-not-store"})
            with self.assertRaises(PartitionError):
                registry.create("Unsafe paths", config_overrides={"output_dir": "C:/elsewhere"})

    def test_legacy_adoption_points_at_existing_configured_folders_without_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "incoming"
            source.mkdir()
            config = root / "podcast_transcribe_config.json"
            config.write_text('{"default_source_dir":"incoming"}', encoding="utf-8")

            first = ensure_legacy_partition(root)
            second = ensure_legacy_partition(root)

            self.assertEqual(first.partition_id, second.partition_id)
            self.assertEqual(source.resolve(), first.intake_dir)
            self.assertEqual((root / "output").resolve(), first.output_dir)


if __name__ == "__main__":
    unittest.main()
