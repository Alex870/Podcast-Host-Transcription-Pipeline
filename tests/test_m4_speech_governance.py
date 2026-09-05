import tempfile
import unittest
from pathlib import Path

from podcast_transcribe.providers.contracts import ProviderIdentity
from podcast_transcribe.providers.governance import (
    acquire_provider_artifact,
    build_speech_provider_run,
    provider_preflight,
    resolve_execution_profile,
    write_immutable_speech_run,
)
from podcast_transcribe.speakers import build_speaker_calibration_set


class MilestoneFourSpeechGovernanceTests(unittest.TestCase):
    def identity(self):
        return ProviderIdentity("transcription", "parakeet", "nvidia/parakeet", model_revision="abc1234")

    def test_preflight_download_revision_and_interrupted_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(provider_preflight(root, self.identity())["available"])

            def downloader(**kwargs):
                path = Path(kwargs["local_dir"])
                path.mkdir(parents=True)
                (path / "weights.bin").write_bytes(b"fixture")
                return str(path)

            acquired = acquire_provider_artifact(root, self.identity(), downloader=downloader)
            self.assertTrue(acquired["available"])
            mismatch = ProviderIdentity("transcription", "parakeet", "nvidia/parakeet", model_revision="def5678")
            self.assertFalse(provider_preflight(root, mismatch)["available"])

    def test_failed_explicit_acquisition_is_visible_as_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def failing_downloader(**kwargs):
                Path(kwargs["local_dir"]).mkdir(parents=True)
                raise OSError("offline")

            with self.assertRaisesRegex(OSError, "offline"):
                acquire_provider_artifact(root, self.identity(), downloader=failing_downloader)
            status = provider_preflight(root, self.identity())
            self.assertTrue(status["interrupted"])
            self.assertFalse(status["available"])

    def test_device_fallback_and_adaptive_cpu_batch(self):
        profile = resolve_execution_profile("cuda", 0, cuda_available=False, cpu_count=4)
        self.assertEqual(profile.resolved_device, "cpu")
        self.assertEqual(profile.batch_size, 1)
        self.assertIn("unavailable", profile.fallback_reason)

    def test_unsupported_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported device override"):
            resolve_execution_profile("directml", 0, cuda_available=False)

    def test_explicit_acquisition_forwards_authentication(self):
        received = {}

        def downloader(**kwargs):
            received.update(kwargs)
            path = Path(kwargs["local_dir"])
            path.mkdir(parents=True)
            return str(path)

        with tempfile.TemporaryDirectory() as tmp:
            acquire_provider_artifact(Path(tmp), self.identity(), token="fixture-token", downloader=downloader)
        self.assertEqual(received["token"], "fixture-token")
        self.assertEqual(received["revision"], "abc1234")

    def test_shadow_run_is_immutable_and_requires_exact_identity(self):
        payload = build_speech_provider_run(
            run_id="run-1",
            evaluation_pack={"pack_id": "pack-1", "source_identity": {"sha256": "0" * 64}},
            audio_identity={"sha256": "1" * 64},
            preprocessing={"sample_rate": 16000, "channels": 1},
            providers=[self.identity()],
            execution={"device": "cpu", "precision": "float32", "batch_size": 1},
            outputs={"raw": "raw.json", "episode_contract": "episode.json"},
            metrics={"wer": 0.1},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(write_immutable_speech_run(root, payload).exists())
            with self.assertRaises(FileExistsError):
                write_immutable_speech_run(root, payload)

    def test_calibration_set_requires_same_and_hard_negative_same_family(self):
        provider = {"provider": "speechbrain_ecapa", "model": "ecapa", "model_revision": "rev1"}
        value = build_speaker_calibration_set(
            calibration_id="cal-1",
            embedding_provider=provider,
            reviewer_id="reviewer-1",
            excerpts=[
                {"excerpt_id": "a", "source_audio_sha256": "a" * 64, "relation": "same_speaker"},
                {"excerpt_id": "b", "source_audio_sha256": "b" * 64, "relation": "hard_negative"},
            ],
        )
        self.assertTrue(value["human_reviewed"])


if __name__ == "__main__":
    unittest.main()
