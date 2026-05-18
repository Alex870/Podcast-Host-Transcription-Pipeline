import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from podcast_transcribe.speakers import (
        detect_speaker_similarity_drift,
        final_host_profile_update,
        merge_profile,
        promotion_candidates,
        reference_sample_quality,
        speaker_aggregate_stats,
    )


@unittest.skipIf(np is None, "numpy is not installed in this Python environment")
class SpeakerProfileTests(unittest.TestCase):
    def test_final_known_host_controls_saved_profile(self):
        existing = np.array([1.0, 0.0], dtype=np.float32)
        guessed_host_profile = merge_profile(existing, np.array([1.0, 0.0], dtype=np.float32))
        known_host_embedding = np.array([0.0, 1.0], dtype=np.float32)

        updated = final_host_profile_update(
            existing_profile=existing,
            speaker_embeddings={
                "SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32),
                "SPEAKER_01": known_host_embedding,
            },
            final_host_speaker="SPEAKER_01",
            candidate_profile=guessed_host_profile,
        )

        expected = merge_profile(existing, known_host_embedding)
        np.testing.assert_allclose(updated, expected)

    def test_no_host_does_not_save_candidate_profile(self):
        existing = np.array([1.0, 0.0], dtype=np.float32)
        candidate = np.array([0.0, 1.0], dtype=np.float32)
        updated = final_host_profile_update(existing, {}, None, candidate)
        np.testing.assert_allclose(updated, existing)

    def test_reference_sample_quality_flags_short_quiet_samples(self):
        quality = reference_sample_quality(duration_seconds=4.0, rms=0.001, peak=0.2, speech_ratio=0.3)

        self.assertEqual(quality["rating"], "poor")
        self.assertIn("sample is very short", quality["warnings"])
        self.assertIn("sample is very quiet", quality["warnings"])

    def test_speaker_aggregate_stats_and_promotion_candidates(self):
        rows = [
            {"host_label": "SPEAKER_01", "host_duration_seconds": 250, "top_host_similarity": 0.6, "review_priority_score": 10},
            {"host_label": "SPEAKER_01", "host_duration_seconds": 250, "top_host_similarity": 0.7, "review_priority_score": 20},
            {"host_label": "SPEAKER_01", "host_duration_seconds": 150, "top_host_similarity": 0.65, "review_priority_score": 15},
        ]

        stats = speaker_aggregate_stats(rows)
        candidates = promotion_candidates(rows)

        self.assertEqual(stats["SPEAKER_01"]["episode_count"], 3)
        self.assertEqual(stats["SPEAKER_01"]["average_similarity"], 0.65)
        self.assertEqual(candidates[0]["speaker"], "SPEAKER_01")

    def test_detect_speaker_similarity_drift(self):
        alerts = detect_speaker_similarity_drift({"HOST": 0.5}, {"HOST": [0.75, 0.78, 0.77]})

        self.assertEqual(alerts[0]["speaker"], "HOST")
        self.assertGreater(alerts[0]["drop"], 0.2)


if __name__ == "__main__":
    unittest.main()

