import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from podcast_transcribe_speakers import final_host_profile_update, merge_profile


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


if __name__ == "__main__":
    unittest.main()
