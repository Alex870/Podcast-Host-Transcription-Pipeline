import json
import tempfile
import unittest
from pathlib import Path

from podcast_transcribe.contract_v2 import (
    EPISODE_CONTRACT_V2,
    episode_contract_status,
    upgrade_episode_bundle_v2,
)
from podcast_transcribe.ecosystem_contracts import (
    build_correction_manifest,
    normalize_correction_manifest,
)
from podcast_transcribe.speaker_workflow import group_recurring_unknown_speakers
from podcast_transcribe.workbench_core import evaluation_queues, propose_quality_campaign


class ContractV2Tests(unittest.TestCase):
    def test_legacy_bundle_upgrades_atomically_and_archives_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            audio = output / "Episode.mp3"
            audio.write_bytes(b"audio")
            cleaned = output / "Episode_cleaned_speaker_transcript.json"
            cleaned.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "source_file": str(audio),
                        "metadata": {
                            "stage_provenance": {"transcription": {"provider": {}}},
                            "speaker_identity_evidence_complete": True,
                        },
                        "speaker_identity_evidence_complete": True,
                        "segments": [{"id": 1, "text": "hello", "speaker": "HOST", "start": 0, "end": 1}],
                    }
                ),
                encoding="utf-8",
            )
            manifest = output / "Episode_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "source_file": str(audio),
                        "source_fingerprint": {"size_bytes": 5},
                        "stage_provenance": {"transcription": {"provider": {}}},
                        "outputs": [],
                    }
                ),
                encoding="utf-8",
            )
            result = upgrade_episode_bundle_v2(audio, output)
            upgraded = json.loads(cleaned.read_text(encoding="utf-8"))
            upgraded_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(upgraded["contract_version"], EPISODE_CONTRACT_V2)
            self.assertEqual(upgraded_manifest["contract_version"], EPISODE_CONTRACT_V2)
            self.assertEqual(
                episode_contract_status(upgraded, upgraded_manifest)["status"],
                "v2_complete",
            )
            self.assertTrue((output / "_contract_archive" / "Episode" / "v1" / cleaned.name).exists())
            second = upgrade_episode_bundle_v2(audio, output)
            self.assertEqual(second["status"], "v2_complete")

    def test_v1_correction_normalizes_and_new_writes_are_v2(self):
        transcript = {"segments": [{"source_span_id": "1", "text": "before"}]}
        correction = {
            "source_span_id": "1",
            "field": "text",
            "before": "before",
            "after": "after",
            "reason_code": "spelling",
            "adjudication_state": "accepted",
        }
        manifest = build_correction_manifest(
            transcript,
            [correction],
            reviewer="reviewer",
            producer={"name": "test", "contract_version": "2"},
        )
        self.assertEqual(manifest["contract_version"], "correction-manifest-v2")
        self.assertTrue(manifest["corrections"][0]["correction_id"].startswith("corr_"))
        legacy = {
            **manifest,
            "contract_version": "correction-manifest-v1",
            "accepted_corrections": [correction],
        }
        legacy.pop("corrections", None)
        normalized = normalize_correction_manifest(legacy)
        self.assertEqual(normalized["contract_version"], "correction-manifest-v2")

    def test_identical_local_labels_do_not_cluster_without_embedding_evidence(self):
        rows = [
            {"episode_id": "a", "speaker": "SPEAKER_01", "identity_evidence": {}},
            {"episode_id": "b", "speaker": "SPEAKER_01", "identity_evidence": {}},
        ]
        self.assertEqual(group_recurring_unknown_speakers(rows), [])

    def test_external_evaluation_pack_queues_and_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            output = root / "output"
            pack = Path(tmp) / "private-pack"
            root.mkdir()
            output.mkdir()
            pack.mkdir()
            (root / "podcast_transcribe_config.json").write_text(
                json.dumps({"evaluation_pack_path": str(pack)}),
                encoding="utf-8",
            )
            (pack / "manifest.json").write_text(
                json.dumps({"gold_set_version": 2, "entries": []}),
                encoding="utf-8",
            )
            for index in range(14):
                (output / f"Episode{index:02d}_cleaned_speaker_transcript.json").write_text(
                    json.dumps(
                        {
                            "source_file": f"Episode{index:02d}.mp3",
                            "transcription": {"duration": 60 + index * 100},
                            "metadata": {},
                            "segments": [],
                        }
                    ),
                    encoding="utf-8",
                )
            queues = evaluation_queues(root, output)
            campaign = propose_quality_campaign(root, output)
            self.assertEqual(queues["counts"]["unlabelled"], 14)
            self.assertEqual(len(campaign["selected"]), 12)
            self.assertEqual(queues["evaluation_pack_path"], str(pack.resolve()))


if __name__ == "__main__":
    unittest.main()
