import hashlib
import json
import unittest
from pathlib import Path

from podcast_transcribe.ecosystem_contracts import (
    ContractError, apply_preview, build_correction_manifest, canonical_id,
    preview_corrections, transcript_hash, validate_correction_manifest,
)


TRANSCRIPT = {"segments": [{"source_span_id": "span-1", "text": "café", "speaker": "A"}]}
CORRECTION = {
    "source_span_id": "span-1", "field": "text", "before": "café", "after": "cafe",
    "reason_code": "spelling", "adjudication_state": "accepted",
}
PRODUCER = {"name": "podcast-host-transcription-pipeline", "contract_version": "1"}


class CorrectionContractTests(unittest.TestCase):
    def test_canonical_v2_fixture_and_origin_checksum(self):
        fixture_dir = Path(__file__).parent / "fixtures" / "contracts" / "correction-manifest-v2"
        fixture_path = fixture_dir / "valid.json"
        origin = json.loads((fixture_dir / "origin.json").read_text(encoding="utf-8"))
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(fixture_path.read_bytes()).hexdigest(), origin["sha256"])
        validate_correction_manifest(payload["manifest"], payload["transcript"])
        self.assertEqual(len(payload["manifest"]["accepted_corrections"]), 1)
        self.assertEqual(payload["manifest"]["corrections"][1]["status"], "rejected")

    def test_canonical_unicode_and_order(self):
        self.assertEqual(canonical_id({"b": "é", "a": 1}, prefix="x"), canonical_id({"a": 1, "b": "é"}, prefix="x"))

    def test_deterministic_replay_and_mutable_notes(self):
        first = build_correction_manifest(TRANSCRIPT, [CORRECTION], reviewer="reviewer-1", producer=PRODUCER)
        second = build_correction_manifest(TRANSCRIPT, [CORRECTION], reviewer="reviewer-1", producer=PRODUCER)
        second["notes"] = "local note"
        validate_correction_manifest(second, TRANSCRIPT)
        self.assertEqual(first["correction_set_id"], second["correction_set_id"])

    def test_content_parent_and_config_are_authoritative(self):
        first = build_correction_manifest(TRANSCRIPT, [CORRECTION], reviewer="reviewer-1", producer=PRODUCER)
        changed = dict(TRANSCRIPT)
        changed["segments"] = [{**TRANSCRIPT["segments"][0], "speaker": "B"}]
        self.assertNotEqual(transcript_hash(TRANSCRIPT), transcript_hash(changed))
        altered = build_correction_manifest(TRANSCRIPT, [CORRECTION], reviewer="reviewer-1", producer={**PRODUCER, "config_fingerprint": "new"})
        self.assertNotEqual(first["correction_set_id"], altered["correction_set_id"])

    def test_stale_hash_and_before_value_fail(self):
        manifest = build_correction_manifest(TRANSCRIPT, [CORRECTION], reviewer="reviewer-1", producer=PRODUCER)
        with self.assertRaisesRegex(ContractError, "stale"):
            validate_correction_manifest(manifest, {"segments": []})
        bad = {**CORRECTION, "before": "wrong"}
        with self.assertRaisesRegex(ContractError, "before"):
            build_correction_manifest(TRANSCRIPT, [bad], reviewer="reviewer-1", producer=PRODUCER)

    def test_preview_requires_exact_approval_and_matches_apply(self):
        preview = preview_corrections(TRANSCRIPT, [CORRECTION], reviewer="reviewer-1", producer=PRODUCER)
        with self.assertRaisesRegex(ContractError, "approval"):
            apply_preview(preview, TRANSCRIPT, approved_preview_id="wrong")
        result, manifest = apply_preview(preview, TRANSCRIPT, approved_preview_id=preview["preview_id"])
        self.assertEqual("cafe", result["segments"][0]["text"])
        self.assertEqual(manifest["result_transcript_hash"], transcript_hash(result))


if __name__ == "__main__":
    unittest.main()
