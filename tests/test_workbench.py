import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from podcast_transcribe.outputs import build_episode_metadata, write_json_output
from podcast_transcribe.workbench_core import (
    apply_preferred_term_addition,
    apply_replacement_map_update,
    apply_text_correction,
    discover_episode_bundles,
    load_audit_log,
    load_episode_bundle,
    preview_text_correction,
    run_semantic_scan,
)


class WorkbenchCoreTests(unittest.TestCase):
    def _write_cleaned_payload(self, output_dir: Path, episode_name: str):
        metadata = build_episode_metadata(f"{episode_name}.mp3")
        cleaned_metadata = {**metadata, "text_version": "cleaned"}
        segment = SimpleNamespace(
            id=1,
            start=0.0,
            end=2.0,
            speaker="HOST",
            text="Chroma DB remains the storage layer for this workflow.",
            original_text="Chroma DB remains the storage layer for this workflow.",
            cleanup_applied=False,
            cleanup_level="normal",
            manual_correction_applied=False,
            original_speaker=None,
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            words=[],
        )
        write_json_output(
            output_dir / f"{episode_name}_cleaned_speaker_transcript.json",
            source_file=f"{episode_name}.mp3",
            info_payload={"duration": 2.0},
            diarized_turns=[],
            segments=[segment],
            speaker_mapping={"SPEAKER_00": "HOST"},
            host_speaker="SPEAKER_00",
            durations={"SPEAKER_00": 2.0},
            known_assignments={},
            metadata=cleaned_metadata,
            text_version="cleaned",
        )
        return segment

    def _write_reviewed_payload(self, output_dir: Path, episode_name: str):
        metadata = build_episode_metadata(f"{episode_name}.mp3")
        reviewed_metadata = {**metadata, "text_version": "reviewed_llm"}
        segment = SimpleNamespace(
            id=1,
            start=0.0,
            end=2.0,
            speaker="HOST",
            text="ChromaDB remains the storage layer for this workflow.",
            original_text="Chroma DB remains the storage layer for this workflow.",
            cleanup_applied=False,
            cleanup_level="normal",
            manual_correction_applied=False,
            original_speaker=None,
            llm_reviewed_text="ChromaDB remains the storage layer for this workflow.",
            review_runtime_profile="high_context_5090",
            review_backend="vllm",
            review_model_name="qwen-review",
            review_stage_flags={"glossary_correction_review": True},
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            words=[],
        )
        write_json_output(
            output_dir / f"{episode_name}_reviewed_speaker_transcript.json",
            source_file=f"{episode_name}.mp3",
            info_payload={"duration": 2.0},
            diarized_turns=[],
            segments=[segment],
            speaker_mapping={"SPEAKER_00": "HOST"},
            host_speaker="SPEAKER_00",
            durations={"SPEAKER_00": 2.0},
            known_assignments={},
            metadata=reviewed_metadata,
            text_version="reviewed_llm",
            review_metadata={
                "review_pipeline_version": 2,
                "review_stage_results": {"glossary_correction_review": {"status": "completed"}},
                "review_input_source": "inline_cleaned_segments",
                "review_runtime_profile": "high_context_5090",
                "review_backend": "vllm",
                "review_model_name": "qwen-review",
                "review_stage_flags": {"glossary_correction_review": True},
            },
        )

    def _write_project_config(self, project_root: Path):
        (project_root / "podcast_transcribe_config.json").write_text(
            json.dumps(
                {
                    "corrections_dir": "corrections",
                    "preferred_terms_file": "preferred_terms.txt",
                    "replacement_map_json": "preferred_replacements.json",
                    "runtime_profile": "high_context_5090",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (project_root / "preferred_terms.txt").write_text("LM Studio\n", encoding="utf-8")
        (project_root / "preferred_replacements.json").write_text(json.dumps({"ChromaDB": ["Chroma DB"]}, indent=2), encoding="utf-8")

    def test_discover_load_and_writeback_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            output_dir = project_root / "output"
            project_root.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            self._write_project_config(project_root)
            self._write_cleaned_payload(output_dir, "Episode 20260628")
            self._write_reviewed_payload(output_dir, "Episode 20260628")

            episodes = discover_episode_bundles(output_dir)
            self.assertEqual(len(episodes), 1)
            self.assertTrue(episodes[0]["has_reviewed"])

            bundle = load_episode_bundle(project_root, output_dir, "Episode 20260628")
            self.assertEqual(bundle["episode_id"], "Episode 20260628")
            self.assertEqual(len(bundle["cleaned"]["segments"]), 1)
            self.assertGreaterEqual(len(bundle["deterministic_findings"]), 1)

            preview = preview_text_correction(project_root, output_dir, "Episode 20260628", 1, "ChromaDB remains the storage layer.")
            self.assertEqual(preview["segment_id"], 1)

            applied = apply_text_correction(project_root, output_dir, "Episode 20260628", 1, "ChromaDB remains the storage layer.")
            self.assertEqual(applied["status"], "ok")
            correction_file = project_root / "corrections" / "Episode 20260628_corrections.csv"
            self.assertTrue(correction_file.exists())

            term_result = apply_preferred_term_addition(project_root, output_dir, "ChromaDB")
            self.assertEqual(term_result["status"], "ok")
            self.assertIn("ChromaDB", (project_root / "preferred_terms.txt").read_text(encoding="utf-8"))

            replacement_result = apply_replacement_map_update(project_root, output_dir, "LM Studio", "LM studio")
            self.assertEqual(replacement_result["status"], "ok")
            replacements = json.loads((project_root / "preferred_replacements.json").read_text(encoding="utf-8"))
            self.assertIn("LM studio", replacements["LM Studio"])

            audit_entries = load_audit_log(project_root, output_dir)
            self.assertGreaterEqual(len(audit_entries), 3)

    def test_semantic_scan_writes_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "project"
            output_dir = project_root / "output"
            project_root.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            self._write_project_config(project_root)
            self._write_cleaned_payload(output_dir, "Episode 20260628")

            with patch(
                "podcast_transcribe.workbench_core._openai_compatible_request",
                return_value={
                    "findings": [
                        {
                            "finding_id": "f1",
                            "issue_type": "possible_proper_noun_error",
                            "severity": "medium",
                            "reason": "Likely preferred spelling mismatch.",
                            "segment_ids": [1],
                            "suggested_text": "ChromaDB remains the storage layer for this workflow.",
                        }
                    ]
                },
            ):
                result = run_semantic_scan(project_root, output_dir, "Episode 20260628")

            self.assertEqual(result["finding_count"], 1)
            cache_path = output_dir / "_workbench" / "semantic_scan" / "Episode 20260628.semantic_scan.json"
            self.assertTrue(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
