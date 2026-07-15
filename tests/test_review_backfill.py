import tempfile
import unittest
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import types
from contextlib import redirect_stdout
from unittest.mock import patch


def _install_cli_test_stubs():
    try:
        import numpy as installed_numpy
    except ModuleNotFoundError:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = object
        numpy_stub.mean = lambda values, axis=0: values[0] if values else None
        numpy_stub.stack = lambda values: values
        numpy_stub.dot = lambda a, b: 0.0
        numpy_stub.linalg = types.SimpleNamespace(norm=lambda value: 1.0)
        sys.modules.setdefault("numpy", numpy_stub)
    else:
        # Keep the real numerical implementation available to the speaker
        # profile tests that run after this CLI-stub module is imported.
        sys.modules.setdefault("numpy", installed_numpy)

    scipy_stub = types.ModuleType("scipy")
    scipy_stub.__version__ = "1.0-test"
    sys.modules.setdefault("scipy", scipy_stub)

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.__version__ = "2.0-test"
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch_stub.device = lambda value: value
    sys.modules.setdefault("torch", torch_stub)

    torchaudio_stub = types.ModuleType("torchaudio")
    torchaudio_stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("torchaudio.load stub should not be used in this test"))
    torchaudio_stub.transforms = types.SimpleNamespace(Resample=type("Resample", (), {}))
    sys.modules.setdefault("torchaudio", torchaudio_stub)

    huggingface_stub = types.ModuleType("huggingface_hub")
    def _hf_hub_download(*args, **kwargs):
        return None
    huggingface_stub.hf_hub_download = _hf_hub_download
    sys.modules.setdefault("huggingface_hub", huggingface_stub)

    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = type("WhisperModel", (), {})
    sys.modules.setdefault("faster_whisper", faster_whisper_stub)

    pyannote_package = types.ModuleType("pyannote")
    pyannote_audio_stub = types.ModuleType("pyannote.audio")
    pyannote_audio_stub.Pipeline = type("Pipeline", (), {})
    pyannote_package.audio = pyannote_audio_stub
    sys.modules.setdefault("pyannote", pyannote_package)
    sys.modules.setdefault("pyannote.audio", pyannote_audio_stub)

    rich_package = types.ModuleType("rich")
    rich_progress_stub = types.ModuleType("rich.progress")
    for name in [
        "BarColumn",
        "Progress",
        "SpinnerColumn",
        "TaskProgressColumn",
        "TextColumn",
        "TimeElapsedColumn",
        "TimeRemainingColumn",
    ]:
        setattr(rich_progress_stub, name, type(name, (), {}))
    rich_package.progress = rich_progress_stub
    sys.modules.setdefault("rich", rich_package)
    sys.modules.setdefault("rich.progress", rich_progress_stub)


_install_cli_test_stubs()

from podcast_transcribe.cli import (
    build_chunk_speaker_embeddings,
    build_child_process_command,
    classify_episode_processing_state,
    diarization_route_decision,
    diarization_runtime_fingerprint,
    diarize_audio,
    load_diarization_history,
    normalize_episode_summary_row,
    parse_args,
    process_audio_batch,
    process_review_backfill_from_cleaned_json,
    resolve_ffprobe_path,
    run_alignment_stage,
    run_deterministic_cleanup_stage,
    run_speaker_attribution_stage,
    runtime_config_payload,
    update_diarization_history,
    write_episode_summary_csv,
)
from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.providers.contracts import ProviderIdentity, StageResult
from podcast_transcribe.outputs import build_episode_metadata, write_json_output, write_text_transcript


@dataclass
class Word:
    start: float
    end: float
    word: str
    speaker: str


class ReviewBackfillTests(unittest.TestCase):
    def test_ffprobe_resolution_rejects_conda_binary_when_no_external_build_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            conda_prefix = Path(tmp) / "conda"
            conda_ffprobe = conda_prefix / "Library" / "bin" / "ffprobe.exe"
            conda_ffprobe.parent.mkdir(parents=True)
            conda_ffprobe.write_bytes(b"test")
            with patch.dict(os.environ, {
                "PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR": "",
                "FFMPEG_BIN_DIR": "",
                "CONDA_PREFIX": str(conda_prefix),
            }, clear=False), patch(
                "podcast_transcribe.cli.shutil.which", return_value=str(conda_ffprobe)
            ):
                self.assertIsNone(resolve_ffprobe_path())

    def test_cli_accepts_provider_and_pipeline_benchmark_options(self):
        with patch.object(
            sys,
            "argv",
            [
                "podcast_transcribe_host.py",
                "--pipeline-benchmark",
                "--asr-provider",
                "faster_whisper",
                "--alignment-provider",
                "whisperx",
                "--speaker-embedding-provider",
                "speechbrain_ecapa",
                "--benchmark-baseline-dir",
                "baseline-output",
                "--review-context-budget",
                "64000",
                "--preferred-term",
                "ChromaDB",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.pipeline_benchmark)
        self.assertEqual(args.alignment_provider, "whisperx")
        self.assertEqual(args.benchmark_baseline_dir, "baseline-output")
        self.assertEqual(args.review_context_budget, 64000)
        self.assertEqual(args.preferred_terms, ["ChromaDB"])
        runtime = runtime_config_payload(args)
        self.assertEqual(runtime["review_context_budget"], 64000)
        self.assertEqual(runtime["preferred_terms"], ["ChromaDB"])

    def test_isolated_worker_preserves_explicit_review_and_custom_profile_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Episode.mp3"
            audio.write_bytes(b"audio")
            with patch.object(sys, "argv", [
                "podcast_transcribe_host.py", "--input-dir", str(root),
                "--no-transcript-cleanup-review", "--no-review-auto-calibrate",
                "--review-context-budget", "64000", "--review-structured-output-support",
                "--preferred-term", "ChromaDB",
            ]):
                args = parse_args()
            command = build_child_process_command(args, audio, root / "output")
            self.assertIn("--no-transcript-cleanup-review", command)
            self.assertIn("--no-review-auto-calibrate", command)
            self.assertIn("--review-context-budget", command)
            self.assertIn("--review-structured-output-support", command)
            self.assertEqual(command[command.index("--preferred-term") + 1], "ChromaDB")
            self.assertNotIn("--alignment-model", command)

    def test_isolated_worker_forwards_alignment_model_only_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Episode.mp3"
            audio.write_bytes(b"audio")
            with patch.object(sys, "argv", [
                "podcast_transcribe_host.py",
                "--input-dir", str(root),
                "--alignment-model", "large-v3",
            ]):
                args = parse_args()
            command = build_child_process_command(args, audio, root / "output")
            self.assertEqual(command[command.index("--alignment-model") + 1], "large-v3")

    def test_speaker_attribution_cache_reuses_matching_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Episode.mp3"
            audio.write_bytes(b"audio")
            reference = root / "guest.wav"
            reference.write_bytes(b"reference-a")
            host_profile = root / "host_profile.json"
            host_profile.write_text("{}", encoding="utf-8")
            segment = SegmentItem(1, 0.0, 1.0, "Hello", "SPEAKER_00", -0.1, 0.0, [])
            identity = ProviderIdentity("speaker_embedding", "speechbrain_ecapa", "test-model")
            kwargs = dict(
                output_dir=root, audio_path=audio, segments=[segment],
                diarized_turns=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], verifier=object(),
                speaker_embedding_identity=identity, host_reference=None, host_profile_path=str(host_profile),
                known_speaker_profiles={"Guest": {"sample_files": [str(reference)]}}, host_threshold=0.45, assume_dominant=True,
                max_embedding_seconds=90.0, min_host_seconds=20.0, historical_similarity_scores={},
                alignment_fingerprint={"hash": "alignment"}, diarization_fingerprint={"hash": "diarization"},
                resume_intermediates=True,
            )
            with patch("podcast_transcribe.cli.choose_host_speaker", return_value=("SPEAKER_00", {}, None, {"SPEAKER_00": 1.0}, {})) as choose, \
                 patch("podcast_transcribe.cli.match_known_speakers", return_value={}), \
                 patch("podcast_transcribe.cli.final_host_profile_update", return_value=None), \
                 patch("podcast_transcribe.cli.load_host_profile", return_value=None), \
                 patch("podcast_transcribe.cli.rename_speakers", return_value={"SPEAKER_00": "HOST"}):
                first = run_speaker_attribution_stage(**kwargs)
                second_kwargs = {**kwargs, "segments": [SegmentItem(1, 0.0, 1.0, "Hello", "SPEAKER_00", -0.1, 0.0, [])]}
                second = run_speaker_attribution_stage(**second_kwargs)
                reference.write_bytes(b"reference-b")
                third = run_speaker_attribution_stage(**second_kwargs)
                host_profile.write_text('{"changed": true}', encoding="utf-8")
                fourth = run_speaker_attribution_stage(**second_kwargs)
            self.assertFalse(first[-1])
            self.assertTrue(second[-1])
            self.assertFalse(third[-1])
            self.assertFalse(fourth[-1])
            self.assertEqual(choose.call_count, 3)

    def test_alignment_cache_reuses_then_invalidates_when_asr_dependency_changes(self):
        class FakeAlignment:
            identity = ProviderIdentity("alignment", "fake", "fake-model")

            def __init__(self):
                self.calls = 0

            def align(self, audio_path, segments, language):
                self.calls += 1
                return StageResult(segments, self.identity, {"aligned": True})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Episode.mp3"
            audio.write_bytes(b"audio")
            provider = FakeAlignment()
            segment = SegmentItem(1, 0.0, 1.0, "Hello", None, -0.1, 0.0, [])
            first = run_alignment_stage(root, audio, [segment], provider, "en", {"hash": "asr-a"}, True)
            second = run_alignment_stage(root, audio, [segment], provider, "en", {"hash": "asr-a"}, True)
            changed = run_alignment_stage(root, audio, [segment], provider, "en", {"hash": "asr-b"}, True)
            self.assertFalse(first[2])
            self.assertTrue(second[2])
            self.assertFalse(changed[2])
            self.assertEqual(provider.calls, 2)

    def test_cleanup_cache_reuses_and_invalidates_on_config_and_correction_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Episode.mp3"
            audio.write_bytes(b"audio")
            correction = root / "Episode_corrections.csv"
            correction.write_text("segment_id,corrected_text\n", encoding="utf-8")
            segment = SegmentItem(1, 0.0, 1.0, "you know, hello", "HOST", -0.1, 0.0, [])
            base = dict(output_dir=root, audio_path=audio, replacement_map={}, correction_path=correction,
                        speaker_attribution_fingerprint={"hash": "speaker"}, resume_intermediates=True)
            first = run_deterministic_cleanup_stage(segments=[segment], cleanup_level="normal", **base)
            second = run_deterministic_cleanup_stage(segments=[segment], cleanup_level="normal", **base)
            aggressive = run_deterministic_cleanup_stage(segments=[segment], cleanup_level="aggressive", **base)
            correction.write_text("segment_id,corrected_text\n1,Hello there\n", encoding="utf-8")
            corrected = run_deterministic_cleanup_stage(segments=[segment], cleanup_level="normal", **base)
            self.assertFalse(first[-1])
            self.assertTrue(second[-1])
            self.assertFalse(aggressive[-1])
            self.assertFalse(corrected[-1])

    def test_summary_csv_accepts_provider_and_alignment_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            row = {
                "episode": "Episode.mp3",
                "asr_provider": "faster_whisper",
                "alignment_provider": "timestamp_passthrough",
                "speaker_embedding_provider": "speechbrain_ecapa",
                "alignment_artifact_reused": True,
                "speaker_attribution_artifact_reused": True,
                "deterministic_cleanup_artifact_reused": True,
            }
            write_episode_summary_csv(path, [row])
            text = path.read_text(encoding="utf-8")
            self.assertIn("alignment_provider", text)
            self.assertIn("timestamp_passthrough", text)
            self.assertIn("speaker_attribution_artifact_reused", text)

    def _episode_qa_only_runtime_config(self):
        return {
            "runtime_profile": "high_context_5090",
            "backend": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "transcript_cleanup_review": False,
            "glossary_correction_review": False,
            "speaker_consistency_review": False,
            "episode_qa_review": True,
        }

    def _write_baseline_bundle(self, root: Path, audio_name: str):
        metadata = build_episode_metadata(audio_name)
        segment = SimpleNamespace(
            id=1,
            start=0.0,
            end=2.0,
            speaker="HOST",
            text="Reviewed baseline text.",
            original_text="Reviewed baseline text.",
            cleanup_applied=False,
            cleanup_level="conservative",
            manual_correction_applied=False,
            original_speaker=None,
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            words=[Word(0.0, 0.5, "Reviewed", "HOST")],
        )
        cleaned_metadata = {**metadata, "text_version": "cleaned"}
        write_text_transcript(root / "Episode 20260512_speaker_transcript.txt", [segment], lambda seconds: f"{seconds:.0f}", metadata=metadata)
        write_text_transcript(root / "Episode 20260512_host_only.txt", [segment], lambda seconds: f"{seconds:.0f}", host_only=True, host_labels={"HOST"}, metadata=metadata)
        write_text_transcript(root / "Episode 20260512_cleaned_speaker_transcript.txt", [segment], lambda seconds: f"{seconds:.0f}", metadata=cleaned_metadata)
        write_text_transcript(root / "Episode 20260512_cleaned_host_only.txt", [segment], lambda seconds: f"{seconds:.0f}", host_only=True, host_labels={"HOST"}, metadata=cleaned_metadata)
        (root / "Episode 20260512_review.csv").write_text("issue_type\n", encoding="utf-8")
        (root / "Episode 20260512_speaker_identity_review.csv").write_text("speaker_id\n", encoding="utf-8")
        write_json_output(
            root / "Episode 20260512_speaker_transcript.json",
            source_file=audio_name,
            info_payload={"duration": 2.0},
            diarized_turns=[],
            segments=[segment],
            speaker_mapping={"SPEAKER_00": "HOST"},
            host_speaker="SPEAKER_00",
            durations={"SPEAKER_00": 2.0},
            known_assignments={},
            metadata=metadata,
        )
        write_json_output(
            root / "Episode 20260512_cleaned_speaker_transcript.json",
            source_file=audio_name,
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

    def _write_current_reviewed_bundle(self, root: Path, audio_name: str):
        metadata = build_episode_metadata(audio_name)
        reviewed_metadata = {**metadata, "text_version": "reviewed_llm_high_context"}
        segment = SimpleNamespace(
            id=1,
            start=0.0,
            end=2.0,
            speaker="HOST",
            text="Reviewed enriched text.",
            original_text="Reviewed baseline text.",
            cleanup_applied=False,
            cleanup_level="conservative",
            manual_correction_applied=False,
            original_speaker=None,
            llm_reviewed_text="Reviewed enriched text.",
            review_runtime_profile="high_context_5090",
            review_backend="vllm",
            review_model_name="qwen-review",
            review_stage_flags={"episode_qa_review": True},
            avg_logprob=-0.1,
            no_speech_prob=0.01,
            words=[Word(0.0, 0.5, "Reviewed", "HOST")],
        )
        write_text_transcript(root / "Episode 20260512_reviewed_speaker_transcript.txt", [segment], lambda seconds: f"{seconds:.0f}", metadata=reviewed_metadata)
        write_text_transcript(root / "Episode 20260512_reviewed_host_only.txt", [segment], lambda seconds: f"{seconds:.0f}", host_only=True, host_labels={"HOST"}, metadata=reviewed_metadata)
        write_json_output(
            root / "Episode 20260512_reviewed_speaker_transcript.json",
            source_file=audio_name,
            info_payload={"duration": 2.0},
            diarized_turns=[],
            segments=[segment],
            speaker_mapping={"SPEAKER_00": "HOST"},
            host_speaker="SPEAKER_00",
            durations={"SPEAKER_00": 2.0},
            known_assignments={},
            metadata=reviewed_metadata,
            text_version="reviewed_llm_high_context",
            review_metadata={
                "review_pipeline_version": 2,
                "review_runtime_profile": "high_context_5090",
                "review_backend": "vllm",
                "review_model_name": "qwen-review",
                "review_stage_flags": {"episode_qa_review": True},
                "review_status": "completed",
                "review_skip_reason": "",
                "reviewed_segment_count": 1,
                "corrected_segment_count": 1,
                "episode_notes": [],
                "review_stage_results": {
                    "episode_qa_review": {
                        "attempted": True,
                        "status": "completed",
                        "skip_reason": "",
                        "corrected_segment_count": 1,
                        "edit_scope": "cross_segment_consistency",
                    }
                },
                "review_enabled_stages": ["episode_qa_review"],
                "review_completed_stages": ["episode_qa_review"],
                "review_skipped_stages": [],
                "episode_qa_mode": "full_episode",
                "review_input_source": "inline_cleaned_segments",
            },
        )

    def _write_legacy_reviewed_bundle(self, root: Path, audio_name: str):
        payload = {
            "schema_version": 2,
            "pipeline": "podcast-host-transcription-pipeline",
            "source_file": audio_name,
            "metadata": build_episode_metadata(audio_name),
            "episode_date": "2026-05-12",
            "episode_date_compact": "20260512",
            "episode_sort_key": 20260512,
            "text_version": "reviewed_llm",
            "segments": [
                {
                    "id": 1,
                    "start": 0.0,
                    "end": 2.0,
                    "speaker": "HOST",
                    "text": "Reviewed enriched text.",
                    "original_text": "Reviewed baseline text.",
                    "llm_reviewed_text": "Reviewed enriched text.",
                    "review_runtime_profile": "high_context_5090",
                    "review_backend": "vllm",
                    "review_model_name": "qwen-review",
                    "review_stage_flags": {"episode_qa_review": True},
                    "episode_date": "2026-05-12",
                    "episode_sort_key": 20260512,
                    "transcription_confidence": {"quality": "high", "warnings": []},
                    "words": [],
                }
            ],
        }
        (root / "Episode 20260512_reviewed_speaker_transcript.txt").write_text("reviewed", encoding="utf-8")
        (root / "Episode 20260512_reviewed_host_only.txt").write_text("reviewed", encoding="utf-8")
        (root / "Episode 20260512_reviewed_speaker_transcript.json").write_text(
            __import__("json").dumps(payload, indent=2),
            encoding="utf-8",
        )

    def test_classifier_marks_legacy_baseline_as_tier2_only_when_review_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)

            state = classify_episode_processing_state(
                audio_path,
                root,
                processed_files={audio_path.name: {}},
                existing_summary_rows={audio_path.name: {"episode": audio_path.name}},
                runtime_config=self._episode_qa_only_runtime_config(),
            )

            self.assertEqual(state["state"], "needs_tier2_only")
            self.assertTrue(state["cleaned_json_usable"])

    def test_classifier_rejects_invalid_cleaned_json_for_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)
            (root / "Episode 20260512_cleaned_speaker_transcript.json").write_text('{"segments":[{}]}', encoding="utf-8")

            state = classify_episode_processing_state(
                audio_path,
                root,
                processed_files={audio_path.name: {}},
                existing_summary_rows={audio_path.name: {"episode": audio_path.name}},
                runtime_config=self._episode_qa_only_runtime_config(),
            )

            self.assertEqual(state["state"], "needs_tier1")
            self.assertIn("failed contract validation", state["cleaned_json_error"])

    def test_classifier_treats_legacy_baseline_as_complete_when_review_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)

            state = classify_episode_processing_state(
                audio_path,
                root,
                processed_files={audio_path.name: {}},
                existing_summary_rows={audio_path.name: {"episode": audio_path.name}},
                runtime_config={"runtime_profile": "baseline_16gb", "backend": "none"},
            )

            self.assertEqual(state["state"], "complete")

    def test_classifier_requests_tier1_delta_when_alignment_provider_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_baseline_bundle(root, "Episode 20260512.mp3")
            audio = root / "Episode 20260512.mp3"
            audio.write_bytes(b"audio")
            cleaned_path = root / "Episode 20260512_cleaned_speaker_transcript.json"
            payload = json.loads(cleaned_path.read_text(encoding="utf-8"))
            payload.setdefault("metadata", {})["stage_provenance"] = {
                "alignment": {
                    "provider": {
                        "provider": "timestamp_passthrough",
                        "model": "asr_native_word_timestamps",
                    }
                }
            }
            cleaned_path.write_text(json.dumps(payload), encoding="utf-8")
            processed = {audio.name: {"size_bytes": audio.stat().st_size, "mtime_ns": audio.stat().st_mtime_ns}}
            summary = {audio.name: {"episode": audio.name}}

            state = classify_episode_processing_state(
                audio,
                root,
                processed,
                summary,
                {"alignment_provider": "whisperx", "backend": "none"},
            )

            self.assertEqual(state["state"], "needs_tier1")
            self.assertEqual(state["provider_shortfalls"], ["alignment:provider"])

    def test_classifier_requests_alignment_delta_for_legacy_unproven_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_baseline_bundle(root, "Episode 20260512.mp3")
            audio = root / "Episode 20260512.mp3"
            audio.write_bytes(b"audio")
            processed = {audio.name: {"size_bytes": audio.stat().st_size, "mtime_ns": audio.stat().st_mtime_ns}}
            summary = {audio.name: {"episode": audio.name}}

            state = classify_episode_processing_state(
                audio,
                root,
                processed,
                summary,
                {"alignment_provider": "whisperx", "backend": "none"},
            )

            self.assertEqual(state["state"], "needs_tier1")
            self.assertEqual(state["provider_shortfalls"], ["alignment:unproven_provider"])

    def test_classifier_treats_current_reviewed_bundle_as_complete_even_with_stale_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)
            self._write_current_reviewed_bundle(root, audio_path.name)

            state = classify_episode_processing_state(
                audio_path,
                root,
                processed_files={audio_path.name: {}},
                existing_summary_rows={audio_path.name: {"episode": audio_path.name, "reviewed_output_written": False}},
                runtime_config=self._episode_qa_only_runtime_config(),
            )

            self.assertEqual(state["state"], "complete")
            self.assertEqual(state["review_bundle_status"], "current_review_complete")

    def test_classifier_marks_legacy_reviewed_bundle_without_stage_proof_for_tier2_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)
            self._write_legacy_reviewed_bundle(root, audio_path.name)

            state = classify_episode_processing_state(
                audio_path,
                root,
                processed_files={audio_path.name: {}},
                existing_summary_rows={},
                runtime_config=self._episode_qa_only_runtime_config(),
            )

            self.assertEqual(state["state"], "needs_tier2_only")
            self.assertEqual(state["review_bundle_status"], "review_stage_shortfall")
            self.assertEqual(state["missing_review_stages"], ["episode_qa_review"])

    def test_classifier_marks_reviewed_bundle_missing_newly_enabled_stage_for_tier2_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)
            self._write_current_reviewed_bundle(root, audio_path.name)

            state = classify_episode_processing_state(
                audio_path,
                root,
                processed_files={audio_path.name: {}},
                existing_summary_rows={},
                runtime_config={
                    **self._episode_qa_only_runtime_config(),
                    "glossary_correction_review": True,
                },
            )

            self.assertEqual(state["state"], "needs_tier2_only")
            self.assertEqual(state["review_bundle_status"], "review_stage_shortfall")
            self.assertEqual(state["completed_review_stages"], ["episode_qa_review"])
            self.assertEqual(state["missing_review_stages"], ["glossary_correction_review"])

    def test_classifier_marks_corrupt_reviewed_bundle_for_tier2_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)
            (root / "Episode 20260512_reviewed_speaker_transcript.txt").write_text("reviewed", encoding="utf-8")
            (root / "Episode 20260512_reviewed_host_only.txt").write_text("reviewed", encoding="utf-8")
            (root / "Episode 20260512_reviewed_speaker_transcript.json").write_text('{"text_version":"reviewed_llm","segments":[{}]}', encoding="utf-8")

            state = classify_episode_processing_state(
                audio_path,
                root,
                processed_files={audio_path.name: {}},
                existing_summary_rows={},
                runtime_config=self._episode_qa_only_runtime_config(),
            )

            self.assertEqual(state["state"], "needs_tier2_only")
            self.assertEqual(state["review_bundle_status"], "review_corrupt")

    def test_review_backfill_writes_reviewed_outputs_and_summary_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)
            existing_summary = normalize_episode_summary_row(
                {
                    "episode": audio_path.name,
                    "review_priority_score": "0",
                    "host_detected": "true",
                    "transcript_segments": "1",
                    "review_row_count": "0",
                    "cleanup_level": "conservative",
                }
            )

            with patch("podcast_transcribe.cli.review_segments") as mock_review:
                mock_review.return_value = {
                    "attempted": True,
                    "skipped": False,
                    "skip_reason": "",
                    "segments": [
                        SimpleNamespace(
                            id=1,
                            start=0.0,
                            end=2.0,
                            speaker="HOST",
                            text="Reviewed enriched text.",
                            original_text="Reviewed baseline text.",
                            cleanup_applied=False,
                            cleanup_level="conservative",
                            manual_correction_applied=False,
                            original_speaker=None,
                            llm_reviewed_text="Reviewed enriched text.",
                            review_runtime_profile="high_context_5090",
                            review_backend="vllm",
                            review_model_name="qwen-review",
                            review_stage_flags={"episode_qa_review": True},
                            avg_logprob=-0.1,
                            no_speech_prob=0.01,
                            words=[Word(0.0, 0.5, "Reviewed", "HOST")],
                        )
                    ],
                    "metadata": {
                        "review_pipeline_version": 2,
                        "review_runtime_profile": "high_context_5090",
                        "review_backend": "vllm",
                        "review_model_name": "qwen-review",
                        "review_stage_flags": {"episode_qa_review": True},
                        "review_status": "completed",
                        "review_skip_reason": "",
                        "reviewed_segment_count": 1,
                        "corrected_segment_count": 1,
                        "episode_notes": ["Backfilled."],
                        "review_stage_results": {
                            "episode_qa_review": {
                                "attempted": True,
                                "status": "completed",
                                "skip_reason": "",
                                "corrected_segment_count": 1,
                                "edit_scope": "cross_segment_consistency",
                            }
                        },
                        "review_enabled_stages": ["episode_qa_review"],
                        "review_completed_stages": ["episode_qa_review"],
                        "review_skipped_stages": [],
                        "episode_qa_mode": "full_episode",
                        "review_input_source": "cleaned_json_backfill",
                    },
                }
                summary = process_review_backfill_from_cleaned_json(
                    audio_path=audio_path,
                    output_dir=root,
                    runtime_config={
                        "model": "large-v3",
                        "runtime_profile": "high_context_5090",
                        "backend": "vllm",
                        "review_base_url": "http://127.0.0.1:8000",
                        "review_model_name": "qwen-review",
                    },
                    existing_summary_row=existing_summary,
                )

            self.assertTrue((root / "Episode 20260512_reviewed_speaker_transcript.json").exists())
            self.assertTrue((root / "Episode 20260512_reviewed_speaker_transcript.txt").exists())
            self.assertEqual(summary["processing_mode"], "tier2-only backfill")
            self.assertTrue(summary["tier1_reused_from_existing"])
            self.assertTrue(summary["review_backfilled_from_cleaned_json"])
            self.assertTrue(summary["reviewed_output_written"])

    def test_review_backfill_prints_stage_and_chunk_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")
            self._write_baseline_bundle(root, audio_path.name)

            def fake_review(
                segments,
                runtime_review_config,
                review_input_source="inline_cleaned_segments",
                calibration_session=None,
                progress_callback=None,
                debug_context=None,
            ):
                if progress_callback:
                    progress_callback({"event": "stage_index", "stage_name": "glossary_correction_review", "stage_label": "glossary", "current": 1, "total": 2})
                    progress_callback({"event": "stage_window_progress", "stage_name": "glossary_correction_review", "stage_label": "glossary", "mode": "local_batch", "current": 2, "total": 5})
                    progress_callback({"event": "stage_index", "stage_name": "episode_qa_review", "stage_label": "episode_qa", "current": 2, "total": 2})
                    progress_callback({"event": "stage_window_progress", "stage_name": "episode_qa_review", "stage_label": "episode_qa", "mode": "chunked", "current": 3, "total": 9})
                return {
                    "attempted": True,
                    "skipped": False,
                    "skip_reason": "",
                    "segments": [],
                    "metadata": {
                        "review_pipeline_version": 2,
                        "review_runtime_profile": "high_context_5090",
                        "review_backend": "vllm",
                        "review_model_name": "qwen-review",
                        "review_stage_flags": {"glossary_correction_review": True, "episode_qa_review": True},
                        "review_status": "completed",
                        "review_skip_reason": "",
                        "reviewed_segment_count": 0,
                        "corrected_segment_count": 0,
                        "episode_notes": [],
                        "review_stage_results": {
                            "glossary_correction_review": {"attempted": True, "status": "completed", "skip_reason": "", "corrected_segment_count": 0, "edit_scope": "text_only"},
                            "episode_qa_review": {"attempted": True, "status": "completed", "skip_reason": "", "corrected_segment_count": 0, "edit_scope": "cross_segment_consistency"},
                        },
                        "review_enabled_stages": ["glossary_correction_review", "episode_qa_review"],
                        "review_completed_stages": ["glossary_correction_review", "episode_qa_review"],
                        "review_skipped_stages": [],
                        "episode_qa_mode": "chunked",
                        "review_input_source": "cleaned_json_backfill",
                    },
                }

            buffer = io.StringIO()
            with patch("podcast_transcribe.cli.review_segments", side_effect=fake_review):
                with redirect_stdout(buffer):
                    process_review_backfill_from_cleaned_json(
                        audio_path=audio_path,
                        output_dir=root,
                        runtime_config={
                            "model": "large-v3",
                            "runtime_profile": "high_context_5090",
                            "backend": "vllm",
                            "review_base_url": "http://127.0.0.1:8000",
                            "review_model_name": "qwen-review",
                        },
                    )
            text = buffer.getvalue()
            self.assertIn("Episode mode: tier2-only backfill", text)
            self.assertIn("Episode stage 1/3: load cleaned transcript", text)
            self.assertIn("Episode stage 2/3: review", text)
            self.assertIn("Review stage 1/2: glossary", text)
            self.assertIn("glossary window 2/5", text)
            self.assertIn("episode qa chunk 3/9", text)

    def test_single_episode_worker_does_not_print_nested_batch_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "Episode 20260512.mp3"
            audio_path.write_bytes(b"audio")

            args = SimpleNamespace(
                input_file=str(audio_path),
                preferred_terms_file=None,
                replacement_map_json=None,
                device="cpu",
                language="en",
                beam_size=5,
                batch_size=8,
                host_reference=None,
                host_profile_json="host_profile.json",
                known_speakers_dir=None,
                host_threshold=0.45,
                assume_dominant_speaker_is_host=True,
                max_embedding_seconds=120.0,
                min_host_seconds=15.0,
                num_speakers=None,
                cleanup_level="normal",
                corrections_dir=None,
                resume_intermediates=True,
                archive_debug_artifacts=False,
                runtime_profile="baseline_16gb",
                backend="none",
                review_base_url="",
                review_model_name="",
                transcript_cleanup_review=None,
                glossary_correction_review=None,
                speaker_consistency_review=None,
                episode_qa_review=None,
                filename_date_preset="strict_iso",
                filename_date_position="last",
                filename_date_formats=None,
                model="distil-large-v3",
                compute_type="float16",
            )

            summary_row = normalize_episode_summary_row(
                {
                    "episode": audio_path.name,
                    "review_priority_score": "0",
                    "host_detected": "true",
                    "transcript_segments": "1",
                    "review_row_count": "0",
                }
            )
            buffer = io.StringIO()
            with patch("podcast_transcribe.cli.load_preferred_terms", return_value=[]), \
                patch("podcast_transcribe.cli.build_prompt_bias", return_value=(None, None)), \
                patch("podcast_transcribe.cli.classify_episode_processing_state", return_value={"state": "needs_tier2_only"}), \
                patch("podcast_transcribe.cli.process_review_backfill_from_cleaned_json", return_value=summary_row), \
                patch("podcast_transcribe.cli.write_episode_summary_csv"), \
                patch("podcast_transcribe.cli.state_save_processed_files"), \
                patch("podcast_transcribe.cli.output_write_batch_report_md"):
                with redirect_stdout(buffer):
                    process_audio_batch(args, root, root, [audio_path])
            text = buffer.getvalue()
            self.assertNotIn("Batch progress: file 1 of 1", text)
            self.assertNotIn("Processing mode for Episode 20260512.mp3", text)

    def test_diarization_route_uses_probe_band_and_recent_failure_suppression(self):
        fingerprint = diarization_runtime_fingerprint("pyannote/test-model", "path_input")
        history_state = {
            "records": [
                {
                    "audio_file": "fail.mp3",
                    "duration_seconds": 4 * 3600 + 10 * 60,
                    "mode": "global",
                    "outcome": "memory_error",
                    "probe": False,
                    "timestamp_epoch_seconds": 100.0,
                    "runtime_fingerprint": fingerprint,
                    "invalidated": False,
                }
            ]
        }

        route = diarization_route_decision(4 * 3600 + 28 * 60, history_state, fingerprint)
        self.assertEqual(route["mode"], "global")
        self.assertTrue(route["probe"])

        exact_floor_route = diarization_route_decision(4 * 3600 + 10 * 60, history_state, fingerprint)
        self.assertEqual(exact_floor_route["mode"], "chunked_preemptive")
        self.assertEqual(exact_floor_route["reason"], "at_or_below_failure_floor")

        history_state["records"].append(
            {
                "audio_file": "probe-fail.mp3",
                "duration_seconds": 4 * 3600 + 24 * 60,
                "mode": "global",
                "outcome": "memory_error",
                "probe": True,
                "timestamp_epoch_seconds": 200.0,
                "runtime_fingerprint": fingerprint,
                "invalidated": False,
            }
        )
        route = diarization_route_decision(4 * 3600 + 28 * 60, history_state, fingerprint)
        self.assertEqual(route["mode"], "chunked_preemptive")
        self.assertFalse(route["probe"])
        self.assertEqual(route["reason"], "recent_nearby_probe_failed")

    def test_diarization_history_invalidates_failure_after_shorter_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            audio_fail = output_dir / "fail.mp3"
            audio_success = output_dir / "success.mp3"
            audio_fail.write_bytes(b"fail")
            audio_success.write_bytes(b"success")
            fingerprint = diarization_runtime_fingerprint("pyannote/test-model", "path_input")

            update_diarization_history(
                output_dir,
                fingerprint,
                audio_fail,
                4 * 3600 + 10 * 60,
                "global",
                "memory_error",
                probe=False,
            )
            update_diarization_history(
                output_dir,
                fingerprint,
                audio_success,
                4 * 3600,
                "global",
                "success",
                probe=False,
            )
            history = load_diarization_history(output_dir)
            active_failures = [
                record for record in history["records"]
                if record.get("runtime_fingerprint") == fingerprint
                and record.get("mode") == "global"
                and record.get("outcome") == "memory_error"
                and not record.get("invalidated")
            ]
            self.assertEqual(active_failures, [])

    def test_diarize_audio_retries_chunked_after_memory_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            audio_path = output_dir / "Episode.mp3"
            audio_path.write_bytes(b"audio")

            with patch("podcast_transcribe.cli.get_audio_duration_seconds", return_value=4 * 3600 + 20 * 60), patch(
                "podcast_transcribe.cli.pyannote_path_input_available",
                return_value=True,
            ), patch(
                "podcast_transcribe.cli._call_global_diarization",
                side_effect=MemoryError("unable to allocate array data"),
            ), patch(
                "podcast_transcribe.cli.diarize_audio_chunked",
                return_value=(
                    [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
                    {
                        "mode": "chunked_fallback_after_failure",
                        "chunk_count": 3,
                        "chunk_overlap_seconds": 90.0,
                        "reconciliation_merge_count": 2,
                        "reconciliation_ambiguous_count": 1,
                    },
                ),
            ):
                turns, metadata = diarize_audio(
                    output_dir=output_dir,
                    pipeline=object(),
                    diarization_model_id="pyannote/test-model",
                    verifier=object(),
                    audio_path=str(audio_path),
                    num_speakers=None,
                    max_embedding_seconds=120.0,
                )

            self.assertEqual(len(turns), 1)
            self.assertEqual(metadata["mode"], "chunked_fallback_after_failure")
            history = load_diarization_history(output_dir)
            outcomes = [(record["mode"], record["outcome"]) for record in history["records"]]
            self.assertIn(("global", "memory_error"), outcomes)
            self.assertIn(("chunked_fallback_after_failure", "success"), outcomes)

    def test_build_chunk_speaker_embeddings_skips_tiny_or_invalid_clips(self):
        tiny_clip = SimpleNamespace(numel=lambda: 10, shape=(1000,))
        bad_clip = SimpleNamespace(numel=lambda: 10, shape=(16000,))
        good_clip = SimpleNamespace(numel=lambda: 10, shape=(32000,))
        with patch(
            "podcast_transcribe.cli.build_speaker_audio_samples",
            return_value={
                "tiny": tiny_clip,
                "bad": bad_clip,
                "good": good_clip,
            },
        ), patch(
            "podcast_transcribe.cli.compute_embedding",
            side_effect=[
                RuntimeError("Padding size should be less than the corresponding input dimension"),
                "good-embedding",
            ],
        ):
            embeddings = build_chunk_speaker_embeddings(
                verifier=object(),
                audio_path="episode.mp3",
                diarized_turns=[],
                max_seconds=10.0,
            )
        self.assertEqual(embeddings, {"good": "good-embedding"})


if __name__ == "__main__":
    unittest.main()
