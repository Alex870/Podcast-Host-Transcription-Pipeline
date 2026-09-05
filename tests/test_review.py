import unittest
import json
import tempfile
from unittest.mock import MagicMock, patch

from podcast_transcribe.models import SegmentItem, WordItem
from podcast_transcribe.review import (
    ReviewCalibrationSession,
    _execute_stage_backend_request,
    _normalize_backend_response_text,
    _normalize_episode_notes,
    _openai_compatible_chat_completion,
    _segment_prompt_payload,
    enrich_backend_capabilities_with_identity,
    resolve_backend_capabilities,
    review_segments,
)


def make_segment(segment_id: int, speaker: str, text: str) -> SegmentItem:
    return SegmentItem(
        id=segment_id,
        start=float(segment_id),
        end=float(segment_id) + 1.0,
        text=text,
        speaker=speaker,
        avg_logprob=-0.1,
        no_speech_prob=0.01,
        words=[WordItem(start=float(segment_id), end=float(segment_id) + 0.1, word=text.split()[0], speaker=speaker)],
    )


class ReviewTests(unittest.TestCase):
    def test_candidate_filter_reviews_only_uncertain_segments_and_keeps_full_output(self):
        segments = [make_segment(1, "HOST", "high confidence"), make_segment(2, "HOST", "uncertain")]
        segments[1].avg_logprob = -1.4
        seen_ids = []

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            seen_ids.append([segment.id for segment in window])
            return {
                "reviewed_segments": [{"id": 2, "text": "uncertain corrected"}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": True,
                    "review_context_budget": 16000,
                    "review_candidate_filter": True,
                },
            )

        self.assertEqual(seen_ids, [[2]])
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0].text, "high confidence")
        self.assertEqual(result["segments"][1].text, "uncertain corrected")
        self.assertEqual(result["metadata"]["review_candidate_count"], 1)
        self.assertEqual(result["metadata"]["review_skipped_segment_count"], 1)

    def test_configured_review_batch_limit_caps_calibration_ceiling(self):
        capabilities = resolve_backend_capabilities(
            {
                "runtime_profile": "high_context_5090",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
                "review_batch_token_limit": 8192,
            }
        )
        session = ReviewCalibrationSession(capabilities)
        self.assertLessEqual(session.families["local_text_review"]["hard_ceiling"], 8192)
        self.assertLessEqual(session.families["long_context_review"]["hard_ceiling"], 8192)

    def test_completed_review_stage_is_reused_from_checkpoint(self):
        segments = [make_segment(1, "HOST", "uncertain")]
        segments[0].avg_logprob = -1.4
        with tempfile.TemporaryDirectory() as tmp:
            debug_context = {
                "audio_path": str(__import__("pathlib").Path(tmp) / "episode.mp3"),
                "output_dir": tmp,
                "review_input_source": "inline_cleaned_segments",
            }
            __import__("pathlib").Path(debug_context["audio_path"]).write_bytes(b"audio")
            runtime = {
                "runtime_profile": "custom",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
                "transcript_cleanup_review": True,
                "review_context_budget": 16000,
                "review_candidate_filter": True,
            }
            with patch(
                "podcast_transcribe.review._execute_stage_backend_request",
                return_value={
                    "reviewed_segments": [{"id": 1, "text": "corrected"}],
                    "corrected_segment_count": 1,
                    "episode_notes": [],
                },
            ) as backend_call:
                first = review_segments(segments, runtime, debug_context=debug_context)
                backend_call.reset_mock()
                second = review_segments(segments, runtime, debug_context=debug_context)

            self.assertEqual(first["segments"][0].text, "corrected")
            self.assertEqual(second["segments"][0].text, "corrected")
            self.assertEqual(second["metadata"]["review_resume_source"], "review_progress_checkpoint")
            backend_call.assert_not_called()

    def test_qwen_reasoning_control_is_sent_per_request(self):
        response = MagicMock()
        response.read.return_value = b'{}'
        backend = {
            "backend_name": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "Qwen3.8-27B",
        }

        with patch("podcast_transcribe.review.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            for effort, expected in (("none", {"enable_thinking": False}), ("low", {"enable_thinking": True, "reasoning_effort": "low"})):
                with self.subTest(effort=effort):
                    backend["review_reasoning_effort"] = effort
                    _openai_compatible_chat_completion(backend, "system", "user", 512)
                    request = urlopen.call_args.args[0]
                    payload = json.loads(request.data.decode("utf-8"))
                    self.assertEqual(payload["chat_template_kwargs"], expected)

    def test_normalizer_falls_back_to_reasoning_field(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "```json\n{\"reviewed_segments\": [], \"corrected_segment_count\": 0, \"episode_notes\": []}\n```",
                    }
                }
            ]
        }

        text = _normalize_backend_response_text(response)
        self.assertIn("\"reviewed_segments\"", text)

    def test_execute_stage_request_accepts_reasoning_json_and_strips_fences(self):
        segment = make_segment(1, "HOST", "short text")
        backend_capabilities = {
            "runtime_profile": "high_context_5090",
            "backend_name": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "max_context_budget": 16000,
            "structured_output_support": True,
        }
        stage_definition = {
            "name": "transcript_cleanup_review",
            "label": "cleanup",
            "edit_scope": "text_only",
            "description": "Conservative transcript cleanup only.",
        }

        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "```json\n{\"reviewed_segments\":[{\"id\":1,\"text\":\"better text\"}],\"corrected_segment_count\":1,\"episode_notes\":[]}\n```",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch("podcast_transcribe.review._openai_compatible_chat_completion", return_value=__import__("json").dumps(raw_response)):
            payload = _execute_stage_backend_request([segment], backend_capabilities, stage_definition, "local_batch")

        self.assertEqual(payload["corrected_segment_count"], 1)
        self.assertEqual(payload["reviewed_segments"][0]["text"], "better text")

    def test_segment_prompt_payload_is_compact_for_review_requests(self):
        segment = make_segment(1, "HOST", "short text")

        payload = _segment_prompt_payload([segment])

        self.assertEqual(payload, [{"id": 1, "speaker": "HOST", "text": "short text"}])

    def test_execute_stage_request_reports_truncated_json_cleanly(self):
        segment = make_segment(1, "HOST", "short text")
        backend_capabilities = {
            "runtime_profile": "high_context_5090",
            "backend_name": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "max_context_budget": 16000,
            "structured_output_support": True,
        }
        stage_definition = {
            "name": "transcript_cleanup_review",
            "label": "cleanup",
            "edit_scope": "text_only",
            "description": "Conservative transcript cleanup only.",
        }

        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "```json\n{\"reviewed_segments\":[{\"id\":1,\"text\":\"better text\"}]",
                    },
                    "finish_reason": "length",
                }
            ]
        }

        with patch("podcast_transcribe.review._openai_compatible_chat_completion", return_value=__import__("json").dumps(raw_response)):
            with self.assertRaises(RuntimeError) as context:
                _execute_stage_backend_request([segment], backend_capabilities, stage_definition, "local_batch")

        self.assertIn("truncated", str(context.exception).lower())

    def test_execute_stage_request_extracts_json_object_from_wrapped_response(self):
        segment = make_segment(1, "HOST", "short text")
        backend_capabilities = {
            "runtime_profile": "high_context_5090",
            "backend_name": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "max_context_budget": 16000,
            "structured_output_support": True,
        }
        stage_definition = {
            "name": "transcript_cleanup_review",
            "label": "cleanup",
            "edit_scope": "text_only",
            "description": "Conservative transcript cleanup only.",
        }

        raw_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here is the patch you requested:\n{\"reviewed_segments\":[{\"id\":1,\"text\":\"better text\"}],\"corrected_segment_count\":1,\"episode_notes\":[]}\nDone.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch("podcast_transcribe.review._openai_compatible_chat_completion", return_value=__import__("json").dumps(raw_response)):
            payload = _execute_stage_backend_request([segment], backend_capabilities, stage_definition, "local_batch")

        self.assertEqual(payload["corrected_segment_count"], 1)
        self.assertEqual(payload["reviewed_segments"][0]["text"], "better text")

    def test_execute_stage_request_includes_active_learned_rules(self):
        segment = make_segment(1, "HOST", "short text")
        backend_capabilities = {
            "runtime_profile": "high_context_5090",
            "backend_name": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "max_context_budget": 16000,
            "structured_output_support": True,
        }
        stage_definition = {
            "name": "glossary_correction_review",
            "label": "glossary",
            "edit_scope": "text_only",
            "description": "Preferred-term consistency only.",
        }
        captured = {}

        def fake_chat_completion(backend_capabilities, system_prompt, user_prompt, max_output_tokens):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            return __import__("json").dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "{\"reviewed_segments\":[],\"corrected_segment_count\":0,\"episode_notes\":[]}",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        with patch("podcast_transcribe.review._openai_compatible_chat_completion", side_effect=fake_chat_completion):
            _execute_stage_backend_request(
                [segment],
                backend_capabilities,
                stage_definition,
                "local_batch",
                learned_rules=[
                    {
                        "rule_id": "rule_1",
                        "summary": "Prefer ChromaDB over Chroma DB.",
                        "directive": "Prefer ChromaDB over Chroma DB.",
                    }
                ],
            )

        self.assertIn("Active learned rules", captured["system_prompt"])
        self.assertIn("\"learned_rules\"", captured["user_prompt"])

    def test_local_batch_request_uses_tighter_output_cap(self):
        segment = make_segment(1, "HOST", "short text")
        backend_capabilities = {
            "runtime_profile": "high_context_5090",
            "backend_name": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "max_context_budget": 32000,
            "structured_output_support": True,
        }
        stage_definition = {
            "name": "transcript_cleanup_review",
            "label": "cleanup",
            "edit_scope": "text_only",
            "description": "Conservative transcript cleanup only.",
        }
        captured = {}

        def fake_chat_completion(backend_capabilities, system_prompt, user_prompt, max_output_tokens):
            captured["max_output_tokens"] = max_output_tokens
            captured["user_prompt"] = user_prompt
            captured["system_prompt"] = system_prompt
            return __import__("json").dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "{\"reviewed_segments\":[],\"corrected_segment_count\":0,\"episode_notes\":[]}",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        with patch("podcast_transcribe.review._openai_compatible_chat_completion", side_effect=fake_chat_completion):
            _execute_stage_backend_request([segment], backend_capabilities, stage_definition, "local_batch")

        self.assertLessEqual(captured["max_output_tokens"], 1400)
        self.assertIn("\"max_changed_segments_hint\": 8", captured["user_prompt"])
        self.assertTrue(captured.get("system_prompt", "").startswith("/no_think "))

    def test_normalize_episode_notes_accepts_string_or_list(self):
        self.assertEqual(
            _normalize_episode_notes({"episode_notes": "Single summary note."}),
            ["Single summary note."],
        )
        self.assertEqual(
            _normalize_episode_notes({"episode_notes": [" one ", "", "two"]}),
            ["one", "two"],
        )

    def test_cleanup_only_review_edits_text_but_not_speaker(self):
        segments = [make_segment(1, "HOST", "bad text")]

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            self.assertEqual(stage_definition["name"], "transcript_cleanup_review")
            return {
                "reviewed_segments": [{"id": 1, "text": "better text", "speaker": "GUEST"}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": True,
                    "review_context_budget": 16000,
                },
            )

        self.assertTrue(result["attempted"])
        self.assertEqual(result["segments"][0].text, "better text")
        self.assertEqual(result["segments"][0].speaker, "HOST")
        self.assertIsInstance(result["segments"][0].words[0], WordItem)
        self.assertEqual(result["metadata"]["review_completed_stages"], ["transcript_cleanup_review"])

    def test_preferred_term_regression_is_reverted(self):
        segments = [make_segment(1, "HOST", "ChromaDB stays exactly spelled this way.")]

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            self.assertIn("ChromaDB", preferred_terms or [])
            return {
                "reviewed_segments": [{"id": 1, "text": "Chroma DB stays exactly spelled this way."}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": True,
                    "review_context_budget": 16000,
                    "preferred_terms": ["ChromaDB"],
                },
            )

        self.assertEqual(result["segments"][0].text, "ChromaDB stays exactly spelled this way.")
        self.assertEqual(result["metadata"]["protected_term_violation_count"], 1)
        self.assertEqual(result["metadata"]["review_guard_interventions"]["protected_term_preservations"], 1)

    def test_cleanup_review_recursively_splits_truncated_windows(self):
        segments = [
            make_segment(1, "HOST", "alpha"),
            make_segment(2, "HOST", "beta"),
        ]
        seen_lengths = []

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            seen_lengths.append(len(window))
            if len(window) > 1:
                raise RuntimeError("Review backend response was truncated before the JSON payload completed.")
            segment = window[0]
            return {
                "reviewed_segments": [{"id": segment.id, "text": f"{segment.text} revised"}],
                "corrected_segment_count": 1,
                "episode_notes": "split retry succeeded",
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": True,
                    "review_context_budget": 16000,
                },
            )

        self.assertEqual(seen_lengths, [2, 1, 1])
        self.assertEqual(result["segments"][0].text, "alpha revised")
        self.assertEqual(result["segments"][1].text, "beta revised")
        self.assertIn("split retry succeeded", " ".join(result["metadata"]["episode_notes"]))

    def test_cleanup_review_splits_oversized_single_segment_into_synthetic_chunks(self):
        long_text = " ".join(f"word{i}" for i in range(400))
        segments = [make_segment(1, "HOST", long_text)]
        seen_windows = []

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            seen_windows.append([segment.id for segment in window])
            if len(window) == 1 and window[0].id == 1:
                raise RuntimeError("Review backend response was truncated before the JSON payload completed.")
            reviewed_segments = []
            for segment in window:
                if str(segment.text).startswith("word0 "):
                    reviewed_segments.append({"id": segment.id, "text": f"{segment.text} revised"})
            return {
                "reviewed_segments": reviewed_segments,
                "corrected_segment_count": len(reviewed_segments),
                "episode_notes": ["synthetic split retry succeeded"],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": True,
                    "review_context_budget": 16000,
                },
            )

        self.assertEqual(seen_windows[0], [1])
        self.assertGreater(len(seen_windows[1]), 1)
        self.assertTrue(result["attempted"])
        self.assertFalse(result["skipped"])
        self.assertIn("revised", result["segments"][0].text)
        self.assertIn("synthetic split retry succeeded", " ".join(result["metadata"]["episode_notes"]))

    def test_cleanup_review_uses_minimum_size_failure_reason_at_single_segment(self):
        segments = [make_segment(1, "HOST", "alpha")]

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            raise RuntimeError("Review backend response was truncated before the JSON payload completed.")

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": True,
                    "review_context_budget": 16000,
                },
            )

        self.assertTrue(result["skipped"])
        self.assertEqual(result["skip_reason"], "invalid_response_at_minimum_size")
        self.assertEqual(
            result["metadata"]["review_stage_results"]["transcript_cleanup_review"]["skip_reason"],
            "invalid_response_at_minimum_size",
        )

    def test_calibration_session_shrinks_on_truncation_and_grows_after_stable_successes(self):
        backend_capabilities = resolve_backend_capabilities(
            {
                "runtime_profile": "custom",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
                "review_context_budget": 16000,
                "review_structured_output_support": True,
                "review_transcript_qa_available": True,
                "review_episode_wide_correction_available": True,
                "review_auto_calibrate": True,
                "review_auto_adapt_upward": True,
            }
        )
        session = ReviewCalibrationSession(backend_capabilities)
        original_budget = session.families["local_text_review"]["current_budget"]

        session.note_truncation("local_text_review", failing_estimate=3000)
        shrunken_budget = session.families["local_text_review"]["current_budget"]
        self.assertLess(shrunken_budget, original_budget)
        self.assertEqual(session.families["local_text_review"]["stable_success_count"], 0)

        session.families["local_text_review"]["cooldown_remaining"] = 0
        for _ in range(10):
            session.note_success("local_text_review")
        self.assertGreater(session.families["local_text_review"]["current_budget"], shrunken_budget)

    def test_review_calibration_session_uses_warm_start_without_marking_run_calibrated(self):
        backend_capabilities = resolve_backend_capabilities(
            {
                "runtime_profile": "custom",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
                "review_context_budget": 16000,
                "review_structured_output_support": True,
                "review_transcript_qa_available": True,
                "review_episode_wide_correction_available": True,
            }
        )
        seed_session = ReviewCalibrationSession(backend_capabilities)
        seed_session.families["local_text_review"]["current_budget"] = 2400
        persisted_state = seed_session.serialize()
        session = ReviewCalibrationSession(backend_capabilities, persisted_state)

        self.assertTrue(session.warm_start_used)
        self.assertFalse(session.calibrated)
        self.assertFalse(session.calibrated_this_run)
        self.assertEqual(session.families["local_text_review"]["current_budget"], 2400)

    def test_review_calibration_session_reuses_persisted_completed_calibration(self):
        backend_capabilities = resolve_backend_capabilities(
            {
                "runtime_profile": "custom",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
                "review_context_budget": 16000,
                "review_batch_token_limit": 12000,
                "review_candidate_filter": True,
                "review_structured_output_support": True,
                "review_transcript_qa_available": True,
                "review_episode_wide_correction_available": True,
            }
        )
        seed_session = ReviewCalibrationSession(backend_capabilities)
        seed_session.calibrated = True
        seed_session.families["local_text_review"]["current_budget"] = 2400

        session = ReviewCalibrationSession(backend_capabilities, seed_session.serialize())

        self.assertTrue(session.warm_start_used)
        self.assertTrue(session.calibrated)
        self.assertFalse(session.calibrated_this_run)
        self.assertEqual(session.families["local_text_review"]["current_budget"], 2400)

    def test_long_context_growth_policy_can_grow_conservatively(self):
        backend_capabilities = resolve_backend_capabilities(
            {
                "runtime_profile": "high_context_5090",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
                "review_auto_adapt_upward": True,
                "episode_qa_review": True,
            }
        )
        session = ReviewCalibrationSession(backend_capabilities)
        session.families["long_context_review"]["current_budget"] = 64000
        original_budget = session.families["long_context_review"]["current_budget"]
        session.families["long_context_review"]["cooldown_remaining"] = 0
        for _ in range(20):
            session.note_success("long_context_review")

        self.assertGreater(session.families["long_context_review"]["current_budget"], original_budget)

    def test_noop_truncation_does_not_append_recent_event(self):
        backend_capabilities = resolve_backend_capabilities(
            {
                "runtime_profile": "custom",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
                "review_context_budget": 16000,
                "review_structured_output_support": True,
            }
        )
        session = ReviewCalibrationSession(backend_capabilities)
        session.families["local_text_review"]["current_budget"] = 512
        session.note_truncation("local_text_review", failing_estimate=400)
        self.assertEqual(session.families["local_text_review"]["recent_events"], [])

    def test_backend_identity_enrichment_uses_models_endpoint_when_available(self):
        backend_capabilities = resolve_backend_capabilities(
            {
                "runtime_profile": "high_context_5090",
                "backend": "vllm",
                "review_base_url": "http://127.0.0.1:8000",
                "review_model_name": "qwen-review",
            }
        )

        with patch(
            "podcast_transcribe.review._read_json_url",
            return_value={"data": [{"id": "qwen-review", "max_model_len": 65536}]},
        ):
            enriched = enrich_backend_capabilities_with_identity(backend_capabilities)

        self.assertEqual(enriched["backend_identity_model_id"], "qwen-review")
        self.assertEqual(enriched["backend_identity_reported_max_context"], 65536)

    def test_review_segments_records_calibration_metadata_snapshot(self):
        segments = [make_segment(1, "HOST", "short text")]
        runtime_config = {
            "runtime_profile": "custom",
            "backend": "vllm",
            "review_base_url": "http://127.0.0.1:8000",
            "review_model_name": "qwen-review",
            "transcript_cleanup_review": True,
            "review_context_budget": 16000,
            "review_structured_output_support": True,
            "review_transcript_qa_available": True,
            "review_episode_wide_correction_available": True,
        }
        session = ReviewCalibrationSession(resolve_backend_capabilities(runtime_config))

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            return {
                "reviewed_segments": [],
                "corrected_segment_count": 0,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                runtime_config,
                calibration_session=session,
            )

        self.assertIn("review_calibration", result["metadata"])
        self.assertIn("families", result["metadata"]["review_calibration"])
        self.assertIn("local_text_review", result["metadata"]["review_calibration"]["families"])
        self.assertIn("review_change_summary", result["metadata"])
        self.assertIn("review_stage_value", result["metadata"])
        self.assertIn("transcript_cleanup_review", result["metadata"]["review_stage_value"])

    def test_speaker_consistency_review_can_relabel_speaker(self):
        segments = [make_segment(1, "SPEAKER_01", "hello there")]

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            return {
                "reviewed_segments": [{"id": 1, "text": "hello there", "speaker": "HOST"}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "speaker_consistency_review": True,
                    "review_context_budget": 16000,
                },
            )

        self.assertEqual(result["segments"][0].speaker, "HOST")
        self.assertEqual(
            result["metadata"]["review_stage_results"]["speaker_consistency_review"]["corrected_segment_count"],
            1,
        )

    def test_episode_qa_is_skipped_when_long_context_unavailable(self):
        segments = [make_segment(1, "HOST", "hello there")]

        result = review_segments(
            segments,
            {
                "runtime_profile": "custom",
                "backend": "lm_studio",
                "review_base_url": "http://127.0.0.1:1234",
                "review_model_name": "mistral-review",
                "episode_qa_review": True,
                "review_transcript_qa_available": False,
                "review_episode_wide_correction_available": False,
                "review_context_budget": 16000,
            },
        )

        stage_result = result["metadata"]["review_stage_results"]["episode_qa_review"]
        self.assertEqual(stage_result["status"], "skipped")
        self.assertEqual(stage_result["skip_reason"], "long_context_unavailable")
        self.assertEqual(result["metadata"]["episode_qa_mode"], "skipped")

    def test_episode_qa_full_episode_mode_runs_when_within_budget(self):
        segments = [make_segment(1, "HOST", "short text")]
        seen_modes = []

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            seen_modes.append(stage_mode)
            return {
                "reviewed_segments": [{"id": 1, "text": "short text revised"}],
                "corrected_segment_count": 1,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "high_context_5090",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": False,
                    "glossary_correction_review": False,
                    "speaker_consistency_review": False,
                    "episode_qa_review": True,
                },
            )

        self.assertEqual(seen_modes, ["full_episode"])
        self.assertEqual(result["metadata"]["episode_qa_mode"], "full_episode")

    def test_episode_qa_full_episode_truncation_falls_back_to_chunked(self):
        segments = [make_segment(index, "HOST", "x" * 2000) for index in range(1, 5)]
        seen_modes = []

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            seen_modes.append(stage_mode)
            if stage_mode == "full_episode":
                raise RuntimeError("Review backend response was truncated before the JSON payload completed.")
            return {
                "reviewed_segments": [],
                "corrected_segment_count": 0,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "high_context_5090",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "transcript_cleanup_review": False,
                    "glossary_correction_review": False,
                    "speaker_consistency_review": False,
                    "episode_qa_review": True,
                },
            )

        self.assertEqual(seen_modes[0], "full_episode")
        self.assertIn("chunked", seen_modes[1:])
        self.assertEqual(result["metadata"]["episode_qa_mode"], "chunked")

    def test_episode_qa_chunked_mode_runs_when_over_budget(self):
        segments = [make_segment(index, "HOST", "x" * 40000) for index in range(1, 7)]
        seen_modes = []

        def fake_stage_call(window, backend_capabilities, stage_definition, stage_mode, debug_context=None, preferred_terms=None):
            seen_modes.append(stage_mode)
            return {
                "reviewed_segments": [{"id": segment.id, "text": segment.text} for segment in window],
                "corrected_segment_count": 0,
                "episode_notes": [],
            }

        with patch("podcast_transcribe.review._execute_stage_backend_request", side_effect=fake_stage_call):
            result = review_segments(
                segments,
                {
                    "runtime_profile": "custom",
                    "backend": "vllm",
                    "review_base_url": "http://127.0.0.1:8000",
                    "review_model_name": "qwen-review",
                    "episode_qa_review": True,
                    "review_context_budget": 50000,
                    "review_transcript_qa_available": True,
                    "review_episode_wide_correction_available": True,
                    "review_structured_output_support": True,
                },
            )

        self.assertTrue(all(mode == "chunked" for mode in seen_modes))
        self.assertEqual(result["metadata"]["episode_qa_mode"], "chunked")


if __name__ == "__main__":
    unittest.main()
