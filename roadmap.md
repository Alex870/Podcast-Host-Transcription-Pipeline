# Roadmap

This roadmap defines how `podcast-host-transcription-pipeline` should stay compatible with smaller-GPU users while adding optional quality-focused post-processing paths for stronger local servers such as a `5090 + vLLM` setup.

## Compatibility Principles

- Keep the current transcription and diarization workflow as the default baseline.
- Do not require a large-context LLM for normal transcript generation.
- Add post-processing and QA as optional layers, not mandatory stages.
- Preserve raw transcript outputs as the source of truth.
- Emit corrected and reviewed variants additively.

## Shared Runtime Profile Model

- Add `runtime_profile` with values:
  - `baseline_16gb`
  - `high_context_5090`
  - `custom`
- Add `backend` with values such as `lm_studio` and `vllm`.
- Resolve profile settings into:
  - post-processing model
  - max context budget
  - structured output support
  - transcript QA availability
  - episode-wide correction-pass availability

## Architecture Direction

- Keep Whisper, diarization, and speaker identification as the baseline path.
- Add optional LLM post-processing stages:
  - transcript cleanup
  - glossary correction review
  - speaker consistency review
  - episode-wide transcript QA
- Keep these stages disabled by default for compatibility.
- Preserve original transcript text and generate reviewed variants separately.

## Quality Improvements

- Add a high-context transcript QA pass that can review long spans of transcript in one shot.
- Add optional episode-level correction review:
  - repeated terms
  - likely speaker-name drift
  - inconsistent capitalization or glossary choices
  - suspected local misrecognitions that only become obvious with larger context
- Add explicit provenance metadata:
  - `original_text`
  - `llm_reviewed_text`
  - `review_runtime_profile`
  - `review_backend`
  - `review_model_name`

## Output And Contract

- Keep current output formats valid for downstream repos.
- Add optional metadata fields rather than changing existing required fields.
- Version any reviewed transcript schema separately from the raw transcript schema.
- Record whether a transcript was:
  - raw only
  - cleaned deterministically
  - LLM reviewed
  - LLM reviewed with high-context profile

## Performance Strategy

- Do not enable episode-wide QA by default.
- Gate long-context review and richer correction passes behind `high_context_5090` or explicit flags.
- Keep batch-friendly deterministic cleanup available for all hardware.

## Testing

- Add tests for:
  - baseline output unchanged when profile is unset
  - optional LLM review metadata
  - reviewed transcript schema compatibility
  - deterministic fallback when no LLM backend is available
- Add fixtures for:
  - glossary corrections
  - speaker-name drift
  - long-context correction opportunities

## Implementation Phases

1. Add `runtime_profile`, `backend`, and additive reviewed-transcript metadata.
2. Add optional LLM post-processing hooks without changing baseline outputs.
3. Add high-context episode QA and speaker consistency review as opt-in stages.
4. Add schema/version markers so downstream repos can detect reviewed transcripts safely.
5. Add regression tests proving baseline behavior remains unchanged when advanced features are disabled.
