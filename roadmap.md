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
- Treat terms listed in `preferred_terms.txt` as reserved spellings for all LLM review stages:
  - do not alter them once they already match the configured preferred glossary
  - do not substitute near-synonyms, alternate spellings, or casing variants unless the glossary explicitly says so
  - do not let episode QA "improve" protected terms away from the configured spelling
- Keep glossary review directional: it may correct toward the configured preferred term, but never away from it.
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

## Calibration And Backend Identity

- Recalibrate review budgets once per processing run instead of treating previous state as already calibrated.
- Reuse saved budgets only as warm-start hints for the current run.
- Include richer backend/model identity in the review calibration fingerprint when the backend exposes it, especially for `vllm`.
- Allow long-context review budgets to adapt upward conservatively in addition to shrinking on truncation.

## Benchmarking

- Add a dedicated review benchmark mode that runs only the tier-2 review pipeline against checked-in cleaned-transcript fixtures.
- Record separate speed, stability, and quality metrics for review-model comparison.
- Emit machine-readable and Markdown benchmark reports for comparing `vllm` and `lm_studio` review models over time.
- Make reserved-term preservation observable in debug and benchmark outputs so preferred-term regressions can be detected quickly.
- Extend benchmark reports with usable-capacity profiling per review stage:
  - cleanup
  - glossary
  - speaker consistency
  - episode QA
- Score review discipline, not just correctness:
  - no-change discipline
  - patch compactness
  - overproduction ratio
  - boundary stability near usable-context limits

## Testing

- Add tests for:
  - baseline output unchanged when profile is unset
  - optional LLM review metadata
  - reviewed transcript schema compatibility
  - deterministic fallback when no LLM backend is available
- Add preferred-term protection checks so:
  - a segment already containing the correct preferred term is left unchanged by cleanup, glossary, and episode-QA review
  - a misspelled alias is corrected to the preferred term
  - a reviewed output that changes a protected preferred term to a different spelling is flagged as a glossary-safety failure
- Add fixtures for:
  - glossary corrections
  - speaker-name drift
  - long-context correction opportunities
  - "already correct preferred term; do not touch" cases

## Implementation Direction

- Pass preferred-term context into the LLM review request rather than relying only on earlier deterministic cleanup.
- Add an explicit prompt invariant that configured preferred terms are reserved and must be preserved exactly when already correct.
- Add a post-review validation/check step that detects when reviewed output changed protected terms away from the configured preferred spelling.
- Feed reserved-term preservation into the review benchmark harness as a glossary-safety metric, not just a general quality note.

## Implementation Phases

1. Add `runtime_profile`, `backend`, and additive reviewed-transcript metadata.
2. Add optional LLM post-processing hooks without changing baseline outputs.
3. Add high-context episode QA and speaker consistency review as opt-in stages.
4. Add schema/version markers so downstream repos can detect reviewed transcripts safely.
5. Add regression tests proving baseline behavior remains unchanged when advanced features are disabled.
