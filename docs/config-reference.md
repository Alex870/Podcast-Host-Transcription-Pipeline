# Config Reference

This document is the authoritative reference for `podcast_transcribe_config.json`.

All keys are optional unless otherwise noted. Relative paths are resolved from the repository root unless a script or CLI override provides a different explicit path.

## Input and Output Paths

### `default_source_dir`

- Type: `string`
- Example: `"source"` or `"D:/Speech_to_text/audio"`
- Default: empty / launcher prompt when not set
- Affects: baseline transcription, migration

Preferred starting folder for the launcher. If valid, the launcher uses it directly instead of prompting.

### `known_speakers_dir`

- Type: `string`
- Default: `"speaker_reference_samples"`
- Affects: speaker identification

Directory containing known-speaker reference clips and `speakers.json`.

### `host_profile_json`

- Type: `string`
- Default: `"host_profile.json"`
- Affects: speaker identification

Persistent host voice profile updated over time to improve host matching across episodes.

### `host_reference`

- Type: `string`
- Default: `""`
- Affects: speaker identification, stage-cache compatibility

Optional clean host-reference audio file. Relative paths are resolved from the repository root.

### `corrections_dir`

- Type: `string`
- Default: `""` in config, with launcher/workbench fallback to `corrections`
- Affects: cleanup/output generation

Optional directory containing manual correction CSVs named like `<audio_stem>_corrections.csv`.

### `review_debug_dir`

- Type: `string`
- Default: `""`
- Affects: review debugging

Optional override directory for review debug artifacts. When blank and `review_debug` is enabled, the pipeline uses:

```text
<output>\_processing_artifacts\<episode>\review_debug
```

## Hugging Face and FFmpeg

The current validated Windows GPU environment uses PyTorch 2.9.0 with CUDA 12.8, TorchAudio 2.9.0, TorchVision 0.24.0, and TorchCodec 0.8.1. Install those versions through `podcast_transcribe_requirements.txt`; they are environment dependencies rather than config keys. A shared FFmpeg build is still required for TorchCodec path-based audio decoding.

### `hf_token`

- Type: `string`
- Default: `""`
- Affects: baseline transcription

Fallback Hugging Face token if `HF_TOKEN` is not already available through the environment or `.env`.

### `ffmpeg_bin_dir`

- Type: `string`
- Default: `""`
- Affects: baseline transcription

Directory containing the FFmpeg Windows binaries/DLLs.

## Whisper and Baseline Transcription

### `asr_provider`

- Type: `string`
- Allowed values: `faster_whisper`
- Default: `"faster_whisper"`
- Affects: baseline transcription, stage-cache compatibility

Selects the ASR provider contract. The initial provider release wraps the established faster-whisper path; candidate ASR providers remain gated on pipeline benchmark results.

### `model`

- Type: `string`
- Default: `"distil-large-v3"` in the example config
- Affects: baseline transcription

Whisper model name passed to `faster-whisper`.

### `language`

- Type: `string`
- Default: `"en"`
- Affects: baseline transcription

Language code passed to Whisper.

### `device`

- Type: `string`
- Allowed values: typically `auto`, `cpu`, `cuda`
- Default: `"auto"`
- Affects: baseline transcription

Runtime device selection for Whisper.

### `compute_type`

- Type: `string`
- Default: `"auto"` in config, `float16` may still be forced by some CLI defaults
- Affects: baseline transcription

`faster-whisper` compute type.

### `beam_size`

- Type: `integer`
- Default: `5`
- Affects: baseline transcription

Decode beam size.

### `batch_size`

- Type: `integer`
- Default: `8`
- Affects: baseline transcription

Whisper transcription batch size.

### `diarization_model`

- Type: `string`
- Default: `"pyannote/speaker-diarization-community-1"`
- Affects: diarization, stage-cache compatibility

Hugging Face model id for the pyannote diarization pipeline.

## Word Alignment

### `alignment_provider`

- Type: `string`
- Allowed values: `timestamp_passthrough`, `whisperx`
- Default: `"timestamp_passthrough"`
- Affects: word timing, speaker attribution, stage-cache compatibility

`timestamp_passthrough` preserves the current faster-whisper word timestamps. `whisperx` enables optional forced alignment and requires the separate optional alignment dependency environment.

### `alignment_model`

- Type: `string`
- Default: `""`
- Affects: forced alignment

Optional WhisperX alignment model override. Blank uses the language-specific provider default.

### `isolate_files`

- Type: `boolean`
- Default: `true`
- Affects: operations/performance

When `true`, the launcher prefers isolated child-process handling so native memory is released between episodes.

### `benchmark_only`

- Type: `boolean`
- Default: `false`
- Affects: operations

Preflight/dry benchmark-plan behavior for the baseline runner. This is separate from the dedicated review benchmark mode.

## Speaker Identification

### `speaker_embedding_provider`

- Type: `string`
- Allowed values: `speechbrain_ecapa`
- Default: `"speechbrain_ecapa"`
- Affects: host profiles, known-speaker matching, stage provenance

Speaker profiles are versioned with this provider and the configured `speaker_model`. Incompatible embedding families are not mixed.

### `speaker_model`

- Type: `string`
- Default: `"speechbrain/spkrec-ecapa-voxceleb"`
- Affects: speaker identification, profile compatibility, stage-cache compatibility

Speaker-verification model used by the active embedding provider.

### `min_host_seconds`

- Type: `number`
- Default: `20`
- Affects: host-profile updates

Minimum diarized duration required before a speaker can update the host profile.

### `max_embedding_seconds`

- Type: `number`
- Default: `90`
- Affects: speaker embeddings, stage-cache compatibility

Maximum speech duration sampled per speaker when building an embedding.

### `num_speakers`

- Type: `integer` or `null`
- Default: omitted
- Affects: diarization, stage-cache compatibility

Optional fixed speaker count passed to pyannote. Omit it when the speaker count is unknown.

### `assume_dominant_speaker_is_host`

- Type: `boolean`
- Default: `true`
- Affects: speaker identification

Fallback bootstrap behavior when the host cannot be identified from better evidence.

### `host_threshold`

- Type: `number`
- Default: `0.45`
- Affects: speaker identification

Similarity threshold used for host and known-speaker matching.

## Cleanup and Glossary

### `preferred_terms_file`

- Type: `string`
- Default: `"examples/preferred_terms.txt"`
- Affects: baseline cleanup, review protection, benchmarking

Glossary file used for transcription biasing and reserved preferred spellings.

### `replacement_map_json`

- Type: `string`
- Default: `"examples/preferred_replacements.json"`
- Affects: baseline cleanup

JSON file containing alias-to-preferred replacements applied after transcription.

### `cleanup_level`

- Type: `string`
- Allowed values: `disabled`, `conservative`, `normal`, `aggressive`
- Default: `"normal"`
- Affects: cleanup

Controls deterministic cleanup strength.

- `normal` is the default balanced mode.
- `aggressive` enables stronger bounded restart-pruning such as removing short filler fragments and abandoned repeated starts.

### `preferred_terms`

- Type: `array[string]`
- Default: `[]`
- Affects: review

Advanced inline alternative to a file-based glossary for review/runtime resolution. Normally the file-based path is simpler and preferred.
The launcher passes each item as a protected preferred term and merges it with `preferred_terms_file`.

## Runtime Profile and Review Backend

### `runtime_profile`

- Type: `string`
- Allowed values: `baseline_16gb`, `high_context_5090`, `custom`
- Default: `"baseline_16gb"`
- Affects: review

Controls the default review capability profile. It does not change baseline Whisper, diarization, or speaker-identification behavior.

Profile behavior:

- `baseline_16gb`: conservative defaults, no review stages enabled by default
- `high_context_5090`: strong-review profile, review stages enabled by default
- `custom`: uses the advanced capability keys described below

### `backend`

- Type: `string`
- Allowed values: `none`, `lm_studio`, `vllm`
- Default: `"none"`
- Affects: review

Configures the optional local review backend.

### `review_base_url`

- Type: `string`
- Default: `""`
- Affects: review, benchmarking

OpenAI-compatible base URL for LM Studio or vLLM.

### `review_model_name`

- Type: `string`
- Default: `""`
- Affects: review, benchmarking

Model identifier exposed by the configured review backend.

### `transcript_cleanup_review`

- Type: `boolean`
- Default: profile-driven
- Affects: review

Enables the cleanup-review stage.
An explicit `false` overrides stages implied by the selected runtime profile.

### `glossary_correction_review`

- Type: `boolean`
- Default: profile-driven
- Affects: review

Enables the glossary-review stage.
An explicit `false` overrides stages implied by the selected runtime profile.

### `speaker_consistency_review`

- Type: `boolean`
- Default: profile-driven
- Affects: review

Enables the speaker-consistency stage.
An explicit `false` overrides stages implied by the selected runtime profile.

### `episode_qa_review`

- Type: `boolean`
- Default: profile-driven
- Affects: review

Enables the episode-QA stage.
An explicit `false` overrides stages implied by the selected runtime profile.

## Review Debugging and Adaptation

### `review_debug`

- Type: `boolean`
- Default: `false`
- Affects: review debugging

When `true`, writes prompt/response debug artifacts for staged review.

### `review_auto_calibrate`

- Type: `boolean`
- Default: effectively `true` when review is enabled, otherwise `false`
- Affects: review

Enables run-scoped calibration of safe review budgets on the first reviewable episode.

### `review_auto_adapt_upward`

- Type: `boolean`
- Default: `true`
- Affects: review

Allows conservative upward adaptation after a strong stability streak. Downward adaptation on truncation remains important even when this is disabled.

## Advanced Custom-Profile Review Controls

These keys are mainly relevant when `runtime_profile = "custom"`.

### `review_context_budget`

- Type: `integer`
- Default: `32768` for `custom`
- Affects: review

Maximum review context budget used by the custom profile.

### `review_structured_output_support`

- Type: `boolean`
- Default: `false` for `custom`
- Affects: review

Declares whether the custom backend/profile should be treated as supporting structured output well enough for the review contract.

### `review_transcript_qa_available`

- Type: `boolean`
- Default: `false` for `custom`
- Affects: review

Declares whether transcript QA behavior is available for the custom profile.

### `review_episode_wide_correction_available`

- Type: `boolean`
- Default: `false` for `custom`
- Affects: review

Declares whether episode-wide correction review should be considered available under the custom profile.

## Filename Date Parsing

### `filename_date`

- Type: `object`
- Default:

```json
{
  "preset": "strict_iso",
  "position": "last"
}
```

- Affects: metadata, downstream RAG chronology

Controls how episode dates are parsed from filenames.

Supported child keys:

#### `filename_date.preset`

- Type: `string`
- Common values: `strict_iso`, `american_podcast`, `mixed_common`
- Default: `"strict_iso"`

Selects a built-in date parser preset.

#### `filename_date.position`

- Type: `string`
- Allowed values: `first`, `last`
- Default: `"last"`

Chooses which valid filename match wins when multiple dates are present.

#### `filename_date.formats`

- Type: `array[string]`
- Default: omitted

Optional ordered explicit format list that overrides the preset behavior for disambiguation.

## Operational Controls

### `resume_intermediates`

- Type: `boolean`
- Default: `true`
- Affects: operations

Allows reuse of `_processing_artifacts` when a prior run completed expensive intermediate stages.

### `archive_debug_artifacts`

- Type: `boolean`
- Default: `false`
- Affects: operations/debugging

Preserves review/debug material after successful processing. Reusable stage caches are governed separately by `resume_intermediates`.

### `child_timeout_seconds`

- Type: `integer`
- Default: `0`
- Affects: operations

Optional timeout guard for isolated child workers. `0` disables the timeout.

## Notes on Defaults

- Example-config defaults are intentionally conservative for compatibility.
- Review-profile defaults can imply review-stage booleans even when you do not spell them out explicitly.
- If `backend = "none"`, review stages are effectively disabled even if their booleans are set to `true`.

## Related Docs

- setup walkthrough: [`quick-start.md`](quick-start.md)
- operational guide: [`user-manual.md`](user-manual.md)
- architecture: [`architecture.md`](architecture.md)
