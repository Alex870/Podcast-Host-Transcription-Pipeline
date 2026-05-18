# Roadmap

This roadmap captures practical feature upgrades for `podcast-host-transcription-pipeline`, the first stage of the podcast RAG toolchain.

## Implementation Status

First implementation pass completed:

- Segment-level `transcription_confidence` metadata in JSON outputs.
- Transcript JSON schema/version metadata and validation before writing.
- Per-episode output manifests with source/config fingerprints, timings, output hashes, and summary data.
- Per-episode speaker identity review CSVs for uncertain speaker assignments.
- Configurable cleanup levels: `disabled`, `conservative`, `normal`, and `aggressive`.
- Optional manual correction CSV ingestion through `corrections_dir`.
- Batch Markdown reports at `_batch_report.md`.
- Unit coverage for cleanup levels, JSON schema metadata, confidence flags, speaker review CSVs, manifests, and batch reports.

Second implementation pass completed:

- Executable transcript contract module with schema summary and validation helpers.
- Contract documentation updated for schema version, pipeline metadata, confidence, and content-quality fields.
- Deterministic segment content-quality tagging for sponsor blocks, music/transition text, boilerplate, repetition, and possible silence/non-speech.
- Language/model warning helper for mismatched or low-confidence Whisper language detection.
- Reference-sample quality checks for known speaker clips.
- Speaker aggregate stats, similarity drift alerts, and recurring unnamed-speaker promotion candidates.
- Batch reports now include speaker aggregate and promotion-candidate sections.
- Unit coverage for contract validation, content quality, language warnings, reference sample quality, speaker aggregates, drift detection, and promotion candidates.

Third implementation pass completed:

- Per-episode transcription and diarization artifacts under `_processing_artifacts` with fingerprint validation.
- Intra-episode resume for completed transcription and diarization stages.
- Controls for disabling intermediate resume and preserving artifacts for debugging.
- Child-process timeout guard for isolated batch runs.
- Disk-space preflight and benchmark-plan mode before heavy model loading.
- Audio-duration-aware batch ETA when durations are available.
- Tiny integration-style output fixture test that validates generated transcript JSON against the executable contract.
- Unit coverage for stage artifact persistence, artifact invalidation, artifact cleanup, and integration output generation.

## Highest-Impact Improvements

- Add a richer batch dashboard or summary report that combines processing time, audio duration, GPU memory, transcription confidence, diarization uncertainty, speaker-match confidence, and review-priority score.
- Add segment-level confidence metadata to the JSON output. Downstream RAG tools could then de-prioritize low-confidence transcript spans or surface them as review-needed evidence.
- Add manual correction ingestion. If a user edits review CSVs or corrected transcript files, the pipeline should be able to merge those corrections back into cleaned JSON without rerunning expensive transcription.
- Add a speaker identity review workflow. The pipeline should emit uncertain speaker assignments in a compact review file and accept user-confirmed speaker mappings for future runs.
- Add stronger resumability inside a single episode. If transcription succeeds but diarization or output writing fails, the next run should reuse the completed intermediate artifacts.

## Transcription And Cleanup Quality

- Add configurable cleanup levels: conservative, normal, aggressive, and disabled. Keep the raw transcript available, but make downstream cleaned JSON behavior explicit.
- Add phrase-level glossary support with replacement provenance so downstream users can tell whether text came from Whisper or post-processing.
- Add optional punctuation and casing repair using a local LLM or smaller text model, gated behind a config flag.
- Add detection for repeated sponsor blocks, music breaks, silence, and intro/outro boilerplate so downstream RAG can optionally exclude or tag them.
- Add language and model auto-detection warnings when the selected Whisper language/model appears mismatched to the audio.

## Speaker Identification

- Add a reference-sample quality checker that scores sample length, silence, clipping, overlap risk, and background noise before using it for speaker matching.
- Add support for multiple host profiles or co-host profiles with stable names across episodes.
- Add speaker-cluster drift detection across a batch, warning when a known speaker match suddenly falls below normal similarity.
- Add an interactive workflow to promote a recurring `SPEAKER_XX` into a named known speaker after reviewing several episodes.
- Add per-speaker aggregate statistics: total speaking time, episode appearances, match confidence distribution, and uncertain segments.

## Pipeline Integration

- Formalize the transcript JSON schema used by `Podcast-RAG-pipeline` and validate every generated JSON file before marking an episode complete.
- Include a pipeline/version field in generated metadata so downstream tools can reason about compatibility.
- Add output manifests per episode with source audio fingerprint, config fingerprint, model versions, output file list, and elapsed timings.
- Add optional direct handoff metadata for `Podcast-RAG-pipeline`, such as recommended input file path, episode date, host speaker name, and reviewed/corrected status.

## Performance And Operations

- Add automatic benchmarking for Whisper model, compute type, batch size, and device choice on a short sample file.
- Add configurable child-process memory and timeout guards, with clearer recovery messages when native ML libraries hang or leak memory.
- Add live ETA at both file and batch level, including audio-hours processed per wall-clock hour.
- Add disk-space preflight checks for large batches and temporary normalized audio.
- Add optional archival of intermediate normalized audio and diarization artifacts for debugging hard episodes.

## Testing And Quality

- Expand unit tests around cleanup, config loading, output schema, state transitions, and speaker matching.
- Add synthetic diarization fixtures to test edge cases: overlapping speakers, very short guest segments, host absent, multiple recurring guests, and noisy audio.
- Add integration tests that run on a tiny audio fixture and assert that all expected output files and metadata fields are produced.
- Add a contract test shared with downstream repos so transcript schema changes are caught before they break RAG preprocessing.
