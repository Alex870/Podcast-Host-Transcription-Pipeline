# Podcast Host Transcription Pipeline

This project batch-processes podcast audio into speaker-labeled transcripts, host-only extracts, structured JSON, QA outputs, and optional reviewed transcript variants for downstream RAG and vector-database workflows.

It is built for shows where speaker identity matters, especially when you want something more useful than plain ASR text:

- host detection from reference clips or a persistent host profile
- recurring speaker labeling from known samples
- deterministic cleanup and glossary normalization
- additive LLM review through a local or LAN OpenAI-compatible backend
- a local browser review workbench for processed transcript inspection and operator write-back
- outputs shaped for the rest of the podcast toolchain

The shared transcript, processed-cache, Chroma metadata, and `podcast.json` expectations across the podcast stack are documented in [`docs/podcast_pipeline_contract.md`](docs/podcast_pipeline_contract.md).

## Clean-machine packaging

The baseline Python dependencies are pinned in `podcast_transcribe_requirements.txt`. The optional WhisperX alignment environment is separately pinned in `podcast_transcribe_alignment_requirements.txt`. The browser workbench uses the checked-in `workbench-ui/pnpm-lock.yaml` and `pnpm-workspace.yaml`; launcher option 5 installs and rebuilds it automatically when needed. The root ecosystem clean-machine diagnostic records CUDA, FFmpeg, TorchCodec, and Hugging Face checks without bundling models or credentials.

## Why This Exists

Basic podcast transcribers usually stop at "faithfully capture the words." That is not enough if the transcript is going to feed a semantic pipeline, vector database, or host-centric chat system. This repository is meant to produce transcript artifacts that are structured, reviewable, and operationally useful beyond simple text search.

## Repository Contents

- `Run Podcast Transcribe.ps1`: root bootstrap launcher
- `scripts/Convert-AudioToDiarizedText.ps1`: main PowerShell runner
- `scripts/Debug-PodcastTranscribeEnvironment.ps1`: environment validation
- `scripts/Migrate-LegacyPodcastTranscribeState.ps1`: migrate config, state, and outputs from an older working directory
- `scripts/Launch-PodcastTranscribeWorkbench.ps1`: local transcript review workbench launcher
- `scripts/Configure-PodcastTranscribeReviewBackend.ps1`: external review-LLM discovery and configuration wizard
- `src/podcast_transcribe/`: Python package for the pipeline
- `workbench-ui/`: React + Vite frontend for the transcript review workbench
- `examples/`: example config, glossary, and replacement files
- `docs/`: Quick Start, user docs, config reference, architecture, and contract docs
- `benchmarks/review_fixtures/`: checked-in cleaned-transcript fixtures for review benchmarking
- `benchmarks/pipeline_gold_set/`: synthetic/template quality fixtures; real private evaluation packs are configured externally

## Launcher Menu

Start here for normal Windows use:

```powershell
.\Run Podcast Transcribe.ps1
```

Current bootstrap options:

1. Run environment validation
2. Run the transcription pipeline
3. Migrate settings and state from a legacy directory
4. Run review benchmark
5. Launch transcript review workbench
6. Run pipeline quality benchmark
7. Configure external review LLM
8. Download pinned transcription models
9. Transcribe committee meeting (anonymous speakers)

All PowerShell entrypoints pause at the end so the console window stays open long enough to read the result.
After an interactive menu action finishes, the bootstrap returns to the main menu; only `Q` closes it. Explicit `-Action` invocations remain one-shot.
Option `5` will also install workbench frontend dependencies and rebuild the bundled UI automatically when needed.
Option `7` discovers models exposed by a vLLM or LM Studio server, tests the selected model, previews the exact config changes, and safely updates the project config.
Option `8` explicitly downloads the selected revision-pinned provider artifacts after preflight; normal processing never downloads models implicitly.
Option `9` runs the `anonymous_meeting` workflow for recordings where diarization labels are useful but host and recurring-speaker identity are not required.

## Outputs

Per episode, the pipeline can emit:

- baseline transcript files: `*_speaker_transcript.txt`, `*_host_only.txt`, `*_speaker_transcript.json`
- cleaned transcript files: `*_cleaned_speaker_transcript.txt`, `*_cleaned_host_only.txt`, `*_cleaned_speaker_transcript.json`
- review and QA files: `*_review.csv`, `*_speaker_identity_review.csv`
- optional reviewed files: `*_reviewed_speaker_transcript.txt`, `*_reviewed_host_only.txt`, `*_reviewed_speaker_transcript.json`
- human-corrected siblings: `*_corrected_speaker_transcript.txt`, `*_corrected_host_only.txt`, `*_corrected_speaker_transcript.json`
- provenance and reporting: `*_manifest.json`

Batch-level outputs include `_episode_review_summary.csv`, `_batch_report.md`, `_review_run_report.*`, and `_speaker_workflow_report.*`.

## Getting Started

For the practical first-run path, start with [`docs/quick-start.md`](docs/quick-start.md).

That guide covers:

- prerequisites
- creating the Python environment
- Hugging Face and FFmpeg setup
- creating `podcast_transcribe_config.json`
- setting up `preferred_terms.txt` and `preferred_replacements.json`
- first validation and first transcription run

## Documentation

Operator-facing docs:

- [`docs/quick-start.md`](docs/quick-start.md)
- [`docs/user-manual.md`](docs/user-manual.md)
- [`docs/config-reference.md`](docs/config-reference.md)

Engineering/supporting docs:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/podcast_pipeline_contract.md`](docs/podcast_pipeline_contract.md)
- [`docs/state-of-the-art-comparison.md`](docs/state-of-the-art-comparison.md)
- [`roadmap.md`](roadmap.md)
- [`docs/workbench-implementation-plan.md`](docs/workbench-implementation-plan.md)

## High-Level Workflow

At a high level, each episode goes through:

1. transcription with the configured ASR provider
2. word alignment, using timestamp passthrough by default
3. diarization with `pyannote.audio`
4. speaker matching with the configured embedding provider
5. deterministic cleanup and glossary normalization
6. optional additive LLM review through a local or LAN OpenAI-compatible backend
7. output writing, manifests, and batch summaries

Expensive stages now record provider-aware fingerprints. The default provider set preserves the established behavior, while optional forced alignment and future model adapters can be evaluated without blindly rerunning every independent stage.

The stable baseline keeps `faster_whisper` and `speechbrain_ecapa` as the defaults. `--asr-provider parakeet` is an optional lazy NeMo/Parakeet experiment with explicit Windows/CUDA diagnostics; `--alignment-provider whisperx` enables forced alignment; and `--speaker-embedding-provider speechbrain_xvector` is a candidate family. Candidate reports must pass the gold-set promotion guardrails before a profile or default is changed.

Milestone 4 makes model acquisition explicit. Pin `--model-revision`, `--diarization-model-revision`, and `--speaker-model-revision` (plus `--alignment-model` and `--alignment-model-revision` for WhisperX), run `--provider-preflight`, then invoke `--download-provider-models` as a separate authenticated action. Normal processing uses only revision-matched artifacts under `--provider-cache-dir`; it never downloads models implicitly. `--batch-size 0` selects a conservative adaptive batch, and `--device auto` records CUDA selection or CPU fallback diagnostics.

Use `--pipeline-benchmark --speech-run-id <id>` to publish an immutable shadow run. Promotion additionally requires an approved pack with exact source identity, target condition slices, and a deployment-machine profile containing measured runtime, peak-memory, and storage limits.

The optional review layer can also backfill reviewed outputs from existing `*_cleaned_speaker_transcript.json` files, so legacy tier-1 work does not need to be rerun just to add tier-2 review artifacts.

With `resume_intermediates` enabled, the pipeline retains fingerprinted stage artifacts and the seek-friendly `speaker_audio_16k_mono.wav` cache under each episode's `_processing_artifacts` directory. Repeated runs reuse these artifacts when the source and relevant settings still match. Console timing output also identifies slow review components and finalization operations without printing every short operation.

For non-podcast recordings such as committee meetings, choose menu option 9, `Transcribe committee meeting (anonymous speakers)`, or pass `--workflow-profile anonymous_meeting`. This profile keeps pyannote diarization labels so speakers remain separable, but does not load reference clips, compute speaker embeddings, infer a host, update `host_profile.json`, or contact an LLM review backend. It is safe to use with a source folder containing no speaker reference audio.

New runs write `episode-contract-v2`. Legacy v1 bundles are promoted on the next normal pass using output-only or cached reconstruction where possible, with automatic tier-1 reprocessing only when required evidence is unavailable. The workbench writes `correction-manifest-v2` histories and builds recurring-speaker candidates from compatible embeddings rather than reusable diarization labels.

The validated Windows GPU environment uses PyTorch 2.9 with TorchCodec 0.8.1 and a shared FFmpeg 7 build. Keep that build in a dedicated directory such as `C:\ffmpeg7\bin`; the launcher preloads its DLLs so another FFmpeg installation on `PATH` cannot interfere. When the native decoder is available, pyannote can receive path-based audio; learned long-file routing still provides a chunked fallback for global diarization memory failures.

## Supported Audio Formats

- `.mp3`
- `.wav`
- `.m4a`
- `.flac`
- `.ogg`
