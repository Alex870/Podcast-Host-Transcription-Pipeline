# Podcast Host Transcription Pipeline

This project batch-processes podcast audio into speaker-labeled transcripts, host-only extracts, structured JSON, QA outputs, and optional reviewed transcript variants for downstream RAG and vector-database workflows.

It is built for shows where speaker identity matters, especially when you want something more useful than plain ASR text:

- host detection from reference clips or a persistent host profile
- recurring speaker labeling from known samples
- deterministic cleanup and glossary normalization
- additive local-LLM review for stronger machines
- a local browser review workbench for processed transcript inspection and operator write-back
- outputs shaped for the rest of the podcast toolchain

The shared transcript, processed-cache, Chroma metadata, and `podcast.json` expectations across the podcast stack are documented in [`docs/podcast_pipeline_contract.md`](docs/podcast_pipeline_contract.md).

## Clean-machine packaging

The baseline Python dependencies are pinned in `podcast_transcribe_requirements.txt`. The optional WhisperX alignment environment is separately pinned in `podcast_transcribe_alignment_requirements.txt`. The browser workbench uses the checked-in `workbench-ui/pnpm-lock.yaml` and `pnpm-workspace.yaml`; run `pnpm install --frozen-lockfile` followed by `pnpm run build`. The root ecosystem clean-machine diagnostic records CUDA, FFmpeg, TorchCodec, and Hugging Face checks without bundling models or credentials.

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
- `benchmarks/pipeline_gold_set/`: human-approved reference spans for full-pipeline quality measurement

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

All PowerShell entrypoints pause at the end so the console window stays open long enough to read the result.
After an interactive menu action finishes, the bootstrap returns to the main menu; only `Q` closes it. Explicit `-Action` invocations remain one-shot.
Option `5` will also install workbench frontend dependencies and rebuild the bundled UI automatically when needed.
Option `7` discovers models exposed by a vLLM or LM Studio server, tests the selected model, previews the exact config changes, and safely updates the project config.

## Outputs

Per episode, the pipeline can emit:

- baseline transcript files: `*_speaker_transcript.txt`, `*_host_only.txt`, `*_speaker_transcript.json`
- cleaned transcript files: `*_cleaned_speaker_transcript.txt`, `*_cleaned_host_only.txt`, `*_cleaned_speaker_transcript.json`
- review and QA files: `*_review.csv`, `*_speaker_identity_review.csv`
- optional reviewed files: `*_reviewed_speaker_transcript.txt`, `*_reviewed_host_only.txt`, `*_reviewed_speaker_transcript.json`
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
6. optional additive local-LLM review
7. output writing, manifests, and batch summaries

Expensive stages now record provider-aware fingerprints. The default provider set preserves the established behavior, while optional forced alignment and future model adapters can be evaluated without blindly rerunning every independent stage.

Stage 7 keeps `faster_whisper` and `speechbrain_ecapa` as the defaults. `--asr-provider parakeet` is an optional lazy NeMo/Parakeet experiment with explicit Windows/CUDA diagnostics; `--alignment-provider whisperx` enables forced alignment; and `--speaker-embedding-provider speechbrain_xvector` is a candidate family. Candidate reports must pass the gold-set promotion guardrails before a profile or default is changed.

The optional review layer can also backfill reviewed outputs from existing `*_cleaned_speaker_transcript.json` files, so legacy tier-1 work does not need to be rerun just to add tier-2 review artifacts.

The validated Windows GPU environment uses PyTorch 2.9 with TorchCodec 0.8.1 and a shared FFmpeg build. When the native decoder is available, pyannote can receive path-based audio; learned long-file routing still provides a chunked fallback for global diarization memory failures.

## Supported Audio Formats

- `.mp3`
- `.wav`
- `.m4a`
- `.flac`
- `.ogg`
