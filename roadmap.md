# Roadmap

This roadmap reflects the current state of `podcast-host-transcription-pipeline` as it exists now: a stable baseline transcription system with an optional, increasingly capable tier-2 review layer for stronger local review backends such as `vLLM` and `LM Studio`.

The project is no longer at the stage of "add optional review at all." That foundation is already in place. The roadmap now centers on refining review quality, model evaluation, and operational clarity while preserving a conservative baseline path for smaller machines.

## Guiding Principles

- Keep Whisper + diarization + speaker identification as the default baseline workflow.
- Keep optional review additive and opt-in.
- Preserve raw and deterministically cleaned outputs as source-of-truth artifacts.
- Let stronger local rigs do more without forcing that complexity onto smaller-GPU users.
- Prefer trustworthy, measurable review behavior over flashy but brittle long-context claims.

## Completed

These are the major roadmap items that are effectively implemented.

### Baseline Pipeline Foundation

- Batch transcription with `faster-whisper`
- Speaker diarization with `pyannote.audio`
- Speaker matching with `speechbrain`
- Host matching from reference samples and persistent profile data
- Known-speaker reference support through `speaker_reference_samples`
- Batch-friendly isolated child-process workflow

### Output Contracts and Provenance

- Transcript JSON schema/version metadata and validation
- Executable transcript contract helpers
- Per-episode manifests with source/config fingerprints, timings, and output hashes
- Batch-level reporting through `_episode_review_summary.csv` and `_batch_report.md`
- Episode date extraction with configurable filename-date parsing

### Deterministic Cleanup and QA

- Cleanup levels: `disabled`, `conservative`, `normal`, `aggressive`
- Deterministic cleanup provenance in cleaned JSON
- Manual correction CSV ingestion through `corrections_dir`
- Deterministic `content_quality` tagging for sponsor/boilerplate/music/silence-like spans
- Reference-sample quality checks for known speakers
- Speaker-review CSVs and batch-level speaker aggregates

### Resume and Operational Safety

- `_processing_artifacts` for transcription/diarization resume
- Intra-episode resume on partial failure
- Debug artifact preservation controls
- Child timeout support for isolated workers
- Disk-space preflight
- Audio-duration-aware progress and ETA reporting

### Optional Tier-2 Review Layer

- `runtime_profile` model:
  - `baseline_16gb`
  - `high_context_5090`
  - `custom`
- `backend` model:
  - `none`
  - `lm_studio`
  - `vllm`
- Additive reviewed transcript outputs:
  - `*_reviewed_speaker_transcript.txt`
  - `*_reviewed_host_only.txt`
  - `*_reviewed_speaker_transcript.json`
- Reviewed-schema metadata and stage provenance
- Staged review pipeline:
  - transcript cleanup review
  - glossary correction review
  - speaker consistency review
  - episode QA review

### Smart Tier-2 Backfill

- Mixed-batch behavior where:
  - new episodes can run `tier1+tier2`
  - legacy tier-1-complete episodes can run `tier2-only backfill`
  - already-complete episodes can be skipped
- Review backfill from `*_cleaned_speaker_transcript.json`
- Stage-aware reviewed-output classification

### Review Calibration and Benchmarking

- Run-scoped review calibration
- Warm-start reuse as hints only
- Adaptive review budgets with downward and conservative upward adjustment
- Backend/model-aware runtime fingerprinting
- Dedicated review benchmark mode against checked-in cleaned-transcript fixtures
- Benchmark scoring for:
  - speed
  - stability
  - quality
  - usable capacity per review stage

### Preferred-Term Protection

- `preferred_terms.txt` treated as reserved spellings for review
- Prompt-level glossary invariants
- Post-review protected-term checks
- Benchmark visibility for glossary safety / protected-term regressions

## Active Work

These are the areas that best describe the current living roadmap.

### 1. Review Quality Hardening

The next big challenge is not adding more review stages. It is making the current stages more useful on real material.

Current focus:

- improve speaker-consistency correction reliability
- improve episode-QA usefulness on long real-world episodes
- reduce missed edits without encouraging over-editing
- keep patch behavior compact and conservative

### 2. Better Model Selection and Evaluation

The benchmark harness exists now, so the opportunity is to use it more intentionally.

Current focus:

- compare review models on practical structured-review behavior, not just published context sizes
- keep evaluating speed vs quality tradeoffs across local vLLM candidates
- make production model choice more evidence-based

### 3. Operational Clarity

The operator experience is much better than it was, but there is still room to make runs easier to interpret.

Current focus:

- clearer summaries of what review actually changed
- better visibility into whether review meaningfully helped on a given run
- continued improvement to console/report language so batch progress and review status are easy to read

## Next Likely Phases

These are the most plausible next roadmap steps from here.

### Phase 1: Review Precision and Utility

- strengthen prompts and reconciliation logic for speaker-consistency review
- improve long-context episode-QA behavior on genuinely long transcripts
- add more real-world fixtures that stress:
  - speaker drift
  - glossary pressure
  - long-context contradictions
  - false-positive over-edit temptation

### Phase 2: Model Benchmarking Maturity

- turn benchmark results into a clearer production recommendation workflow
- add more explicit reporting around:
  - quality-per-second
  - correction yield
  - stage usefulness by model
- continue capacity profiling so model swaps are less guessy

### Phase 3: Richer Run Reporting

- add stronger run-level summaries or dashboards combining:
  - processing time
  - audio duration
  - review changes
  - confidence/risk indicators
  - speaker-match uncertainty
- make it easier to see which episodes most deserve human review

### Phase 4: Speaker Workflow Expansion

- stronger cross-episode speaker drift detection
- better recurring unnamed-speaker promotion workflow
- possible multi-host / co-host profile support

### Phase 5: Broader Integration and Test Depth

- expand fixture and integration coverage further
- keep transcript-contract compatibility tight with downstream repos
- add more shared validation assumptions across the podcast toolchain

## Deprioritized or Reframed Work

These are ideas that still matter, but no longer define the center of the roadmap.

- "Add optional review" is no longer a future item; it is implemented.
- "Add benchmarking" is no longer a future item; the benchmark now exists and the real work is improving how it is used.
- "Add cleanup levels" is done; the active question is how far deterministic cleanup should go while remaining trustworthy.

## Summary

The project now has a strong baseline and a real optional tier-2 review system. The roadmap from here is mostly about refinement:

- make review quality better
- make model choice more evidence-based
- make runs easier to understand
- preserve compatibility for smaller and simpler setups
