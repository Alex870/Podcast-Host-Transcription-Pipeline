# Roadmap

This roadmap reflects the current state of `podcast-host-transcription-pipeline` as it exists now: a stable baseline transcription system with an optional, increasingly capable tier-2 review layer, plus a new local transcript review workbench for inspecting processed outputs and feeding approved fixes back into the pipeline.

The project is no longer at the stage of "add optional review at all." That foundation is already in place. The roadmap now has two connected tracks:

- continue hardening review quality, benchmarking, and operational clarity in the pipeline itself
- grow the workbench into a practical operator console for episode review, issue triage, and reproducible write-back

## Guiding Principles

- Keep Whisper + diarization + speaker identification as the default baseline workflow.
- Keep optional review additive and opt-in.
- Preserve raw and deterministically cleaned outputs as source-of-truth artifacts.
- Let stronger local rigs do more without forcing that complexity onto smaller-GPU users.
- Prefer trustworthy, measurable review behavior over flashy but brittle long-context claims.
- Keep the workbench local-first, operator-focused, and tied to reproducible pipeline inputs rather than direct transcript mutation.

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

### Transcript Review Workbench Foundation

- FastAPI backend and React + Vite frontend
- Root bootstrap option `5` to launch the workbench
- Automatic frontend dependency install/build when needed
- Session-aware launch with project-root and output-folder defaults
- Episode loading from cleaned and reviewed transcript bundles
- Cleaned vs reviewed transcript comparison surface
- On-demand semantic scan through the configured review backend
- Approved write-back into:
  - episode correction CSVs
  - `preferred_terms.txt`
  - `preferred_replacements.json`
- Workbench scan cache and audit-log persistence

## Active Work

These are the areas that best describe the current living roadmap.

### 1. Review Quality Hardening

Current focus:

- improve speaker-consistency correction reliability
- improve episode-QA usefulness on long real-world episodes
- reduce missed edits without encouraging over-editing
- keep patch behavior compact and conservative

### 2. Better Model Selection and Evaluation

Current focus:

- compare review models on practical structured-review behavior, not just published context sizes
- keep evaluating speed vs quality tradeoffs across local vLLM candidates
- make production model choice more evidence-based

### 3. Operational Clarity

Current focus:

- clearer summaries of what review actually changed
- better visibility into whether review meaningfully helped on a given run
- continued improvement to console/report language so batch progress and review status are easy to read

### 4. Workbench Usability

Current focus:

- make the episode review flow faster and less click-heavy
- improve transcript navigation and findings triage
- make semantic-scan results more understandable and actionable
- keep write-back behavior safe, visible, and reproducible

## Next Likely Phases

These are the most plausible next roadmap steps from here.

### Pipeline Phase 1: Review Precision and Utility

- strengthen prompts and reconciliation logic for speaker-consistency review
- improve long-context episode-QA behavior on genuinely long transcripts
- add more real-world fixtures that stress:
  - speaker drift
  - glossary pressure
  - long-context contradictions
  - false-positive over-edit temptation

### Pipeline Phase 2: Model Benchmarking Maturity

- turn benchmark results into a clearer production recommendation workflow
- add more explicit reporting around:
  - quality-per-second
  - correction yield
  - stage usefulness by model
- continue capacity profiling so model swaps are less guessy

### Pipeline Phase 3: Richer Run Reporting

- add stronger run-level summaries or dashboards combining:
  - processing time
  - audio duration
  - review changes
  - confidence/risk indicators
  - speaker-match uncertainty
- make it easier to see which episodes most deserve human review

### Pipeline Phase 4: Speaker Workflow Expansion

- stronger cross-episode speaker drift detection
- better recurring unnamed-speaker promotion workflow
- possible multi-host / co-host profile support

### Pipeline Phase 5: Broader Integration and Test Depth

- expand fixture and integration coverage further
- keep transcript-contract compatibility tight with downstream repos
- add more shared validation assumptions across the podcast toolchain

### Workbench Phase 1: Stabilize the Core Review Surface

- polish session/open flow, recent paths, and current-target visibility
- improve loading, empty, and backend-unavailable states
- improve transcript readability with sticky headers, row focus, and jump-to-finding
- strengthen audit/log visibility so operators can see what changed and where it was written

### Workbench Phase 2: Better Review Ergonomics

- add transcript filtering, sorting, speaker-only views, changed-only views, and findings-only views
- improve cleaned-vs-reviewed diff presentation
- add saved view preferences and stronger transcript navigation shortcuts
- surface episode-list quality signals such as unresolved findings and prior corrections

### Workbench Phase 3: Smarter Semantic Review Assistance

- define a clearer issue taxonomy for semantic scan findings
- add scan modes such as lightweight, glossary-focused, and deep semantic pass
- show stronger evidence, confidence, and deduplication for findings
- make preferred-term preservation visible in the review surface

### Workbench Phase 4: Write-Back Intelligence and Pipeline Feedback

- add stronger write-preview and conflict handling for corrections
- promote repeated episode fixes into glossary/replacement suggestions
- explain the downstream effect of each action more clearly
- add grouped approval flows and per-episode history of applied actions

### Workbench Phase 5: Cross-Episode and Run-Level Insight

- show `_episode_review_summary.csv`, `_review_run_report.*`, and `_speaker_workflow_report.*` in the UI
- rank episodes by likely review value or unresolved risk
- add cross-episode views for recurring issue patterns, glossary candidates, and speaker instability
- make the workbench useful as a library-level triage console, not just a single-episode viewer

## Deprioritized or Reframed Work

These are ideas that still matter, but no longer define the center of the roadmap.

- "Add optional review" is no longer a future item; it is implemented.
- "Add benchmarking" is no longer a future item; the benchmark now exists and the real work is improving how it is used.
- "Add cleanup levels" is done; the active question is how far deterministic cleanup should go while remaining trustworthy.
- "Make the workbench possible at all" is done; the active question is how quickly and safely it can support real operator review workflows.

## Summary

The project now has:

- a strong baseline transcription pipeline
- a real optional tier-2 review system
- a practical first-generation local review workbench

The roadmap from here is mostly about refinement and leverage:

- make review quality better
- make model choice more evidence-based
- make runs easier to understand
- make the workbench faster and safer for human review
- preserve compatibility for smaller and simpler setups
