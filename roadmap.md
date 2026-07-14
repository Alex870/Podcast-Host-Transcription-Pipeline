# Roadmap

`podcast-host-transcription-pipeline` is a mature local-first system for speaker-aware podcast transcripts and optional review. The next stage is measurable transcript/speaker quality, safe human feedback, and durable downstream contracts rather than more unconstrained LLM review.

## Principles

- Keep faster-whisper, diarization, speaker matching, and deterministic cleanup as the dependable baseline.
- Keep LLM review optional, evidence-preserving, and auditable.
- Preserve raw, cleaned, reviewed, and human-corrected artifacts with provenance.
- Measure review benefit against labeled data, not fluency alone.
- Keep output compatible with the downstream RAG/import/chat contract.

## Current Foundation

- Batch transcription, diarization, known-speaker matching, host extraction, cleanup, manifests, resume support, review backfill, and reports.
- Optional LM Studio/vLLM review, calibration, benchmark mode, glossary protection, and preferred-term safeguards.
- React/FastAPI review workbench with comparison, findings, and controlled write-back.
- Provider contracts for ASR, alignment, diarization provenance, and speaker embeddings.
- Provider-aware stage fingerprints with selective intermediate reuse.
- Optional WhisperX forced-alignment adapter with timestamp passthrough as the stable default.
- Versioned speaker profiles that prevent incompatible embedding families from being mixed.
- Full-pipeline gold-set benchmark and workbench reference annotation path.
- Atomic, dependency-validated stage caches through speaker attribution and deterministic cleanup.
- Permutation-aware diarization scoring, error-taxonomy slices, resource metrics, baseline comparisons, and promotion gates.
- Learned long-file diarization routing with same-run chunked fallback and runtime-scoped probe history.
- Staged review calibration/adaptation, protected preferred-term enforcement, practical capacity profiling, and project-local Teach-Me rules.

## Completed Modernization Phases 0-2

- **Phase 0, foundations:** provider contracts, stage-oriented orchestration boundaries within the current CLI, lazy optional alignment dependencies, stable provenance, and interruption-safe state persistence. Further decomposition of the large CLI remains a maintainability track rather than a prerequisite for provider experimentation.
- **Phase 1, measurement:** versioned workbench-authored gold references, WER, speaker-attributed WER, permutation-aware DER, host and glossary metrics, timing/resource evidence, taxonomy reporting, and baseline/candidate promotion decisions.
- **Phase 2, selective recomputation:** provider/config/dependency fingerprints for transcription, alignment, diarization, speaker attribution, and deterministic cleanup, with strict corruption rejection and legacy-baseline compatibility rules.
- **Phase 3, operational resilience:** atomic stage caches, provider-aware manifests, long-file diarization fallback/routing, run-scoped review calibration, adaptive overflow recovery, and review/debug reporting.
- **Structural Phase 5 work:** React/FastAPI workbench, gold-set annotation path, pipeline-quality benchmark reports, controlled write-back, learned review rules, and frontend build automation are implemented as additive foundations. Broader library-level triage remains future work.

Populating the gold set with representative human-approved excerpts is ongoing dataset stewardship rather than an unfinished software phase.

## Current Product Surface

The implemented system now has three connected operator loops:

1. **Production loop:** tier-1 transcription through deterministic cleanup, with optional tier-2 review and cleaned-JSON backfill.
2. **Measurement loop:** option `4` for review-model fixtures/capacity and option `6` for full-pipeline gold-set quality measurement.
3. **Human-feedback loop:** option `5` for transcript inspection, semantic scans, controlled write-back, gold annotations, and Teach-Me review-rule induction.

The next work should improve evidence quality and operator leverage rather than add another unmeasured model stage.

## Priority 1: Quality Measurement

- Build a gold set with audio, speaker labels, timestamps, glossary terms, and human-reviewed transcripts.
- Report ASR word accuracy where references exist, diarization/speaker attribution, host precision/recall, glossary preservation, and review-change precision.
- Track quality by crosstalk, noise, accents, music, sponsor reads, and long-form discussions.
- Separate deterministic cleanup changes from LLM review changes.
- Use an error taxonomy instead of one overall score.
- Grow the initial versioned gold-set structure into representative, human-approved podcast excerpts and use option `6` for baseline/candidate comparisons.

## Priority 2: Speaker Identity And Human Feedback

- Add cross-episode speaker drift detection with confidence and evidence clips.
- Support approved promotion of recurring unknown speakers with rollback.
- Improve multi-host/co-host support and profile versioning.
- Treat accepted/rejected workbench corrections as supervised feedback data, but validate before activating rules.
- Keep write-back append-only and auditable.

## Priority 3: Safer Review Intelligence

- Use server/model capability discovery instead of machine-specific profiles.
- Budget review prompts from actual context limits and reserve output tokens.
- Require schema validation, bounded retries, and change summaries for review.
- Add no-op regression tests to prevent over-editing good transcripts.
- Surface uncertainty and evidence for proposed corrections.

## Priority 4: Operations And Contracts

- Publish and enforce the shared transcript contract.
- Record audio hash, model versions, speaker-profile version, cleanup/review config, and output hashes.
- Improve batch reports with risk ranking, review yield, elapsed time, and operator action links.
- Test health checks, resumability, idempotency, and partial-failure recovery.

## Priority 5: Workbench Maturity

- Improve changed-only/speaker-only views, triage, write preview, and conflicts.
- Add cross-episode issue views and grouped approvals.
- Keep broad analytics in RAGScope.

## Sequencing

1. Populate the implemented gold-set framework and establish baseline metrics.
2. Evaluate forced alignment against timing and speaker-attributed WER before changing defaults.
3. Add an isolated Parakeet provider and compare it against faster-whisper.
4. Calibrate a candidate speaker embedder using the versioned profile contract.
5. Harden speaker drift, correction audit trails, and cross-episode triage using measured failures.
6. Feed contract-validated outputs into end-to-end downstream tests.
