# Roadmap

Updated: 2026-07-17

`podcast-host-transcription-pipeline` is the evidence-creation and human-correction boundary for the podcast ecosystem. The core pipeline, provider contracts, gold-set tooling, selective recomputation, speaker workflows, review workbench, and clean-machine diagnostics are implemented. The roadmap now prioritizes representative human evidence and a correction loop that remains traceable downstream.

## Product Direction

- Keep faster-whisper, timestamp passthrough, pyannote, SpeechBrain ECAPA, and deterministic cleanup as the dependable defaults until measured alternatives win.
- Treat human-approved transcripts and speaker labels as authoritative evidence.
- Preserve raw, cleaned, reviewed, and corrected artifacts with immutable provenance.
- Make every correction consumable by Podcast-RAG without requiring unrelated stages to rerun.
- Keep private audio local and make provider/network boundaries explicit.

## Current Foundation

- Batch ASR, diarization, speaker matching, cleanup, manifests, resume, review backfill, and reports.
- Provider-aware stage fingerprints and atomic caches through deterministic cleanup.
- React/FastAPI review workbench with comparison, annotations, controlled write-back, and Teach-Me rules.
- Versioned gold-set evaluation with WER, SA-WER, permutation-aware DER, glossary/host measures, condition slices, resource metrics, and promotion gates.
- Optional WhisperX, Parakeet, and candidate speaker-embedding paths with guarded diagnostics.
- Cross-episode speaker evidence, reversible recurring-speaker promotion, and profile-family compatibility checks.
- Versioned transcript contract consumed by downstream fixture tests.

## Value-Ordered Priorities

### 1. Populate and operate the real quality set

- Select representative approved excerpts across duration, noise, music, crosstalk, accents, sponsor reads, hosts, and recurring guests.
- Link gold spans to the shared local evaluation pack and publish only approved aggregate results.
- Add workbench queues for missing references, adjudication conflicts, and high-impact errors.
- Establish release-critical ASR, diarization, speaker, glossary, timing, and resource baselines.

### 2. Make corrections first-class incremental artifacts

- Assign durable IDs to accepted transcript, timing, glossary, and speaker corrections.
- Emit append-only correction manifests with before/after hashes and affected source spans.
- Preview downstream invalidation before write-back and expose a machine-readable change set to Podcast-RAG.
- Detect corrections that invalidate evaluation judgments and route them for re-review rather than silently changing labels.

### 3. Mature recurring-speaker identity

- Rank cross-episode drift and unknown-speaker candidates by confidence, recurrence, and evaluation impact.
- Support multi-host and alias histories without mixing incompatible embedding families.
- Add evidence-clip review, merge/split proposals, conflict resolution, and rollback.
- Measure speaker-profile changes against false-match and missed-match rates before activation.

### 4. Run measured provider and alignment experiments

- Compare faster-whisper with Parakeet on the approved condition slices.
- Compare timestamp passthrough with WhisperX on timing, SA-WER, runtime, and failure recovery.
- Calibrate candidate speaker embedders using versioned profiles and identical excerpts.
- Keep provider installation and model licensing optional and isolated.

### 5. Improve operator resilience and privacy

- Add disk/capacity estimates, batch risk ranking, actionable recovery links, and background progress/cancellation.
- Validate backup/restore and interrupted-run recovery on a cache-free target.
- Add configurable retention/redaction for audio clips, review exports, logs, and temporary workbench data.
- Continue decomposing the large CLI where it reduces change risk or enables testing; avoid a rewrite for its own sake.

## Sequencing

1. Approve and populate the real gold/evaluation set.
2. Record the current default baseline.
3. Ship durable correction manifests and downstream invalidation preview.
4. Harden recurring-speaker review using measured failures.
5. Run provider/alignment/embedder comparisons and promote only demonstrated gains.
6. Complete target-machine resilience, privacy, and packaging checks.

The ecosystem-level sequence and promotion rules live in `../PODCAST_ECOSYSTEM_ROADMAP.md` when these repositories share a workspace.
## Phases 0–2 implementation status (2026-07-17)

Correction contract, deterministic preview/apply, canonical identities, and fixtures are implemented. Real one-podcast correction acceptance awaits the approved private evaluation pack.
