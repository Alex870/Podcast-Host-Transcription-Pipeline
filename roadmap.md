# Roadmap

Updated: 2026-07-24

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
- Persistent bootstrap menu with safe external vLLM/LM Studio discovery, model validation, and atomic project-config updates.
- Versioned gold-set evaluation with WER, SA-WER, permutation-aware DER, glossary/host measures, condition slices, resource metrics, and promotion gates.
- Optional WhisperX, Parakeet, and candidate speaker-embedding paths with guarded diagnostics.
- Cross-episode speaker evidence, reversible recurring-speaker promotion, and profile-family compatibility checks.
- Versioned transcript contract consumed by downstream fixture tests.

## Value-Ordered Priorities

### 1. Populate and operate the real quality set — implemented

- The external private evaluation pack is configurable through `evaluation_pack_path`.
- The workbench exposes unlabelled, pending-review, adjudication, and human-approved queues.
- A guided 12-episode sampler enforces the 3 short / 6 typical / 3 long mix and records required condition coverage.
- Baseline acceptance requires 12 human-approved episodes, the required duration mix, and crosstalk, recurring-speaker, and noise/music coverage.

### 2. Make corrections first-class incremental artifacts — implemented

- `correction-manifest-v2` adds deterministic correction IDs, source anchors, before-value guards, statuses, supersession, and provenance while retaining v1 readers.
- Approved text changes write corrected JSON/TXT siblings and a compatibility correction CSV; rollback regenerates all projections.
- Correction notifications identify affected episode/span IDs for Podcast-RAG and RAGScope, with `downstream_pending` persisted when a consumer is unavailable.
- `episode-contract-v2` is emitted natively and legacy v1 episodes are upgraded by delta, cached rebuild, or automatic full reprocessing as required.

### 3. Mature recurring-speaker identity — implemented

- Episode-local speaker labels are no longer treated as identities across episodes.
- Speaker attribution emits versioned embedding evidence; candidates use deterministic complete-link clustering within one embedding family.
- The workbench supports evidence review, threshold-gated promotion, host/co-host/guest roles, merge, split, and rollback.
- `speakers.json` is normalized additively to schema v2 with stable IDs, aliases, roles, evidence, status, and history.

### 4. Run measured provider and alignment experiments

- Compare faster-whisper with Parakeet on the approved condition slices.
- Compare timestamp passthrough with WhisperX on timing, SA-WER, runtime, and failure recovery.
- Calibrate candidate speaker embedders using versioned profiles and identical excerpts.
- Keep provider installation and model licensing optional and isolated.

### 5. Improve operator resilience and privacy

- Add disk/capacity estimates, batch risk ranking, actionable recovery links, and background progress/cancellation.
- Validate backup/restore and interrupted-run recovery on a cache-free target.
- Add configurable retention/redaction for audio clips, review exports, logs, and temporary workbench data.
- Keep backend credentials and cloud-provider management out of the local/LAN configuration wizard unless a concrete provider requirement justifies a separate secure design.
- Continue decomposing the large CLI where it reduces change risk or enables testing; avoid a rewrite for its own sake.

## Sequencing

1. Complete the private 12-episode annotation campaign and accept its first measured baseline.
2. Exercise one real correction through Podcast-RAG and RAGScope using the new v2 notifications.
3. Review and promote recurring-speaker candidates produced by real episode embeddings.
4. Run provider/alignment/embedder comparisons and promote only demonstrated gains.
5. Complete target-machine resilience, privacy, and packaging checks.

The ecosystem-level sequence and promotion rules live in `../PODCAST_ECOSYSTEM_ROADMAP.md` when these repositories share a workspace.
## Phases 0–3 implementation status (2026-07-24)

Quality-set operations, contract-aware v2 upgrades, first-class correction artifacts, downstream notifications, and embedding-backed recurring-speaker identity are implemented. The remaining work is operational evidence: complete the private campaign, accept its baseline, exercise a real cross-repository correction, and promote only speaker/provider changes supported by that evidence.
