# Transcript Review Workbench Implementation Plan

This plan turns the workbench roadmap into a concrete execution sequence. It is intentionally scoped to the workbench itself, while reusing the existing pipeline contracts, reviewed artifacts, and correction/glossary write-back paths.

## Current Implementation Baseline

The initial workbench foundation described by this plan is now implemented:

- React/TypeScript/Vite frontend with a FastAPI local backend
- launcher-provided project/output defaults and automatic dependency/build handling
- cleaned/reviewed transcript comparison and semantic scan caching
- correction CSV, preferred-term, and replacement-map preview/apply flows
- gold-set segment annotation for pipeline-quality benchmarking
- project-local Teach-Me rule induction, bounded validation, approval, current-episode rerun, and optional backfill APIs/UI

The phases below are therefore forward-looking maturity work. They should not be read as prerequisites for the current workbench to launch or process existing outputs.

## Implementation Strategy

Build the workbench in five practical phases:

1. stabilize the current episode-review experience
2. improve operator speed and navigation
3. make semantic findings more actionable
4. deepen write-back and pipeline feedback loops
5. expand from single-episode review to library-level triage

The default rule throughout is:

- read cleaned transcript JSON as the canonical episode layer
- treat reviewed JSON as additive comparison/provenance
- never edit transcript outputs directly
- write approved fixes back into pipeline input artifacts only

## Phase 1: Stabilize the Core Review Surface

Goal: make the current workbench reliable enough for repeated daily use.

Key changes:

- session and launch polish
  - preserve launcher-provided project/output defaults end to end
  - persist recent/last-opened paths in UI state
  - show the currently active project root and output folder clearly
- UI state hardening
  - add explicit loading, empty, and error states for:
    - session open
    - episode list
    - episode detail
    - semantic scan
    - audit log
  - avoid blank or ambiguous panels when data is absent
- transcript review basics
  - add sticky transcript table headers
  - add jump-to-finding behavior from the findings panel
  - highlight the active segment/finding row clearly
- audit visibility
  - surface recent write-back actions with timestamp, action type, and target path
  - make it obvious whether an action was previewed or applied

Backend/API work:

- keep the current API surface but normalize response shapes for empty/error/loading scenarios
- make workbench session and audit responses stable enough for richer UI state handling

Acceptance criteria:

- option `5` launches reliably and opens a usable review screen
- project/output fields populate correctly from the launcher session
- an operator can select an episode, inspect transcript and findings, run a semantic scan, and apply a correction without confusion

## Phase 2: Better Review Ergonomics

Goal: make long-episode review faster and less fatiguing.

Key changes:

- transcript table controls
  - sort by segment id, time, speaker, change status
  - filter by speaker
  - filter to changed-only, findings-only, or reviewed-only rows
  - remember filter/sort/view state between launches
- diff presentation
  - improve cleaned-vs-reviewed comparison with stronger visual diff cues
  - allow inline compare and side-by-side compare modes
- navigation improvements
  - next/previous finding navigation
  - jump to next changed segment
  - keyboard shortcuts for common navigation actions
- episode list quality signals
  - show reviewed-present, semantic-scan-present, finding count, recent correction count, and unresolved issue cues

Frontend work:

- centralize transcript/filter/view state in a durable UI state model
- add local persistence for table/view preferences

Acceptance criteria:

- the operator can rapidly isolate interesting transcript rows
- cleaned vs reviewed changes are easy to inspect
- saved view preferences survive relaunch

## Phase 3: Smarter Semantic Review Assistance

Goal: make scans more selective, more legible, and more actionable.

Key changes:

- finding taxonomy
  - define explicit categories such as:
    - likely mistranscription
    - malformed restart/disfluency
    - glossary regression
    - speaker drift suspicion
    - contradiction or cross-segment inconsistency
- scan modes
  - lightweight pass
  - glossary-focused pass
  - deep semantic pass
- evidence and confidence
  - attach supporting segment ids and nearby context
  - include backend/model identity and scan timestamp
  - include confidence/severity fields in a normalized UI shape
- preferred-term visibility
  - show when a finding intersects a protected preferred term
  - make it visible when the review guard would preserve a reserved spelling
- finding cleanup
  - deduplicate low-value repeated findings
  - suppress noisy repeated warnings for identical spans

Backend/API work:

- extend semantic scan cache payloads with:
  - scan mode
  - backend/model identity
  - confidence/evidence fields
  - normalized issue type
- keep backward compatibility with older cached scan files by tolerating missing fields

Acceptance criteria:

- semantic scan results explain themselves clearly enough for an operator to accept or reject them quickly
- preferred-term safety is visible in the workbench, not only in logs/debug artifacts

## Phase 4: Write-Back Intelligence and Pipeline Feedback

Goal: make the workbench a reliable improvement loop for future pipeline runs.

Key changes:

- stronger correction previews
  - show exact before/after text
  - show target correction CSV path before apply
  - show whether the segment already has an existing correction entry
- conflict handling
  - surface existing correction values and let the operator replace or keep them deliberately
- glossary promotion
  - turn repeated approved episode fixes into suggested preferred-term or replacement-map updates
  - make the user choose the write-back target explicitly
- action history
  - show per-episode applied actions
  - show which findings led to which correction/glossary updates
- downstream-effect clarity
  - distinguish:
    - episode-only correction
    - future-run glossary/replacement impact

Backend/API work:

- extend preview endpoints with conflict/existing-entry context
- extend audit entries with richer linkage between finding, preview, apply action, and target file

Acceptance criteria:

- operators can understand the consequence of each write-back before applying it
- repeated cleanup work can be promoted into reusable pipeline knowledge without guesswork

## Phase 5: Cross-Episode and Run-Level Insight

Goal: turn the workbench into a review and triage console for the processed library.

Key changes:

- run/report views
  - display `_episode_review_summary.csv`
  - display `_review_run_report.*`
  - display `_speaker_workflow_report.*`
- ranking and prioritization
  - highlight episodes with:
    - high finding counts
    - no reviewed output
    - recent semantic-scan failures
    - speaker instability signals
    - heavy correction activity
- cross-episode workflows
  - recurring issue clusters
  - repeated glossary candidates
  - recurring unnamed speaker evidence
  - episodes where reviewed output changed the most
- library-level dashboards
  - review coverage
  - unresolved issue counts
  - correction history
  - scan freshness

Backend/API work:

- add endpoints that aggregate existing batch/run artifacts into UI-friendly summaries
- avoid recalculating pipeline logic in the workbench when the pipeline already emits the required report artifacts

Acceptance criteria:

- an operator can choose what to review next based on actual risk or value
- the workbench becomes useful even before opening a single episode

## Shared Testing and Validation

Carry these checks across all phases:

- launcher starts and cleans up the backend process correctly
- launcher-provided session defaults populate the UI correctly
- episodes load with cleaned-only outputs and cleaned+reviewed outputs
- semantic scan handles:
  - backend available
  - backend unavailable
  - malformed backend response
- all write-back paths stay inside the selected project root
- transcript outputs are never mutated directly
- correction/glossary writes remain reproducible for later pipeline reruns
- UI state persistence never corrupts operational data

## Suggested Delivery Order

Recommended order of execution:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5

That order keeps the product usable at every step and avoids building more semantic or cross-episode complexity on top of a shaky episode-review foundation.
