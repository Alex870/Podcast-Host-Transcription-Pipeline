# User Manual

This manual describes how to operate the pipeline once the basic setup is done.

## Bootstrap Options

Run:

```powershell
.\Run Podcast Transcribe.ps1
```

The menu options are:

1. `Run environment validation`
2. `Run transcription pipeline`
3. `Migrate settings and state from a legacy directory`
4. `Run review benchmark`
5. `Launch transcript review workbench`

## Normal Transcription Workflow

The normal operator flow is:

1. place new audio files in the configured source directory
2. verify `podcast_transcribe_config.json`
3. run the bootstrap and choose option `2`
4. review outputs in the output directory

Tier 1 baseline processing includes:

- Whisper transcription
- speaker diarization
- host and speaker matching
- deterministic cleanup
- baseline JSON/TXT/CSV/manifests

If review is enabled, tier 2 runs after tier 1 and writes additive reviewed outputs.

## What Option 2 Actually Does

Option `2` is the smart production path. It can handle mixed folders in one run:

- brand-new episode: `tier1+tier2`
- baseline-complete episode missing required review stages: `tier2-only backfill`
- fully complete episode for the currently enabled review stages: skipped

This means you do not need a separate operator mode just to backfill review on older cleaned JSON outputs.

## Environment Validation Workflow

Option `1` is the first thing to run after setup changes.

It checks:

- token resolution and pyannote access
- FFmpeg path usability
- Python package availability
- CUDA visibility
- speaker reference discovery
- effective runtime profile and review toggles
- configured review backend reachability when review is enabled

Use it whenever the environment, CUDA stack, FFmpeg install, or backend settings change.

## Migration Workflow

Option `3` helps move a new checkout onto the state of an older working directory.

It can migrate:

- `podcast_transcribe_config.json`
- `preferred_terms.txt`
- `preferred_replacements.json`
- `speaker_reference_samples`
- `speakers.json`
- processed-state files
- `host_profile.json`
- pretrained speaker model directories
- corrections directory contents
- prior output directory contents

## Transcript Review Workbench

Option `5` launches the local browser workbench.

The workbench is meant for already-processed episodes. It reads cleaned and reviewed transcript bundles from the output directory and presents:

- cleaned transcript text
- reviewed transcript differences when present
- summary/provenance data
- deterministic QA signals
- optional semantic issue findings from the configured review backend

The launcher now tries to keep the frontend bundle current automatically:

- if frontend dependencies are missing, it runs `npm install`
- if the built bundle is missing or older than the tracked frontend source/config files, it runs `npm run build`
- it then serves the built frontend from the backend, rather than switching normal operator use into a Vite dev server

Node.js/npm still needs to be installed on the machine for that automatic setup to work.

The first run asks for:

- the project root
- the processed output folder

Approved write-back actions in v1 are intentionally narrow:

- episode text corrections -> correction CSVs
- preferred glossary additions -> `preferred_terms.txt`
- alias/replacement updates -> `preferred_replacements.json`

The workbench writes semantic scan cache files under `_workbench/` and audit entries under `.workbench/`.
- configured source directory contents when that source directory lives inside the legacy repo

Important migration behavior:

- the script warns once before overwriting existing target files
- it prints a pass/warn checklist at the end
- it pauses for Enter so the result stays visible
- repo-local absolute paths in the migrated config are rewritten to fit the new repository layout

## Review-Enabled Workflow

Optional LLM review is additive. It does not replace raw or deterministically cleaned outputs.

Review stages run in this order:

1. transcript cleanup review
2. glossary correction review
3. speaker consistency review
4. episode QA review

Review stages are controlled by:

- `runtime_profile`
- `backend`
- `review_base_url`
- `review_model_name`
- the four stage booleans

If the backend is unavailable or returns unusable output even at minimum adaptive size, baseline outputs still complete and the stage status is recorded in metadata and reports.

## Tier-2 Backfill Behavior

Tier-2 backfill uses `*_cleaned_speaker_transcript.json` as the canonical input.

That means:

- old episodes can gain reviewed outputs later
- Whisper, diarization, and speaker matching do not need to be rerun
- review-enabled runs can enrich a legacy library over time

The pipeline classifies each episode based on:

- whether baseline outputs are complete
- whether required reviewed outputs exist
- whether the reviewed JSON proves completion for the review stages enabled in the current run

## Cleanup Levels

`cleanup_level` controls deterministic cleanup before optional LLM review.

Available values:

- `disabled`: no deterministic cleanup
- `conservative`: only the safest cleanup passes
- `normal`: default balanced cleanup
- `aggressive`: stronger readability cleanup, including bounded restart-pruning

Aggressive mode now removes a narrow class of low-value spoken restarts such as:

- `you know,`
- abandoned repeated clause starts
- short dead-end negated restarts that are immediately superseded

It is still deterministic and bounded. It does not attempt open-ended paraphrasing.

## Review Calibration and Adaptive Sizing

When review is enabled, the first reviewable episode in a run triggers a fresh run-scoped calibration.

The review system then:

- starts from a safe calibrated budget
- shrinks quickly when truncation occurs
- grows conservatively after a long streak of clean successes
- keeps adapting separately for:
  - local text review
  - local speaker review
  - long-context review

Overflow is treated as an adaptation problem, not an immediate stage skip. The review engine keeps shrinking until it reaches a minimum adaptive floor. Only true backend or invalid-response failures at the minimum size can leave a stage incomplete.

## Review Debugging

Set `review_debug` to `true` to capture per-stage request/response artifacts.

By default, debug output is written under:

```text
<output>\_processing_artifacts\<episode>\review_debug
```

You can override that location with `review_debug_dir`.

These artifacts are useful when:

- diagnosing truncation
- inspecting prompt/response behavior
- checking preferred-term regressions
- comparing model behavior during review benchmarking

## Preferred-Term Protection

`preferred_terms.txt` is now more than a hint. It acts as the reserved spelling list for optional review.

Review stages should:

- preserve an already-correct preferred term exactly
- correct aliases toward the configured preferred term
- avoid "improving" the text away from the reserved spelling

Protected-term regressions are surfaced in debug and benchmark outputs.

## Benchmark Mode

Option `4` runs the dedicated review benchmark.

It does not process audio. Instead, it runs the staged review pipeline against checked-in cleaned-transcript fixtures.

The benchmark reports on:

- speed
- stability
- quality
- usable structured-review capacity per stage

Capacity profiling uses the real prompt and parser path, so it is much closer to practical usable space than a model's advertised context length.

## How to Read Reviewed Outputs

Reviewed files are siblings of the cleaned outputs:

- `*_reviewed_speaker_transcript.txt`
- `*_reviewed_host_only.txt`
- `*_reviewed_speaker_transcript.json`

The reviewed JSON includes:

- `review_schema_version`
- `review_metadata`
- per-stage results
- whether review ran inline or as backfill
- whether episode QA ran in full-episode or chunked mode
- segment-level reviewed text/provenance fields

## Manifests and Batch Reports

Per episode:

- `*_manifest.json`

Per batch:

- `_episode_review_summary.csv`
- `_batch_report.md`

These files help answer:

- what ran
- what was skipped
- whether review changed anything
- which backend/profile/model were used
- which stages completed

## Common Operator Decisions

Use `baseline_16gb` when:

- you want only the baseline transcription pipeline
- no local review backend is available
- you are optimizing for compatibility

Use `high_context_5090` when:

- you have a strong local or LAN-served backend
- you want all review stages available by default
- you want episode-QA review and benchmarking

Use `custom` when:

- you want explicit control over context budget and capability flags
- you are experimenting with non-default review backends or model limits

Use `cleanup_level = aggressive` when:

- you want cleaner readability
- you accept stronger but still bounded deterministic cleanup

Stay on `cleanup_level = normal` when:

- you want conservative cleanup as the default source for review and downstream processing

## Troubleshooting Entry Points

Start with option `1` when the problem is environmental.

Look at review debug artifacts when the problem is review-specific.

Run option `4` when you are comparing review models or checking if a new vLLM model is behaving sanely before a long production run.

Check:

- `_episode_review_summary.csv`
- `_batch_report.md`
- per-episode `*_manifest.json`
- `_processing_artifacts`

for the fastest picture of what the run actually did.

## Related Docs

- setup path: [`quick-start.md`](quick-start.md)
- full config key reference: [`config-reference.md`](config-reference.md)
- transcript/data contract: [`podcast_pipeline_contract.md`](podcast_pipeline_contract.md)
- architecture: [`architecture.md`](architecture.md)
