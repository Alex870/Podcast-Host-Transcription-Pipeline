# Pipeline Gold Set

This directory stores human-approved transcript references for the full speech pipeline benchmark.

`manifest.json` is authoritative. Each enabled entry identifies a reference transcript JSON and either an explicit candidate JSON or an `audio_stem` resolved against the benchmark candidate directory.

Entries may use `approval_status` (`pending_review`, `human_approved`, or `adjudication_required`) and a stable `reviewer_id`. The Stage 7 readiness report refuses to call an entry ready until it has reference segments, condition tags where applicable, and explicit human approval. Private audio is never required in this checked-in directory.

Reference files use the normal transcript JSON contract so the workbench, contract validator, and benchmark runner share one representation.

Use the workbench Gold-set reference panel to add transcript excerpts, word timing, speaker labels, glossary terms, and notes. Tag excerpts with `crosstalk`, `noise`, `accent`, `music`, `sponsor_read`, `short_turn`, or `long_episode`. Option 6 reports aggregate and per-condition quality, including timing error.

For provider promotion, compare a candidate directory with a retained baseline directory using `--benchmark-baseline-dir`. The manifest's `promotion_thresholds` are intentionally strict by default: a candidate must complete every enabled entry without regressing WER or speaker-attributed WER.
