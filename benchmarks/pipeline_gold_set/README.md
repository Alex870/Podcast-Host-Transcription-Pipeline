# Pipeline Gold Set

This directory stores human-approved transcript references for the full speech pipeline benchmark.

`manifest.json` is authoritative. Each enabled entry identifies a reference transcript JSON and either an explicit candidate JSON or an `audio_stem` resolved against the benchmark candidate directory.

Reference files use the normal transcript JSON contract so the workbench, contract validator, and benchmark runner share one representation.

Use the workbench Gold-set reference panel to add human-approved segments. Tag excerpts with an error taxonomy such as `crosstalk`, `noise`, `accent`, `music`, `sponsor_read`, `short_turn`, or `long_form`. Option 6 reports aggregate and per-taxonomy quality.

For provider promotion, compare a candidate directory with a retained baseline directory using `--benchmark-baseline-dir`. The manifest's `promotion_thresholds` are intentionally strict by default: a candidate must complete every enabled entry without regressing WER or speaker-attributed WER.
