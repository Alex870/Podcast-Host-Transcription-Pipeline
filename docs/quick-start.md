# Quick Start

This guide is the shortest path from a fresh clone to a successful transcription run.

## 1. Prerequisites

You will need:

- Windows with PowerShell
- Conda or Miniconda on `PATH`
- a Hugging Face account and access token
- accepted access for:
  - `pyannote/speaker-diarization-community-1`
  - `pyannote/segmentation-3.0`
- a shared FFmpeg 7 build installed with a usable `bin` directory such as `C:/ffmpeg7/bin`
- the validated Windows GPU stack: PyTorch 2.9, TorchAudio 2.9, TorchVision 0.24, and TorchCodec 0.8.1
- Node.js with a working `npm` if you want to use the transcript review workbench
- enough local compute for Whisper, pyannote, and PyTorch-based audio processing

An NVIDIA GPU is strongly recommended for practical throughput.

## 2. Clone the Repository

```powershell
git clone https://github.com/Alex870/Podcast-Host-Transcription-Pipeline.git
cd Podcast-Host-Transcription-Pipeline
```

## 3. Create the Python Environment

The PowerShell launchers assume a conda environment named `podcast-transcribe`, so using that name is the easiest path.

```powershell
conda create -n podcast-transcribe python=3.10 -y
conda activate podcast-transcribe
pip install -r podcast_transcribe_requirements.txt
```

The requirements file installs CUDA 12.8 PyTorch wheels and the matching TorchCodec version. The active environment should report `torch.cuda.is_available() == True` on a supported NVIDIA system.

If you prefer a different environment name, update the scripts under `scripts/`.

## 4. Set Up Hugging Face Access

The diarization stages will not run without a valid token.

First:

1. sign in to Hugging Face
2. request and accept access to `pyannote/speaker-diarization-community-1`
3. request and accept access to `pyannote/segmentation-3.0`
4. create an access token

Then provide the token in one of these ways:

- `HF_TOKEN` in the shell environment
- `HF_TOKEN` in a local `.env` file
- `hf_token` in `podcast_transcribe_config.json`

Example:

```powershell
$env:HF_TOKEN = "your_token_here"
```

## 5. Create the Runtime Config

Copy the example config:

```powershell
Copy-Item .\examples\podcast_transcribe_config.example.json .\podcast_transcribe_config.json
```

Then edit `podcast_transcribe_config.json` for your machine. A good first baseline is:

```json
{
  "default_source_dir": "D:/Speech_to_text/audio",
  "hf_token": "",
  "ffmpeg_bin_dir": "C:/ffmpeg7/bin",
  "known_speakers_dir": "speaker_reference_samples",
  "preferred_terms_file": "preferred_terms.txt",
  "replacement_map_json": "preferred_replacements.json",
  "runtime_profile": "baseline_16gb",
  "backend": "none",
  "cleanup_level": "normal",
  "host_profile_json": "host_profile.json",
  "model": "distil-large-v3",
  "language": "en",
  "device": "auto",
  "compute_type": "auto",
  "beam_size": 5,
  "batch_size": 8,
  "isolate_files": true,
  "assume_dominant_speaker_is_host": true,
  "host_threshold": 0.45
}
```

You do not need to edit JSON to create additional processing spaces. Start `Run Podcast Transcribe.ps1`, choose `10. Manage processing spaces`, and create a space for each podcast, meeting context, or interview collection. The wizard creates separate intake/output/state folders and stores the space configuration in `config/partitions.sqlite3`. Drop recordings into the selected space's intake folder and run that space explicitly with the launcher or workbench.

## 6. Set Up `preferred_terms.txt` and `preferred_replacements.json`

These files are worth setting up early because they improve consistency across large runs.

Create working copies:

```powershell
Copy-Item .\examples\preferred_terms.txt .\preferred_terms.txt
Copy-Item .\examples\preferred_replacements.json .\preferred_replacements.json
```

Point your config at the working copies:

```json
{
  "preferred_terms_file": "preferred_terms.txt",
  "replacement_map_json": "preferred_replacements.json"
}
```

Use them like this:

- `preferred_terms.txt`: one protected preferred term or phrase per line
- `preferred_replacements.json`: alias-to-preferred cleanup mappings for common mistranscriptions

Preferred terms are now treated as reserved spellings during optional LLM review. If a term is already correct, the review stages should preserve it.

## 7. Optional: Add Known Speaker Samples

If you want stable speaker naming across episodes, add clean reference clips to `speaker_reference_samples` and edit `speaker_reference_samples/speakers.json`.

Example:

```json
{
  "speakers": [
    {
      "name": "HOST",
      "is_host": true,
      "files": ["host_sample.wav"]
    },
    {
      "name": "Guest_A",
      "files": ["guest_a_sample.wav"]
    }
  ]
}
```

Best practices:

- use short, clean single-speaker clips
- avoid overlap, music beds, and heavy noise
- keep the clips in the configured `known_speakers_dir`

## 8. Optional: Configure Filename Date Parsing

By default, episode dates are read from the last valid `YYYYMMDD` token in the filename.

If your filenames use another convention, update `filename_date` in `podcast_transcribe_config.json`:

```json
{
  "filename_date": {
    "preset": "american_podcast",
    "position": "last",
    "formats": ["YYYY-MM-DD", "MM-DD-YYYY"]
  }
}
```

## 9. Run Environment Validation

Use the bootstrap:

```powershell
.\Run Podcast Transcribe.ps1
```

Choose `1` for environment validation.

The validation script checks:

- config resolution
- Hugging Face token discovery and pyannote access
- FFmpeg path resolution
- Python dependencies and CUDA visibility
- TorchCodec and pyannote `AudioDecoder` availability for path-based audio decoding
- speaker reference config discovery
- effective runtime profile and review settings
- review backend reachability when review is enabled

## 10. Run the Pipeline

Use the same bootstrap and choose `2`.

The launcher will:

- use configured paths when they are valid
- fall back to prompts only when needed
- write prompted values back into the config
- pause at the end so results stay visible

Option `8` explicitly downloads the selected revision-pinned provider artifacts after preflight. Normal processing never downloads models implicitly. Option `9` selects the `anonymous_meeting` profile, which keeps diarization labels but skips speaker identity, host profiling, and LLM review.

## 11. Optional: Enable LLM Review

If you have a stronger review machine or LAN-served vLLM backend, enable review in `podcast_transcribe_config.json`.

The recommended setup path is bootstrap option `7`, `Configure external review LLM`. Enter the server IP address, hostname, or URL; choose one of the models discovered from the server; and review the exact config preview before saving. The wizard tests chat completions before it writes anything and preserves all unrelated settings.

Example:

```json
{
  "runtime_profile": "high_context_5090",
  "backend": "vllm",
  "review_base_url": "http://192.168.1.230:8000",
  "review_model_name": "Inferact/Qwen3.8-27B-NVFP4",
  "review_reasoning_effort": "none",
  "review_batch_token_limit": 12000,
  "review_candidate_filter": true,
  "transcript_cleanup_review": true,
  "glossary_correction_review": true,
  "speaker_consistency_review": true,
  "episode_qa_review": true
}
```

New episodes will run tier 1 and tier 2 together. Legacy episodes that already have valid cleaned JSON can be backfilled to reviewed outputs without rerunning Whisper, diarization, or speaker matching.

## 12. Optional: Run the Review Benchmark

Use the bootstrap and choose `4` to compare review models without rerunning heavy audio processing.

Benchmark mode runs the checked-in cleaned-transcript fixtures through the staged review pipeline and writes:

- `review_benchmark_report.json`
- `review_benchmark_report.md`

The option `4` reports include review-model speed, stability, quality, and usable-context capacity.

After you have saved human-approved gold-set references in the workbench, bootstrap option `6` writes separate full-pipeline reports:

- `pipeline_quality_benchmark_report.json`
- `pipeline_quality_benchmark_report.md`

The option `6` reports cover transcript and speaker quality, timing, glossary preservation, completion, and available processing/resource metrics. They do not run review-model capacity probes.

## 13. Optional: Launch the Transcript Review Workbench

Use the bootstrap and choose `5`.

The workbench is a local browser app for reviewing already-processed episodes. It can:

- load cleaned and reviewed transcript bundles
- show transcript, metadata, and provenance in one view
- run on-demand semantic issue scans through the configured local/LAN review backend
- write approved transcript corrections into episode correction CSVs
- write approved glossary changes into `preferred_terms.txt` and `preferred_replacements.json`

Option `5` now handles frontend setup for you:

- if `workbench-ui/node_modules` is missing, it installs frontend dependencies automatically
- if `workbench-ui/dist` is missing or stale, it rebuilds the workbench bundle automatically

You still need Node.js/npm available on the machine the first time that setup is required.

The launcher pre-fills the project root from the running repository and the output folder from the configured output location when available. You can confirm or change both values on first use:

- the project root
- the processed output folder

## 14. Where Outputs Appear

Outputs are written to the configured output directory, typically an `output` folder beside the source folder.

Important files:

- per-episode transcript, cleaned transcript, reviewed transcript, CSV, and manifest files
- `_episode_review_summary.csv`
- `_batch_report.md`
- `_review_run_report.json`
- `_speaker_workflow_report.json`
- `_workbench/` for semantic scan cache files
- `_processing_artifacts/` for reusable resume/debug artifacts; the fingerprinted `speaker_audio_16k_mono.wav` cache is created only while processing and removed after successful completion

## 15. If You Are Migrating from an Older Working Directory

Use the bootstrap and choose `3`.

The migration helper can copy forward:

- runtime config
- preferred terms and replacements
- speaker reference samples and `speakers.json`
- processed state and output contents
- host profile
- pretrained speaker model directory
- corrections directory
- configured source directory when it lives inside the legacy repo

It also rewrites repo-local absolute paths in the migrated config so the new repository is portable.

## Next Stop

- feature walkthroughs: [`user-manual.md`](user-manual.md)
- full config key reference: [`config-reference.md`](config-reference.md)
- engineering details: [`architecture.md`](architecture.md)
