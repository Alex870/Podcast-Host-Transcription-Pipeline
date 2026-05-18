# Podcast Pipeline Data Contract

This document describes the JSON and Chroma metadata contract shared by the four podcast tools:

1. `Podcast-Host-Transcription-Pipeline`
2. `Podcast-RAG-pipeline`
3. `Chroma DB Import`
4. `PodCast Chat`

The goal is to make every handoff explicit enough that a downstream tool can validate inputs before starting expensive transcription, LLM preprocessing, embedding, or chat work.

## Transcript JSON

Produced by `Podcast-Host-Transcription-Pipeline` and consumed by `Podcast-RAG-pipeline`.

Expected top-level fields:

- `schema_version`: transcript schema version. Current version: `2`.
- `pipeline`: producer identifier, currently `podcast-host-transcription-pipeline`.
- `pipeline_version`: producer/model version string when available.
- `source_file`: original audio filename or path.
- `episode_title`: human-readable title when known.
- `metadata`: optional object containing episode-level metadata.
- `segments`: ordered transcript segment array.

Expected segment fields:

- `start`: segment start time in seconds.
- `end`: segment end time in seconds.
- `speaker`: normalized speaker label.
- `text`: transcript text.
- `transcription_confidence`: object derived from Whisper confidence fields.
- `content_quality`: deterministic review/filtering hints, such as possible sponsor blocks, boilerplate, repetition, music/transition text, or silence/non-speech.

Recommended episode date fields:

- `episode_date`: ISO date, for example `2026-02-04`.
- `episode_date_compact`: compact date, for example `20260204`.
- `episode_sort_key`: numeric `YYYYMMDD` value.

The executable contract for this repository lives in `src/podcast_transcribe/contract.py`. It defines the current transcript schema version and required fields and is used before transcript JSON outputs are written.

## Processed RAG Cache

Produced by `Podcast-RAG-pipeline` and consumed by `Chroma DB Import`.

Each `*.processed_documents.json` file represents one episode. The expected shape is:

```json
{
  "source_file": "episode_speaker_transcript.json",
  "source_fingerprint": "stable fingerprint",
  "episode_title": "Episode title",
  "documents": []
}
```

Each document must contain:

- `page_content`: retrieval text to embed.
- `metadata`: Chroma-ready metadata object.

Required document metadata:

- `node_id`: stable unique ID for the document.
- `node_type`: one of `leaf_chunk`, `cluster_summary`, `episode_thesis`, or `position_card`.
- `episode_id`: stable episode identifier.
- `episode_title`: human-readable episode title.
- `episode_date`: ISO episode date when known.
- `episode_sort_key`: numeric date key when known.

Speaker metadata:

- `speaker`: primary speaker for single-speaker nodes.
- `speakers`: JSON array or list of speaker names for multi-speaker nodes.
- `speaker_scope`: `single`, `multiple`, `mixed`, or empty.

Embedding metadata:

- `embedding_model`: recommended on each cache or metadata manifest when available.
- `embedding_dimension`: recommended after vectors are generated.

## Chroma Export

Produced by `Chroma DB Import` and consumed by `PodCast Chat`.

Each export is a self-contained folder:

```text
Podcast Name/
  chroma.sqlite3
  podcast.json
  ...Chroma internal files...
```

`podcast.json` expected fields:

- `podcast_name`
- `database_id`
- `collection_name`
- `embedding_model`
- `embedding_dimension`
- `embedding_device`
- `description`
- `date_range.start`
- `date_range.end`
- `episode_count`
- `chunk_count`
- `speakers`
- `episodes`
- `generated_at`
- `generated_by`

Speaker entries:

```json
{
  "id": "speaker-slug",
  "name": "Speaker Name"
}
```

Episode entries:

- `source_file`
- `source_fingerprint`
- `episode_id`
- `episode_title`
- `episode_date`
- `document_count`
- `speakers`
- `imported_at`

## Compatibility Rules

- `Podcast-RAG-pipeline`, `Chroma DB Import`, and `PodCast Chat` must agree on the embedding model.
- If an export was embedded with one model and queried with another, retrieval distances are unreliable.
- `episode_sort_key` should be preserved from transcript through Chroma metadata so date filtering works.
- Omitted speakers should be omitted from export metadata as if they were not imported.
- Episode-level thesis nodes and multi-speaker summary nodes may be preserved even when speaker-specific nodes are filtered out.

## Validation Expectations

Every stage should fail early with clear messages when required fields are missing:

- RAG should validate transcript segment shape before LLM preprocessing.
- Chroma import should validate processed documents before embedding.
- Podcast Chat should validate export metadata, Chroma collection availability, embedding model compatibility, and vector availability for speakers.
