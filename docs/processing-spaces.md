# Processing Spaces

Processing spaces keep unrelated recordings and extracted corpora independent. Use one space for each podcast, work-meeting stream, interview project, or other context that should have separate speaker, glossary, review, and downstream history.

## Create a space

Run the root launcher and choose:

```text
10. Manage processing spaces
```

Choose `C` to create a managed space. The wizard asks for a name and context type, then creates a private layout under `partitions/<slug>`:

```text
intake/
output/
state/
speaker-references/   # podcast spaces
corrections/
```

Use `A` to adopt an existing source/output pair. Adoption records the folders without moving, copying, or deleting existing audio and transcript data.

Use `S` to inspect intake status or `P` to process one active space. The launcher starts only the selected space's pipeline.

The same operations are available in the browser workbench. Its processing-space selector controls which intake folder and transcript corpus are visible.

## Process new recordings

Place recordings in the selected space's `intake` folder, then run an explicit scan or processing run. The system does not start long GPU jobs just because a file appears.

From the command line:

```powershell
python podcast_transcribe_host.py --partition <partition-id>
```

The registry records each source file as discovered, ready, processing, completed, failed, quarantined, or missing. A completed file is skipped when its source fingerprint and relevant configuration are unchanged. A changed file is eligible for the minimum required recomputation.

## Context defaults

- `podcast` uses host and recurring-speaker identity workflows.
- `meeting` selects `anonymous_meeting`, retaining diarization labels without requiring reusable speaker identity.
- `custom` starts from the podcast workflow and can be adjusted through the workbench.

Speaker references, corrections, and glossary paths are partition-local by default. Nothing is shared across spaces unless an explicit shared-profile feature is added later.

## Registry and safety

The managed registry is stored in `config/partitions.sqlite3`. Operators do not edit it directly. The launcher can make a timestamped registry backup. The registry rejects overlapping active intake folders, prevents output/state folders from being placed inside intake, and keeps a per-space processing lock and run history.

Global credentials remain in the existing environment or global configuration. They are not copied into partition records or portable transcript metadata.

## Downstream corpus identity

Every transcript manifest includes the partition ID, display name, context type, workflow profile, and partition configuration fingerprint. Downstream processed artifacts and corpus releases use that identity so Podcast-RAG, Chroma Import, Podcast Chat, and RAGScope can select one corpus without accidentally mixing spaces.
