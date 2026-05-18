# Podcast Transcription & Host Identification Pipeline

This document describes the architecture and processing flow of `podcast_transcribe_host.py`.

---

## 🧠 System Flow Diagram

```mermaid
flowchart TD

    A[Start] --> 
    B["Setup + Validation
    ─ parse_args"] --> 
    C["Load Configs + Models
    ─ load_preferred_terms, load_replacement_map, get_device"] --> 
    D["Scan Audio Files"]

    D --> E{Files Found?}
    E -->|No| Z[Exit]
    E -->|Yes| F["Process Each File
    ─ process_file"]

    subgraph Processing
        direction TB
        F --> G["Transcription + Diarization
        ─ transcribe_audio, diarize_audio"]
        G --> H["Speaker Assignment
        ─ assign_speakers_to_segments"]
        H --> I["Host Detection + Matching
        ─ choose_host_speaker, match_known_speakers"]
        I --> J["Speaker Renaming
        ─ rename_speakers"]
        J --> K["Text Normalization
        ─ coalesce_segments"]
        K --> L["Quality Review + QA
        ─ collect_review_rows"]
    end

    subgraph Outputs
        direction TB
        L --> O1["Write Transcripts
        ─ write_text_transcript"]
        L --> O2["Write JSON + Review CSV
        ─ write_json_output, write_review_csv"]
    end

    O2 --> P{Update Host Profile?}
    P -->|Yes| Q["Save Profile
    ─ save_host_profile"]
    P -->|No| R[Skip]

    Q --> S["Build Episode Summary
    ─ build_episode_summary_row"]
    R --> S

    S --> F

    F --> T["Write Batch Summary CSV
    ─ write_episode_summary_csv"]
    T --> U[End]
```

---

## 🔍 Architectural Layers

### 1. Input & Configuration Layer

Handles CLI arguments and environment setup.

**Key responsibilities:**
- Parse runtime parameters (`parse_args`)
- Validate input directory and Hugging Face token
- Load:
  - Preferred vocabulary terms
  - Replacement mappings
  - Known speaker configurations
  - Host voice profile (if available)

---

### 2. Core Processing Pipeline (Per Audio File)

#### A. Transcription
- **Function:** `transcribe_audio`
- **Model:** Faster-Whisper

**Outputs:**
- Time-aligned segments
- Word-level timestamps
- Confidence metadata

---

#### B. Diarization
- **Function:** `diarize_audio`
- **Model:** Pyannote

**Followed by:**
- `assign_speakers_to_segments`

**Purpose:**
- Identify speaker turns
- Align speakers to transcript segments and words

---

#### C. Speaker Intelligence

This is the most advanced part of the system.

**Key functions:**
- `choose_host_speaker`
- `match_known_speakers`
- `rename_speakers`

**Capabilities:**
- Speaker embedding generation (SpeechBrain)
- Host identification via:
  - Reference audio
  - Persistent profile
  - Cosine similarity
- Fallback:
  - Dominant speaker assumption
- Known speaker matching
- Final label normalization (HOST, SPEAKER_01, etc.)

---

### 3. Text Normalization Layer

**Function:** `coalesce_segments`

**Features:**
- Merge adjacent segments from same speaker
- Apply glossary replacements
- Normalize transcription output
- Track replacement events for QA

---

### 4. Quality & Review System

**Function:** `collect_review_rows`

Generates structured QA signals:

**Examples:**
- Host not detected
- Low-confidence host match
- Ambiguous speaker identification
- Low host speech coverage
- Glossary corrections applied

---

### 5. Output Layer

For each episode:

- **Speaker transcript (TXT)**
- **Host-only transcript (TXT)**
- **Structured transcript (JSON)**
- **Review report (CSV)**

Batch-level:

- `_episode_review_summary.csv`

---

## ⚡ Notable Design Features

### 🔹 Adaptive Host Identification
- Learns host voice over time
- Updates embedding profile across episodes

---

### 🔹 Multi-Stage Speaker Resolution
1. Diarization (raw speaker IDs)  
2. Embedding similarity matching  
3. Known speaker override  
4. Final label normalization  

---

### 🔹 Feedback Loop
- Host profile persists and improves accuracy over time

---

### 🔹 Built-in QA System
- Automatically flags:
  - Weak matches
  - Ambiguities
  - Data quality issues

---

## 📌 Summary

This pipeline goes beyond transcription:

> It is a **speaker-aware, self-improving podcast intelligence system** that combines ASR, diarization, speaker recognition, and QA into a unified workflow.
