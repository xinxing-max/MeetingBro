# Architecture Overview

This document describes MeetingBro's technical architecture for contributors and developers.

---

## High-level data flow

```
Audio source
    │
    ▼
Audio capture (soundcard loopback / sounddevice mic)
    │
    ▼
VAD (Silero) — filters non-speech frames
    │
    ▼
Audio conditioning — normalizes RMS, peak limiting
    │
    ├──────────────────────────────┐
    ▼                              ▼
Formal ASR (Whisper)          Preview ASR (Qwen3 or tiny-Whisper)
accurate, ~2–5s latency       fast subtitles, ~0.5–1s latency
    │                              │
    └──────────────┬───────────────┘
                   ▼
           Session Manager
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Speaker diarization   Translation (LLM)
   (energy-based)        zh ↔ en ↔ de
          │                 │
          └────────┬─────────┘
                   ▼
          Summarization (LLM)
          ├── Rolling summary (3–5 min window)
          └── Meeting Board (cumulative)
                   │
                   ▼
          Storage (SQLite) + WebSocket → Frontend (Electron/React)
```

---

## Repository structure

```
MeetingBro/
├── app/
│   ├── backend/
│   │   └── meetingbro/
│   │       ├── main.py          # FastAPI app, HTTP + WebSocket routes
│   │       ├── schemas.py       # Pydantic data models
│   │       ├── exporter.py      # Meeting export (Markdown + JSON)
│   │       ├── hardware.py      # Hardware profile detection
│   │       ├── asr/             # ASR backends
│   │       │   ├── base.py      # Abstract ASR interface
│   │       │   ├── faster_whisper_adapter.py
│   │       │   └── qwen3_asr_adapter.py
│   │       ├── audio/           # Audio input pipeline
│   │       │   ├── capture.py   # Core capture logic
│   │       │   ├── loopback.py  # Windows WASAPI loopback
│   │       │   ├── mixed.py     # Mixed audio source
│   │       │   ├── vad.py       # Voice Activity Detection
│   │       │   └── enhancement.py  # Audio conditioning
│   │       ├── diarization/     # Speaker identification
│   │       │   ├── base.py
│   │       │   └── energy.py    # Energy-based diarizer
│   │       ├── llm/             # LLM client
│   │       │   └── openai_compatible.py
│   │       ├── session/         # Orchestration
│   │       │   ├── manager.py   # SessionManager
│   │       │   └── profiles.py  # Runtime profiles
│   │       ├── storage/
│   │       │   └── db.py        # SQLite schema and queries
│   │       ├── summarization/
│   │       │   ├── llm.py       # LLM-based summarization
│   │       │   └── heuristic.py # Fallback (no LLM)
│   │       └── translation/
│   │           ├── llm.py       # LLM-based translation
│   │           └── passthrough.py  # Fallback (no LLM)
│   └── frontend/
│       ├── electron/
│       │   └── main.cjs         # Electron main process
│       └── src/
│           ├── App.tsx          # Main React component
│           ├── types.ts         # TypeScript type contracts
│           └── session/
│               └── useSessionSocket.ts  # WebSocket state hook
├── data/                        # Sample audio, SQLite DB (gitignored)
├── docs/                        # Documentation
├── exports/                     # Meeting export output (gitignored)
├── models/                      # ML models (gitignored)
├── scripts/                     # Benchmark and verification scripts
└── tests/                       # Unit tests
```

---

## Backend: FastAPI + WebSocket

The backend is a Python FastAPI application. It exposes:

- **REST endpoints** for creating meetings, exporting, and fetching history
- **WebSocket endpoint** (`/ws/session/{meeting_id}`) for live session data

The WebSocket protocol:

**Backend → Frontend events:**

| `type` | Description |
|---|---|
| `transcript_segment` | A new transcription result (preview or formal) |
| `summary_snapshot` | A new summary (rolling, cumulative, or final) |
| `speaker_update` | Speaker label assigned to a segment |
| `note_saved` | Confirmation of a saved note |
| `session_state` | State change (starting, active, stopping, stopped) |
| `error` | Error with a code and message |

**Frontend → Backend commands:**

| `type` | Description |
|---|---|
| `save_note` | Save a note with optional source reference |
| `stop` | End the session and produce a final summary |

---

## Two ASR layers

MeetingBro runs two ASR paths in parallel to balance latency and accuracy:

**Preview ASR** (fast subtitles):
- Model: Qwen3 0.6B int8 (via sherpa-onnx) or tiny-Whisper
- Latency: ~0.5–1.0 seconds
- Use: displayed immediately as live subtitles
- Configurable via `MEETINGBRO_PREVIEW_ASR_BACKEND`

**Formal ASR** (accurate):
- Model: Whisper (small, medium, or auto-selected)
- Latency: ~2–5 seconds
- Use: replaces preview text with higher-accuracy result
- Configurable via `MEETINGBRO_WHISPER_SIZE`

Both paths share the same audio buffer but operate on independent worker threads.

---

## Two live summary layers

**Rolling Summary:**
- Generated every 60–90 seconds
- Covers the most recent 3–5 minutes of transcript
- Answers "what did I just miss?"
- Stored as `summary_type = "rolling_summary"` in the database

**Meeting Board (cumulative summary):**
- Generated every 3–5 minutes
- Covers the entire meeting from start to now
- Structured: topics → decisions → action items → open questions
- Answers "where is this meeting overall?"
- Stored as `summary_type = "cumulative_meeting_summary"`

---

## Session Manager

`session/manager.py` is the central orchestrator. It:

1. Receives audio chunks from the capture source
2. Passes chunks through VAD and conditioning
3. Dispatches chunks to both ASR workers
4. Merges preview and formal transcript results
5. Triggers diarization, translation, and summarization
6. Sends events to connected WebSocket clients via an async event queue

The event queue is bounded (`maxsize=1024`) with backpressure: if the frontend falls behind, older events are dropped rather than blocking ASR.

---

## Runtime profiles

Runtime profiles are named presets for the ASR and audio pipeline settings:

| Profile | Use case |
|---|---|
| `balanced` | Default — good trade-off for most meetings |
| `low_latency` | Faster subtitles, may fragment on long pauses |
| `robust` | Longer context windows, better for noisy audio |
| `multilingual` | Same as balanced, language lock explicitly off |
| `single_language` | Language lock on, slightly faster |

Profiles can be selected in the UI or set via `MEETINGBRO_RUNTIME_PROFILE`.

---

## Database schema

MeetingBro stores all data in a local SQLite database (`data/meetingbro.db`):

```sql
meetings (id, started_at, ended_at, preferred_summary_language)

transcript_segments (
  id, meeting_id, start_time, end_time, text,
  original_language, speaker_id, confidence,
  quality,         -- 'ok' | 'uncertain' | 'low'
  translations,    -- JSON: {lang: text}
  created_at
)

summary_snapshots (
  id, meeting_id, summary_type, time_start, time_end,
  language, content, source_segment_ids,
  is_latest, translations, created_at
)

notes (id, meeting_id, content, source_type, source_id, time_seconds, created_at)

speakers (id, meeting_id, display_name, inferred_label, confidence, is_local_user)
```

---

## Adding a new ASR backend

1. Create a new class in `app/backend/meetingbro/asr/` that extends `ASRAdapter` from `base.py`.
2. Implement the `transcribe(audio: np.ndarray, sample_rate: int) -> TranscriptResult` method.
3. Register the new backend in `session/manager.py` where the adapter is instantiated.
4. Add the corresponding `MEETINGBRO_*` config variables to `.env.example`.

The same pattern applies for adding a new diarization, translation, or summarization backend.

---

## Frontend

The frontend is an Electron + React + TypeScript application built with Vite.

- `App.tsx` — main component, manages session state and UI layout
- `useSessionSocket.ts` — custom hook that manages the WebSocket connection, parses events, and exposes reactive state to the UI
- `types.ts` — shared TypeScript types that mirror the Pydantic schemas in `schemas.py`

The frontend connects to `ws://localhost:8000/ws/session/{meeting_id}` when a session starts.
