# Local Meeting Assistant

A platform-agnostic meeting assistant for online and offline scenarios.

This project is designed to work with:
- Zoom
- Microsoft Teams
- BBB
- Google Meet
- and other meeting platforms

It avoids deep platform-specific integrations by using **local audio capture**.

## Core Idea

Instead of building separate real-time integrations for every meeting platform, this project treats the meeting platform as just an audio source.

The system supports two modes:

### 1. Online meeting mode
Capture:
- system audio
- optionally microphone audio

Use cases:
- Zoom meetings
- Teams meetings
- BBB sessions
- online lectures
- remote interviews

### 2. Offline / in-person mode
Capture:
- built-in microphone
- external microphone

Use cases:
- classroom discussion
- office meeting room
- seminar
- interview
- group discussion

## Target Features

- real-time transcription
- speaker diarization
- speaker-aware summaries
- **two live summary layers visible during the session** (see below)
- time-window summaries
- full meeting transcript
- rolling notes
- export transcript and summary
- Chinese / English / German support
- transcript and summary translation among Chinese, English, and German

## Two Live Summary Layers

The session UI is built around **two live summary views** that run side by side with the raw transcript. Both are treated as core MVP features, not post-MVP polish.

### 1. Latest Rolling Summary
- short summary of the most recent **3–5 minutes** of conversation
- refreshes automatically during the session
- always shows the covered time range (for example, `10:32 – 10:36`)
- lets the user catch up on what was just said without scrolling the transcript

### 2. Current Meeting Summary
- compact **cumulative** summary of the meeting from its start to now
- refreshes periodically as the session progresses
- surfaces emerging topics, decisions, and open questions
- lets the user see the overall state of the discussion at a glance

### Why two layers matter
- **Rolling** answers "what did I just miss?"
- **Cumulative** answers "where is this meeting overall?"
- Together they give users both short-term recovery and long-term orientation, without forcing them to read the full transcript.

Both summary panels support:
- **Copy** — copy the current snapshot to the clipboard
- **Expand** — open a full-size view of the snapshot
- **Save to notes** — append the snapshot into the meeting's notes

These two live summaries are kept distinct from:
- the **raw transcript** (source of truth)
- the **final exported summary** (produced at meeting end)

The data model stores each snapshot with a `summary_type` such as `rolling_summary`, `cumulative_meeting_summary`, or `final_summary`, so the history of both live layers is preserved and exportable.

## Product Principles

- platform-agnostic first
- local capture first
- useful MVP over perfect attribution
- original transcript is the source of truth
- translations are derived outputs
- modular pipeline, not tightly coupled to one provider

## Suggested MVP Stack

### Frontend
- Electron
- React
- TypeScript

### Backend
- Python
- FastAPI

### Audio
- system loopback capture (Windows WASAPI loopback via `soundcard` — online meeting mode; macOS/Linux loopback paths planned)
- microphone capture (offline / in-person mode via `sounddevice`, cross-platform)
- WAV replay source for deterministic E2E testing

### Speech-to-text
- faster-whisper or equivalent
- pluggable ASR interface

### Diarization
- pyannote or equivalent
- abstracted speaker service

### Summarization / Translation
- provider-agnostic modules
- support Chinese, English, and German

### Storage
- SQLite
- local export files

## LongCat API configuration

MeetingBro can use LongCat for cloud-based summary and translation while
keeping audio capture and Whisper transcription local. Do **not** paste API
keys into source files; set them in your local shell before starting the
backend.

PowerShell example:

```powershell
$env:LONGCAT_API_KEY="your_api_key_here"
$env:MEETINGBRO_LLM_MODEL="LongCat-Flash-Chat"
```

Optional generic OpenAI-compatible settings:

```powershell
$env:MEETINGBRO_LLM_API_KEY="your_api_key_here"
$env:MEETINGBRO_LLM_BASE_URL="https://api.longcat.chat/openai"
$env:MEETINGBRO_LLM_MODEL="LongCat-Flash-Chat"
```

If no cloud key is configured, the backend falls back to local heuristic
summaries and passthrough translation markers.

Whisper model size is configurable. The backend default is `base`; use `small`
or `medium` for better multilingual Chinese/English/German recognition when
CPU speed is acceptable:

```powershell
$env:MEETINGBRO_WHISPER_SIZE="small"
```

You may also put the same settings in a project-root `.env` file. Both dotenv
syntax and PowerShell-style syntax are accepted:

```text
MEETINGBRO_LLM_API_KEY=your_api_key_here
MEETINGBRO_LLM_BASE_URL=https://api.longcat.chat/openai
MEETINGBRO_LLM_MODEL=LongCat-Flash-Chat
```

```powershell
$env:MEETINGBRO_LLM_API_KEY="your_api_key_here"
$env:MEETINGBRO_LLM_BASE_URL="https://api.longcat.chat/openai"
$env:MEETINGBRO_LLM_MODEL="LongCat-Flash-Chat"
```

## Suggested Repository Structure

```text
.
├── agent.md
├── README.md
├── tasks.md
├── docs/
│   ├── architecture.md
│   ├── decisions.md
│   └── ux-notes.md
├── app/
│   ├── frontend/
│   └── backend/
├── data/
├── exports/
├── scripts/
└── tests/
