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

### 2. Meeting Board
- compact **cumulative** meeting-state view from the start of the meeting to now
- refreshes periodically as the session progresses
- surfaces the current state through **topics, decisions, action items, and open questions**
- stays more stable than the rolling recap so users can orient themselves at a glance

### Why two layers matter
- **Rolling** answers "what did I just miss?"
- **Meeting Board** answers "where is this meeting overall?"
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

Whisper model size is configurable. The backend default is `medium`; use
`small` for lower CPU usage or keep/increase `medium` for better multilingual
Chinese/English/German recognition when CPU speed is acceptable:

```powershell
$env:MEETINGBRO_WHISPER_SIZE="small"
```

## ASR performance / robustness tuning

The default runtime mode is **Balanced**: local Whisper on CPU, pre-ASR VAD
enabled, light audio conditioning enabled, early flush enabled, and language
lock disabled so mixed Chinese/English/German meetings can be auto-detected
segment by segment.

Runtime modes are available in the UI and through the WebSocket
`runtime_profile` parameter:

- `balanced` — default real-meeting trade-off between latency and stability.
- `low_latency` — faster first preview/commit; may fragment more on long/noisy audio.
- `robust` — longer context and less aggressive commits for noisy/long meetings.
- `multilingual` — balanced settings with language lock explicitly off.
- `single_language` — balanced settings with language lock on for mostly one-language meetings.

Useful local `.env` options:

```text
# Faster or GPU-capable machines
MEETINGBRO_RUNTIME_PROFILE=balanced
MEETINGBRO_WHISPER_SIZE=medium
MEETINGBRO_WHISPER_DEVICE=cpu
MEETINGBRO_WHISPER_COMPUTE_TYPE=int8
MEETINGBRO_WHISPER_CPU_THREADS=0
MEETINGBRO_WHISPER_NUM_WORKERS=1

# Noise / interruption handling
MEETINGBRO_AUDIO_CONDITIONING_ENABLED=true
MEETINGBRO_AUDIO_CONDITIONING_TARGET_RMS=0.035
MEETINGBRO_PRE_VAD_ENABLED=true
MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED=true
MEETINGBRO_PRE_VAD_CONDITIONING_TARGET_RMS=0.03
MEETINGBRO_PRE_VAD_CONDITIONING_MIN_RMS=0.001
MEETINGBRO_PRE_VAD_CONDITIONING_MAX_GAIN=4.0
MEETINGBRO_PRE_VAD_THRESHOLD=0.38
MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD=0.005
MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS=8
MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED=true
MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN=0.0008
MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX=0.02
MEETINGBRO_WEAK_SPEECH_RESCUE_WINDOW_SECONDS=6
MEETINGBRO_WEAK_SPEECH_RESCUE_COOLDOWN_SECONDS=8
MEETINGBRO_WHISPER_VAD_THRESHOLD=0.3
MEETINGBRO_ASR_RETRY_ENABLED=true
MEETINGBRO_ASR_SAFEGUARD_ENABLED=true
MEETINGBRO_ASR_SAFEGUARD_RTF_THRESHOLD=0.9
MEETINGBRO_ASR_SAFEGUARD_COOLDOWN_WINDOWS=5

# Mixed-language meetings: keep false unless one dominant language should be forced
MEETINGBRO_LANGUAGE_LOCK_ENABLED=false

# Live subtitle translation: keep newest segments responsive
MEETINGBRO_TRANSLATION_EXECUTOR_WORKERS=2
MEETINGBRO_LIVE_TRANSLATION_BACKFILL_LIMIT=20
MEETINGBRO_LIVE_TRANSLATION_MAX_PENDING=12
MEETINGBRO_LIVE_TRANSLATION_SAFEGUARD_MAX_PENDING=4
```

For noisy rooms, try slightly increasing `MEETINGBRO_WHISPER_VAD_THRESHOLD`
(`0.35`â€“`0.45`). For very quiet speakers, prefer raising microphone/system
gain first; then lower `MEETINGBRO_PRE_VAD_THRESHOLD` toward `0.3` and
`MEETINGBRO_WHISPER_VAD_THRESHOLD` toward `0.25`. If realtime transcription
falls behind on CPU, try `MEETINGBRO_WHISPER_SIZE=small`, reduce
`MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS` to `4`, or temporarily set
`MEETINGBRO_ASR_RETRY_ENABLED=false`.

## ASR benchmark

Use the local ASR benchmark before/after tuning so accuracy and realtime changes
are visible instead of purely subjective. The benchmark is LLM-free: it does not
call LongCat/OpenAI for summary or translation.

```powershell
python scripts\benchmark_asr.py data\sample_en.wav --language en
python scripts\benchmark_asr.py data\*.wav --language auto --model-size small --json-out exports\asr_benchmark.json
python scripts\benchmark_asr.py data\sample_en.wav --keywords Alice Bob budget timeline
```

Important columns:
- `rtf`: total pipeline wall time divided by audio duration; lower is faster.
- `asr_rtf`: latest ASR window realtime factor.
- `safe`: ASR safeguard trigger count.
- `rescue`: weak-voice rescue emitted/attempted count.
- `kw`: matched/expected keyword count when `--keywords` is provided.

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
