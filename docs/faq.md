# Frequently Asked Questions

## General

### Does MeetingBro record my meetings?

MeetingBro processes audio in real time but does not save audio recordings. The transcript text is saved to a local SQLite database on your machine. Audio is only held in memory for the duration of the current processing window and is never written to disk.

---

### Does any of my audio or transcript leave my computer?

**Audio:** Audio capture and Whisper transcription run entirely on your machine. Audio data is not sent to any external service.

**Transcript text:** only if you configure a cloud LLM API (for summaries and translation). In that case, the text transcript is sent to the LLM provider you choose. If you use a local LLM such as Ollama, nothing leaves your machine at all.

See [docs/llm-providers.md](llm-providers.md) for privacy details per provider.

---

### Does MeetingBro work without an internet connection?

**Transcription:** yes, fully offline once the Whisper model is downloaded.

**Qwen3 preview subtitles:** yes, the model runs locally.

**AI summaries and translation:** only if you use Ollama (local LLM). Cloud providers like OpenAI require internet access.

---

### Is MeetingBro free to use?

MeetingBro itself is free and open source (MIT license). However:
- Some LLM providers charge per API call (e.g., OpenAI). Costs are typically very low for meeting summaries.
- Groq and OpenRouter offer free tiers that are sufficient for casual use. See [docs/llm-providers.md](llm-providers.md).

---

### What languages does MeetingBro support?

MeetingBro currently supports **Chinese (Simplified), English, and German** for transcription, translation, and summaries.

Whisper itself can transcribe many more languages. The translation and summary features are tuned for Chinese/English/German multilingual meetings.

---

## Setup

### Do I need Conda? Can I use pip/venv instead?

Conda is recommended because it makes it easy to manage Python 3.12 alongside other Python installations. You can use a regular `python -m venv` virtual environment if you prefer:

```bash
python3.12 -m venv .venv
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate        # Windows
pip install -e app/backend
```

---

### The Whisper model download is very slow. Can I download it manually?

Yes. The models are downloaded from Hugging Face. You can download them manually using the `huggingface_hub` CLI:

```bash
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-small --local-dir models/faster-whisper-small
```

The backend will use the cached model on the next run.

---

### Can I use a GPU to make transcription faster?

Yes. Set these in your `.env`:

```env
MEETINGBRO_WHISPER_DEVICE=cuda
MEETINGBRO_WHISPER_COMPUTE_TYPE=float16
```

Requirements:
- NVIDIA GPU with CUDA support
- `cuda` toolkit installed
- `faster-whisper` is installed with CUDA support (usually automatic)

You can verify GPU usage with `nvidia-smi` after starting a session.

---

### Can I run the backend and frontend on different machines?

Not out of the box. The frontend connects to `localhost:8000` by default. Running them on separate machines would require changing the WebSocket URL in the frontend config.

---

## Usage

### How long does it take for the first transcript to appear?

On first run, Whisper loads the model into memory. This typically takes 5–15 seconds on a modern CPU. After the model is loaded, transcription latency is 1–5 seconds depending on your settings and hardware.

---

### Why does transcription sometimes produce repeated or incorrect text?

This can happen when:
- The audio signal is very quiet — try increasing your microphone or speaker volume
- The room has significant background noise — try enabling `MEETINGBRO_AUDIO_CONDITIONING_ENABLED=true`
- The Whisper model is too small for the audio complexity — try `MEETINGBRO_WHISPER_SIZE=medium`

---

### What is the difference between "Rolling Summary" and "Meeting Board"?

**Rolling Summary** — a short AI summary of the most recent 3–5 minutes of conversation. It refreshes automatically and answers "what did I just miss?"

**Meeting Board** — a cumulative overview of the entire meeting so far: current topics, decisions made, action items, and open questions. It answers "where is this meeting going overall?"

Both panels can be copied, expanded, or saved to notes.

---

### Can I use MeetingBro with a virtual meeting on macOS?

Yes, but you need a virtual audio cable to capture system audio. **BlackHole** is a popular free option:

1. Install [BlackHole](https://existential.audio/blackhole/)
2. In macOS Audio MIDI Setup, create a Multi-Output Device that includes both BlackHole and your regular speakers
3. Set that Multi-Output Device as your system output
4. In MeetingBro, select BlackHole as the input device

See [docs/platform-support.md](platform-support.md) for detailed steps.

---

### How do I export my meeting notes?

Click **Export Meeting** in the Notes panel. You can choose a save location (the default is a timestamped folder under `exports/`).

The export includes:
- `transcript.md` — full timestamped transcript with translations
- `summary.md` — rolling summary, Meeting Board, final summary, and saved notes
- `metadata.json` — session metadata

---

### The meeting ended but I forgot to export. Can I recover the data?

Yes. All session data is stored in the local SQLite database (`data/meetingbro.db`). You can export any past meeting from the meeting history view, or query the database directly.

---

## Technical

### What is the difference between "preview" and "formal" transcription?

MeetingBro runs two ASR paths in parallel:

- **Preview (fast subtitles):** uses a small, fast model (Qwen3 or tiny-Whisper). Results appear within 0.5–1 second. These are displayed immediately in the transcript as subtitles.
- **Formal (accurate):** uses the main Whisper model (small/medium). Results appear 2–5 seconds later. These replace the preview text with higher-accuracy transcription.

---

### How does MeetingBro handle mixed-language meetings?

By default, Whisper auto-detects the language of each audio segment. A language-vote mechanism prevents rapid oscillation (e.g., switching between "zh" and "en" on every segment). You can lock to a single language with:

```env
MEETINGBRO_LANGUAGE_LOCK_ENABLED=true
```

---

### Where is my data stored?

| Data | Location |
|---|---|
| Transcript and sessions | `data/meetingbro.db` (SQLite) |
| Meeting exports | `exports/` (Markdown files) |
| Whisper models | Hugging Face cache (system-level) |
| Qwen3 model | `models/` (project directory) |
| Configuration | `.env` (project root) |

All data is local. Nothing is synced to any cloud service.
