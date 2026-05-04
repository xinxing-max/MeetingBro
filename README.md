# MeetingBro

**An open-source, local-first AI meeting assistant.**

MeetingBro listens to your meetings, transcribes speech in real time, translates between languages, and generates live summaries — all without requiring any platform plugin or integration.

It works with Zoom, Teams, Google Meet, BBB, and any other platform, because it captures audio directly from your computer rather than connecting to a meeting API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

<video src="https://github.com/user-attachments/assets/54111a42-2e80-416f-9ee5-1402a68e3cd7" autoplay loop muted playsinline width="100%"></video>

---

## What can MeetingBro do?

| Feature | Details |
|---|---|
| **Live transcription** | Real-time speech-to-text using Whisper, running locally on your machine |
| **Live subtitles** | Fast preview subtitles with ultra-low latency (Qwen3 or tiny-Whisper) |
| **Live translation** | Automatic translation between Chinese, English, and German |
| **Rolling summary** | AI-generated summary of the last 3–5 minutes, refreshes automatically |
| **Meeting Board** | Cumulative overview: topics, decisions, action items, open questions |
| **Exportable notes** | Export full transcript, summary, and notes to Markdown files |
| **System audio capture** | Capture any meeting platform's audio (Windows) |
| **Microphone capture** | Works on Windows, macOS, and Linux (in-person meetings) |
| **Local or cloud LLM** | Use OpenAI, Groq, Ollama, or any OpenAI-compatible API for summaries |

> **No API key is required to use transcription.** Summaries fall back to local heuristics if no LLM is configured.

---

## Who is this for?

- Students attending online lectures who want automatic notes
- Professionals who attend meetings in a second language
- Researchers who need multilingual transcripts
- Privacy-conscious users who want local-first AI tools
- Developers who want to build on or extend a local meeting assistant

---

## Before you start

**Please read this before installing.**

- You will need to type a few commands in a terminal (command prompt). The instructions are step-by-step.
- **Windows is the easiest platform** right now. On Windows, MeetingBro can capture system audio (everything your speakers play), so it works with any meeting platform automatically.
- On **macOS and Linux**, microphone capture works but system audio capture is not yet supported. You can still use MeetingBro for in-person meetings or with a virtual audio cable.
- **Whisper models download automatically on first run.** The `small` model is about 460 MB. This happens once and is saved to your disk.
- **The Qwen3 preview model is optional** (about 700 MB). It makes subtitles appear faster. You can skip it and use Whisper-only mode.
- **An LLM API key is optional.** Without one, transcription still works perfectly. You only need a key for AI-generated summaries and translation. A free-tier key from Groq or OpenRouter is enough to get started.

---

## Quick Start

This is the shortest path to a working session. See [Step-by-step installation](#step-by-step-installation) below for more detail.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/MeetingBro.git
cd MeetingBro

# 2. Create a Python environment
conda create -n MeetingBro python=3.12 -y
conda activate MeetingBro

# 3. Install the backend
cd app/backend
pip install -e "."
pip install "soundcard>=0.4"   # Windows only — skip on macOS/Linux
cd ../..

# 4. Copy the config template
cp .env.example .env           # On Windows: copy .env.example .env

# 5. Install the frontend
cd app/frontend
npm install
cd ../..
```

Then open **two terminals** and run:

**Terminal 1:**
```bash
conda activate MeetingBro
cd app/backend
meetingbro-backend
```

**Terminal 2:**
```bash
cd app/frontend
npm run dev
```

The Electron window opens. Select your audio device and click **Start Session**.

---

## Step-by-step installation

### Requirements

| Tool | Version | How to get it |
|---|---|---|
| **Python** | 3.12 or later | [python.org](https://www.python.org/downloads/) or via Conda |
| **Conda** | any recent | [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) (Miniconda recommended) |
| **Node.js** | 18 or later | [nodejs.org](https://nodejs.org/) |
| **Git** | any | [git-scm.com](https://git-scm.com/) |

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MeetingBro.git
cd MeetingBro
```

### Step 2 — Create a Python environment

```bash
conda create -n MeetingBro python=3.12 -y
conda activate MeetingBro
```

You will need to run `conda activate MeetingBro` every time you open a new terminal to work on MeetingBro.

### Step 3 — Install the backend

```bash
cd app/backend
pip install -e "."
```

**Windows only** — also install the system audio loopback library:
```bash
pip install "soundcard>=0.4"
```

This lets MeetingBro capture audio from Zoom, Teams, and other apps without any plugin.

Go back to the project root:
```bash
cd ../..
```

### Step 4 — Configure

Copy the configuration template:
```bash
cp .env.example .env       # macOS / Linux
copy .env.example .env     # Windows Command Prompt
```

Open `.env` in any text editor. The defaults work without changes, but if you want AI summaries, add your LLM key (see [LLM setup](#llm-setup-for-ai-summaries) below).

### Step 5 — (Recommended) Download the Qwen3 preview model

This step is optional but recommended. The Qwen3 model makes live subtitles appear faster.

```bash
pip install sherpa-onnx

python -c "
import sherpa_onnx
sherpa_onnx.download_model('sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', dest='models/')
"
```

The model is about 700 MB and downloads once.

To skip this step and use Whisper-only preview instead, add this to your `.env`:
```env
MEETINGBRO_PREVIEW_ASR_BACKEND=whisper
```

### Step 6 — Install the frontend

```bash
cd app/frontend
npm install
cd ../..
```

---

## LLM setup for AI summaries

MeetingBro uses an **optional** cloud or local LLM for:
- AI meeting summaries (Rolling Summary and Meeting Board)
- Translation between Chinese, English, and German

**Without a key:** transcription still works. Summaries fall back to keyword extraction.

**With a key:** set these three lines in your `.env` file:

```env
MEETINGBRO_LLM_API_KEY=your_api_key_here
MEETINGBRO_LLM_BASE_URL=https://api.openai.com/v1
MEETINGBRO_LLM_MODEL=gpt-4o-mini
```

Supported providers include **OpenAI, Groq, Mistral AI, OpenRouter, Together AI**, and **Ollama** (fully local — no key, no data sent anywhere).

→ **See [docs/llm-providers.md](docs/llm-providers.md) for full setup instructions for each provider, including how to get a free key.**

> **Privacy note:** Only the text transcript is sent to the LLM. Audio capture and Whisper transcription run entirely on your machine. No audio is ever transmitted.

---

## Running MeetingBro

Always start the backend first, then the frontend.

### Backend

```bash
conda activate MeetingBro
cd app/backend
meetingbro-backend
```

You should see output like:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Frontend

In a new terminal:

```bash
cd app/frontend
npm run dev
```

The Electron window opens automatically. If it does not, open your browser and go to `http://localhost:5173`.

---

## First successful run checklist

Before starting a session, verify:

- [ ] Backend is running (see terminal output above)
- [ ] Electron window has opened
- [ ] Audio device is selected in the UI
- [ ] For online meetings (Windows): select "System Audio" or the loopback device
- [ ] For in-person meetings: select your microphone

Click **Start Session** and speak a few words. You should see text appear in the transcript panel within a few seconds.

If the Whisper model hasn't been used before, the first transcription may take 10–30 seconds while the model loads. Subsequent transcriptions are faster.

---

## How it works

MeetingBro captures audio from your computer and processes it through a local pipeline:

```
Audio source (mic or system audio)
    ↓ Voice Activity Detection (VAD)
    ↓ Whisper ASR  ←→  Qwen3 preview (fast subtitles)
    ↓ Speaker diarization (optional)
    ↓ Translation (optional, via LLM)
    ↓ Summarization (optional, via LLM)
    ↓ Live UI (transcript + summaries)
    ↓ Export (Markdown files)
```

Two audio capture modes:
- **Online mode** (Windows): captures system audio — works with any meeting platform
- **Offline mode** (all platforms): captures microphone — works for in-person meetings

See [docs/architecture.md](docs/architecture.md) for a full technical overview.

---

## Troubleshooting

Common issues and solutions are in **[docs/troubleshooting.md](docs/troubleshooting.md)**.

Quick checks:
1. Run `python scripts/dep_check.py` to verify all dependencies are installed.
2. Run `python scripts/list_audio_devices.py` to see available audio devices.
3. Make sure the backend is running before starting the frontend.

---

## Platform support

| Platform | Microphone | System audio | Notes |
|---|---|---|---|
| Windows 10/11 | ✅ | ✅ | Full support |
| macOS | ✅ | ⚠️ | System audio needs a virtual audio cable (e.g. BlackHole) |
| Linux | ✅ | ⚠️ | System audio via PulseAudio/PipeWire loopback (manual setup) |

See [docs/platform-support.md](docs/platform-support.md) for detailed instructions per platform.

---

## Frequently asked questions

See **[docs/faq.md](docs/faq.md)** for answers to common questions.

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to set up a development environment, run tests, and submit pull requests.

---

## Documentation index

| Document | Contents |
|---|---|
| [README.md](README.md) | This file — overview and quick start |
| [docs/llm-providers.md](docs/llm-providers.md) | How to get and configure an LLM API key |
| [docs/platform-support.md](docs/platform-support.md) | OS-specific audio setup |
| [docs/architecture.md](docs/architecture.md) | Technical architecture and component overview |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common problems and solutions |
| [docs/faq.md](docs/faq.md) | Frequently asked questions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and contribution guide |
| [agent.md](agent.md) | Product and engineering principles (for contributors) |

---

## License

MIT — see [LICENSE](LICENSE).
