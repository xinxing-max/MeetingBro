# Contributing to MeetingBro

Thank you for your interest! Contributions of all kinds are welcome — bug fixes, documentation improvements, new features, and new ASR or LLM backend integrations.

---

## Getting started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/armpro24-blip/MeetingBro.git
   cd MeetingBro
   ```
3. Follow the [installation steps in README.md](README.md#step-by-step-installation).
4. Create a branch for your change:
   ```bash
   git checkout -b my-feature
   ```

---

## Development setup

### Backend

```bash
conda activate MeetingBro
cd app/backend
pip install -e ".[dev]"     # installs pytest and httpx
meetingbro-backend
```

### Frontend

```bash
cd app/frontend
npm install
npm run dev
```

---

## Verifying your changes

### Dependency check

Before submitting a PR, confirm all dependencies are satisfied:

```bash
python scripts/dep_check.py
```

### List audio devices

Verify MeetingBro can see your audio devices:

```bash
python scripts/list_audio_devices.py
```

### ASR benchmark

If you changed anything related to audio processing or ASR:

```bash
python scripts/benchmark_asr.py data/sample_en.wav --language en
```

Check that `rtf` (realtime factor) stays below 1.0, meaning transcription keeps up with real time.

### WebSocket smoke test

Verify the backend API is healthy (backend must be running):

```bash
python scripts/ws_smoketest.py
```

### Feature verification scripts

For specific features, run the corresponding `verify_*.py` script in `scripts/`:

```bash
python scripts/verify_c1.py     # concurrency guards
python scripts/verify_c5.py     # language vote stickiness
python scripts/verify_c7.py     # stop/cancel race condition
# etc.
```

---

## Pull request guidelines

- **Keep PRs focused.** One concern per PR — a bug fix should not include unrelated refactoring.
- **New config variables** must be added to `.env.example` with a comment explaining what they do.
- **ASR or audio changes** should include benchmark results.
- **Documentation changes** should be clear and follow the same writing style as existing docs.
- **New dependencies** should have a clear justification.

### PR description template

```
## What does this change?

## Why is this change needed?

## How was it tested?
```

---

## Architecture overview

See [docs/architecture.md](docs/architecture.md) for a complete technical description.

Key design points:
- **Two capture modes:** system audio (online meetings) and microphone (in-person)
- **Two ASR layers:** Whisper (formal, accurate) + Qwen3/tiny-Whisper (preview, fast)
- **Two live summaries:** rolling window (last 3–5 min) and cumulative board
- **Modular backends:** ASR, translation, diarization, and summarization all implement abstract base classes — adding a new backend doesn't require touching the session manager

---

## Reporting issues

Please open an issue and include:

- Your OS, Python version, and Node.js version
- Output of `python scripts/dep_check.py`
- The full error output from the backend terminal
- Steps to reproduce the problem

For security issues, please contact the maintainers privately rather than filing a public issue.

---

## Code style

- Python: no strict linter enforced, but follow the conventions in existing files (snake_case, type hints on public functions)
- TypeScript: follow the existing patterns in `App.tsx` and `types.ts`
- Documentation: plain English, short sentences, no unnecessary jargon
