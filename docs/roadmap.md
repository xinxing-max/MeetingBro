# tasks.md

## Goal

Build a platform-agnostic local meeting assistant that supports:

- online meeting mode via system audio
- offline meeting mode via microphone
- real-time transcription
- **latest rolling summary** (live, last 3–5 minutes)
- **current meeting summary** (live, cumulative from session start)
- time-window summaries
- full transcript export
- Chinese / English / German support
- transcript and summary translation among Chinese, English, and German

---

## Phase 0 - Setup and planning

### Task 0.1
Read:
- `agent.md`
- `README.md`
- `tasks.md`

### Task 0.2
Create initial repository structure:

```text
docs/
app/frontend/
app/backend/
data/
exports/
scripts/
tests/
```

### Task 0.3
Define shared types / contracts between frontend and backend for:
- transcript segments
- summary snapshots (with `summary_type` field)
- session state events

---

## Phase 1 - Skeleton

### Task 1.1
Create desktop shell (Electron + React + TypeScript) with a single empty session screen.

### Task 1.2
Create local backend service (FastAPI or equivalent) with health check and WebSocket endpoint for session events.

### Task 1.3
Wire frontend to backend over a single session WebSocket carrying transcript and summary events.

### Task 1.4
Implement SQLite schema based on the data model in `agent.md`:
- `meetings`
- `transcript_segments`
- `speakers`
- `summary_snapshots` — must support multiple `summary_type` values, including `rolling_summary`, `cumulative_meeting_summary`, `time_window_summary`, `speaker_summary`, `final_summary`
- `action_items`

---

## Phase 2 - Audio capture and transcription

### Task 2.1
Implement audio capture abstraction with two adapters:
- system audio loopback (online mode) — **Windows-only for now** via `soundcard`'s
  WASAPI loopback device; `sounddevice` 0.5.x does not expose the loopback
  flag, so the loopback path deliberately uses a separate library from the mic
  path. macOS (BlackHole) and Linux (PipeWire) loopback land in a later phase.
  Sources selectable via WS `source=` query param: `mic` / `loopback` /
  `system` / `file:<path>`.
- microphone (offline mode) — cross-platform via `sounddevice`.

### Task 2.2
Implement VAD + chunking layer emitting fixed-duration audio frames.

### Task 2.3
Integrate pluggable ASR provider (default: faster-whisper) behind a clean interface.

### Task 2.4
Emit timestamped `TranscriptSegment` events over the session stream and persist them.

### Task 2.5
Render the **Live Transcript** panel on the session screen with auto-scroll and timestamp display.

---

## Future TODO - Optional Speaker Separation

- Leave speaker separation out of the active MVP until transcript quality is stable.
- Revisit per-speaker labelling later if we have a diarization approach that is measurably reliable on MeetingBro's target audio.
- When revisiting, keep it optional and behind a feature flag rather than enabled by default.

---

## Phase 4 - Live summary layers (core MVP)

### Task 4.1 — Rolling summary generation cadence
Implement a background worker that produces a `rolling_summary` snapshot:
- window: most recent 3–5 minutes of transcript
- cadence: trigger on the earlier of (a) ~60–90 seconds elapsed since last snapshot, or (b) a meaningful new transcript chunk has been added
- input: recent `TranscriptSegment`s within the window
- output: `SummarySnapshot` with `summary_type = "rolling_summary"`, `time_start`, `time_end`, `language`, `content`
- language: follow user's selected summary output language
- the worker must not block the transcription pipeline

### Task 4.2 — Cumulative meeting summary generation cadence
Implement a background worker that produces a `cumulative_meeting_summary` snapshot:
- window: from session start to now
- cadence: approximately every 3–5 minutes, or after significant new content
- input: full transcript so far (or the previous cumulative snapshot plus new segments, to keep prompt size bounded)
- output: `SummarySnapshot` with `summary_type = "cumulative_meeting_summary"`, `time_start = session_start`, `time_end = now`
- should surface emerging topics, decisions, and open questions
- language: follow user's selected summary output language
- must run independently of the rolling summary worker

### Task 4.3 — Summary snapshot persistence
- store each snapshot row in `summary_snapshots`
- keep the full history of rolling and cumulative snapshots for the session
- maintain an `is_latest` flag per `summary_type` so the UI can fetch the live view in O(1)
- ensure snapshots survive a session reload

### Task 4.4 — Session UI layout for both summary panels
Lay out the live session screen so that during a meeting the user sees all of the following without navigating away:

- **Live Transcript** (primary area)
- **Latest Rolling Summary** panel — shows current rolling snapshot + covered time range (e.g. "10:32 – 10:36"); refreshes automatically when a new snapshot arrives
- **Current Meeting Summary** panel — shows current cumulative snapshot; refreshes when a new snapshot arrives
- **Notes / Speakers / Quick Actions** panel

Requirements:
- both summary panels must be visible during the session (not hidden behind tabs by default)
- both panels must be visually distinct from the raw transcript and from each other (e.g. distinct headers, colors, or framing)
- each panel must show its own "last updated" timestamp
- layout must stay usable on a typical laptop screen (≥ 1280 px wide)

### Task 4.5 — Copy / Expand / Save actions
For each of the two summary panels, add three actions:
- **Copy** — copy the current snapshot text to the clipboard
- **Expand** — open a modal or drawer showing the full snapshot plus the previous N snapshots of the same type for history
- **Save to notes** — append the snapshot (with its time range and type) into the meeting's notes

Persist "save to notes" entries on the `meeting.notes` field so they are included in exports.

### Task 4.6 — Event plumbing for live updates
- backend emits `summary_snapshot` events on the session WebSocket whenever a new rolling or cumulative snapshot is produced
- frontend stores the latest snapshot per type in session state and re-renders the matching panel
- panel refresh must not jump the transcript scroll position

---

## Phase 5 - On-demand and final summaries

### Task 5.1
Implement time-window summary on user request (last 5 min / last 15 min / custom range).

### Task 5.2
Keep speaker summary out of the active roadmap for now; revisit only after optional speaker separation is reliable enough to support it.

### Task 5.3
Implement **final summary** at session end, producing a `SummarySnapshot` with `summary_type = "final_summary"`, including key points, decisions, action items, and open questions.

### Task 5.4
Extract action items from the final summary pass and persist them in `action_items`.

---

## Phase 6 - Translation

### Task 6.1
Add provider-agnostic translation module with support for zh / en / de.

### Task 6.2
Apply selected output language to:
- live transcript (view-time translation)
- latest rolling summary
- current meeting summary
- time-window summary
- speaker summary
- final summary

### Task 6.3
Store translations as derived outputs; never overwrite the original transcript.

---

## Phase 7 - Export

### Task 7.1
Export the session as Markdown, TXT, and JSON, including:
- full transcript
- **history of rolling snapshots** (with time ranges)
- **history of cumulative meeting snapshots**
- final summary
- action items
- saved notes

### Task 7.2
Support exporting in original language, translated language, or bilingual format.

---

## Phase 8 - Acceptance testing

### Task 8.1 — Rolling summary acceptance
- start a ≥ 10-minute recorded session
- verify the Latest Rolling Summary panel refreshes at least every ~90 seconds
- verify it always shows a covered time range within the last ~5 minutes
- verify copy / expand / save-to-notes all work
- verify the rolling snapshot history is preserved in storage and export

### Task 8.2 — Cumulative meeting summary acceptance
- in the same session, verify the Current Meeting Summary panel refreshes at least every 3–5 minutes
- verify its `time_start` equals session start and `time_end` advances toward the current time
- verify it reflects content added since the previous cumulative snapshot
- verify copy / expand / save-to-notes all work
- verify the cumulative snapshot history is preserved in storage and export

### Task 8.3 — Summary layer separation
- verify the UI visually distinguishes raw transcript, latest rolling summary, current meeting summary, and the final exported summary
- verify each summary type is stored with the correct `summary_type` value
- verify the final export lists them as separate sections

### Task 8.4 — Multilingual acceptance
- repeat 8.1 and 8.2 with the summary output language set to each of zh, en, de
- verify both live panels render correctly in the selected language

### Task 8.5 — Resilience
- pause and resume capture mid-session; verify both summary workers recover without duplicating or losing snapshots
- simulate a provider error for one summary worker; verify the other continues and the UI surfaces a non-blocking error state

---

## Phase 9 - Polish and non-goals

### Task 9.1
Speaker rename UI, bilingual transcript view, searchable transcript.

### Task 9.2
Error handling, consent / privacy mode prompts, onboarding flow.

### Non-goals for MVP
See `agent.md`:
- no platform-specific bots
- no marketplace apps
- no enterprise admin features
- no calendar sync
