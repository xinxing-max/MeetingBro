# agent.md

## Rule zero

- Think it through before you start. Don’t make assumptions or hide confusion. If something is genuinely ambiguous and blocks correct implementation, surface the ambiguity explicitly instead of guessing.
- Simplicity first. If you can solve it in 50 lines, don’t use 200. Don’t add features that were not explicitly requested.
- Make precise changes. Change only what needs to be changed. Don’t refactor just because it feels convenient, and don’t rename adjacent code while fixing a focused issue.
- Be goal-driven. Translate “fix it” into “write a test that reproduces the problem, then make the test pass.”

## Project

Build a cross-platform meeting assistant for ordinary students and employees.

The system should work with **Microsoft Teams, Zoom, BBB, and potentially any other meeting platform** by relying on **local audio capture and local meeting context capture**, rather than deep platform-specific real-time media integrations.

The product goal is to let a user open an app during a meeting and get:

1. **Real-time note taking**
2. **Speaker-aware summaries**
3. **Two live summary layers shown during the session**
   - **Latest rolling summary** covering the most recent 3–5 minutes
   - **Current meeting summary** covering the whole session so far
4. **Conversation summaries for a selected time window**
5. **A full meeting transcript**
6. **A simple app-like experience that works regardless of meeting software**
7. **Chinese, English, and German language support**
8. **Transcript and summary conversion among Chinese, English, and German**

### Two live summary layers (core MVP requirement)

The session UI must include two live summary views, both visible during the meeting:

**1. Latest rolling summary**
- short rolling summary of the most recent 3–5 minutes of conversation
- updated continuously during the session
- must clearly show the covered time range (e.g. "10:32 – 10:36")
- refreshed automatically whenever a new rolling snapshot is produced
- visually distinct from raw transcript and from the cumulative summary

**2. Current meeting summary**
- compact cumulative summary of the meeting from its start to now
- updated periodically as the session progresses
- helps the user understand the overall discussion state at a glance
- should surface emerging topics, decisions, and open questions
- visually distinct from both the raw transcript and the rolling summary

Both summary views must:
- be treated as core MVP features, not post-MVP enhancements
- support UI actions: **copy**, **expand**, and **save to notes**
- be clearly separated from the raw transcript, the rolling summary, the cumulative meeting summary, and the final exported summary

Reference inspiration: products like **Meetscribe**, but the design should prioritize **platform independence via local capture**.

---

## Core Product Assumption

Use **two capture modes** as the primary architecture.

### Mode A: Online meeting mode

Use **local system audio capture** and optionally microphone capture.

This mode is for:
- Zoom
- Microsoft Teams
- BBB
- Google Meet
- Tencent Meeting
- Feishu / Lark / Slack Huddles
- or any other meeting platform that outputs audio through the user's device

Main idea:
- Capture meeting audio from the user's computer
- Avoid relying on deep platform-specific bot integrations for the core MVP
- Treat the meeting platform as interchangeable

### Mode B: Offline / in-person meeting mode

Use **local device microphone capture** to record in-room speech and generate the same outputs.

This mode is for:
- classroom discussion
- office meeting room
- interview
- seminar
- study group
- any face-to-face conversation near the device

Main idea:
- Use the user's laptop or mobile device as the recording endpoint
- Capture ambient speech through microphone input
- Run the same transcription, diarization, note-taking, summarization, and translation pipeline

This two-mode architecture is important because it changes feasibility significantly:

- **Platform compatibility becomes much easier**
- **The same core pipeline can support both online and offline usage**
- **MVP scope becomes realistic while still covering more real-world scenarios**
- **Engineering risk shifts from platform APIs to audio quality, diarization, multilingual processing, and UX**

---

## Difficulty Assessment

### Overall

This is a **medium-difficulty MVP** and a **hard production-grade product**.

### Why the local-source approach helps

If we tried to integrate deeply with each meeting platform, this would be a high-complexity, long-cycle infrastructure project because of:

- Different APIs and permissions
- Different real-time media access models
- Different tenant restrictions
- Different app review and deployment rules

Using **two local capture modes** removes most of that platform-specific complexity.

- **Online meeting mode** handles virtual meetings by capturing device audio
- **Offline meeting mode** handles in-person meetings by capturing microphone audio

This greatly expands use cases without requiring separate platform integrations.

### Difficulty by subsystem

#### 1. Local audio capture
**Difficulty: Medium**

Need to support two input paths:
- system audio for online meetings
- microphone audio for offline meetings
- optional mixed mode using both

Challenges:
- OS-specific handling (Windows/macOS/Linux)
- loopback audio support differs by OS
- echo and duplication management
- handling headphones vs speakers
- room noise and distance-to-microphone issues in offline mode

#### 2. Real-time transcription
**Difficulty: Medium**

This is solvable with existing ASR models/APIs, but tradeoffs exist between:
- speed
- cost
- accuracy
- multilingual support
- on-device vs cloud

#### 3. Speaker identification / diarization
**Difficulty: Medium to Hard**

This is one of the hardest parts in practice.

Need to distinguish:
- local user
- remote speakers
- possibly multiple remote speakers mixed into a single device output
- multiple in-room speakers in offline recordings

Important limitation:
With only a mixed local output stream or a single room microphone, perfect speaker attribution is difficult. The system should support **best-effort diarization** and expose confidence levels.

#### 4. Real-time note generation and summarization
**Difficulty: Medium**

This is relatively straightforward once timestamped transcript chunks are available.

Need:
- incremental summarization
- **rolling summary** over the most recent 3–5 minutes, refreshed continuously
- **cumulative current meeting summary**, refreshed periodically
- speaker-level summaries
- time-window summaries
- meeting-level recap (final summary)

The rolling summary and cumulative meeting summary are distinct pipeline outputs with different cadences and different scopes. Both must be produced and surfaced live.

#### 5. Translation and language conversion
**Difficulty: Medium**

The system must support Chinese, English, and German, plus conversion among them.

Challenges:
- language detection
- mixed-language sessions
- code-switching
- preserving timestamps and speaker structure across translation
- keeping summaries faithful across languages

#### 6. Full transcript generation
**Difficulty: Easy to Medium**

Once ASR pipeline is stable, transcript storage and export are straightforward.

#### 7. Cross-platform product UX
**Difficulty: Medium**

Need a workflow simple enough for ordinary users:
- open app
- choose mode
- choose audio source
- choose language settings
- start meeting capture
- see live transcript and notes
- translate output if needed
- export results

---

## Product Decision

For MVP, prioritize **two platform-agnostic local capture modes** over official integrations.

### Mode definitions

#### Online meeting mode
Use local system audio capture to work with Teams, Zoom, BBB, and similar meeting tools.

#### Offline meeting mode
Use device microphone capture to work in classrooms, meeting rooms, and face-to-face discussions.

Both modes should feed into the same downstream pipeline:
- transcription
- diarization
- speaker summaries
- time-window summaries
- final transcript and notes
- translation

### Non-goals for MVP

Do **not** start with:
- platform bots joining meetings
- official Teams/Zoom marketplace apps
- deep BBB server plugins
- organization-wide admin deployment
- calendar automation
- enterprise compliance workflows

These can be later phases.

---

## Primary User Stories

### User story 1
As a student, I want to start the app before or during an online lecture, so I can get a full transcript and concise notes afterward.

### User story 2
As an employee, I want the app to listen to my online meeting locally and generate notes in real time without depending on whether the company allows meeting bots.

### User story 3
As a student or employee, I want to place my device in a physical room and capture an offline discussion, class, or meeting with the same transcript and summary features.

### User story 4
As a user, I want to know who spoke and what each person mainly said.

### User story 5
As a user, I want a summary of the conversation during a selected time range, such as the last 5 minutes or 10:30–10:45.

### User story 6
As a user, I want the system to support Chinese, English, and German speech.

### User story 7
As a user, I want to translate transcripts and summaries among Chinese, English, and German.

### User story 8
As a user, I want to export a complete transcript and meeting summary after the meeting.

---

## Functional Requirements

### FR-1 Audio ingestion

The app must support two capture modes.

#### Online meeting mode
The app must be able to ingest one or more of the following:
- system audio
- microphone audio
- virtual loopback audio

#### Offline meeting mode
The app must be able to ingest:
- built-in microphone
- external microphone
- uploaded audio recording as fallback

### FR-2 Live transcription

The app must produce timestamped transcript chunks in near real time.

The transcription pipeline must support:
- Chinese
- English
- German
- mixed-language sessions where feasible

### FR-3 Speaker diarization

The app must assign transcript segments to inferred speakers such as:
- Speaker 1
- Speaker 2
- Speaker 3

If confidence is sufficient, optionally allow user labeling:
- Alice
- Bob
- Me

### FR-4 Speaker summaries

The app must generate a summary of each identified speaker's remarks.

The user must be able to request the speaker summary in:
- Chinese
- English
- German

### FR-5 Time-window summaries

The app must generate a summary for a user-selected time period.

Examples:
- last 5 minutes
- last 15 minutes
- custom timestamp range

The user must be able to request the time-window summary in:
- Chinese
- English
- German

### FR-6 Meeting summary

The app must generate an overall meeting summary, including:
- key points
- decisions
- action items
- open questions

The user must be able to request the meeting summary in:
- Chinese
- English
- German

### FR-6a Latest rolling summary (live)

The app must continuously generate and display a **rolling summary** of the most recent 3–5 minutes of conversation during the live session.

- refresh cadence: automatic, approximately every 60–90 seconds or whenever enough new transcript material accumulates
- must display the covered time range alongside the text
- must be visually distinct from the raw transcript and from the cumulative meeting summary
- must remain available for copy, expand, and save-to-notes actions
- language must follow the user's selected summary output language (Chinese / English / German)

### FR-6b Current meeting summary (live cumulative)

The app must periodically generate and display a **cumulative summary of the meeting so far**, spanning from session start to the current moment.

- refresh cadence: periodic, approximately every 3–5 minutes or after significant new content
- must be compact enough to read at a glance, while reflecting the overall state of discussion
- should surface emerging topics, decisions, and open questions
- must be visually distinct from the raw transcript and from the rolling summary
- must remain available for copy, expand, and save-to-notes actions
- language must follow the user's selected summary output language (Chinese / English / German)

### FR-6c Summary layer separation

The system must clearly distinguish four outputs as separate artifacts, both in the data model and in the UI:

1. **Raw transcript** — timestamped speech segments, source of truth
2. **Latest rolling summary** — short, recent-window live view
3. **Cumulative current meeting summary** — compact whole-session live view
4. **Final exported summary** — produced at session end, may be longer and more structured than the live cumulative summary

### FR-7 Full transcript

The app must provide a complete transcript with timestamps and speaker labels.

The user must be able to:
- view the original transcript language
- translate the transcript into Chinese, English, or German
- optionally display original and translated transcript side by side

### FR-8 Live notes panel

The app should show rolling notes during the meeting.

The user should be able to choose the rolling notes language:
- Chinese
- English
- German

### FR-9 Translation and conversion

The app must support conversion among:
- Chinese -> English
- Chinese -> German
- English -> Chinese
- English -> German
- German -> Chinese
- German -> English

The app should support:
- transcript translation
- summary translation
- speaker summary translation
- time-window summary translation

### FR-10 Export

The app should export:
- transcript
- summary
- action items
- speaker summaries

Exports should support:
- original language output
- translated output in Chinese, English, or German
- bilingual output where practical

### FR-11 Resilience

The app should tolerate:
- temporary audio interruptions
- ASR delays
- diarization uncertainty
- translation delays
- user pausing and resuming capture

---

## Non-Functional Requirements

### NFR-1 Platform coverage

Must work independently of Teams, Zoom, BBB, and similar platforms as long as audio is audible on the device.

### NFR-2 Privacy modes

Support multiple privacy modes:
- local-only processing where feasible
- cloud transcription mode
- user-visible consent and recording notices

### NFR-3 Latency

Target live transcript latency of roughly 1–5 seconds for MVP.

### NFR-4 Accuracy transparency

Show that speaker attribution, transcription, summaries, and translations may be approximate.

### NFR-5 Simplicity

The onboarding flow should be simple enough for non-technical users.

### NFR-6 Language flexibility

Do not hard-code the language pipeline around a single provider or model.

---

## Key Technical Reality

### Important limitation: speaker attribution is imperfect

If all remote participants are mixed into one output channel, or if multiple in-room voices are captured by one microphone, diarization can separate voices probabilistically, but it cannot always map them perfectly to real participant identities.

Therefore the MVP should distinguish between:

1. **Speaker segmentation**: separating different voices
2. **Speaker naming**: assigning those voices to actual people

Speaker segmentation is feasible.
Speaker naming may require:
- user confirmation
- participant list matching
- optional voice enrollment
- platform metadata in later versions

The implementation should preserve this distinction in system design and UI copy.

### Important limitation: translation should preserve original meaning, not overwrite source truth

The system should treat the **original transcript** as the source record.
Translations and summaries should be stored as derived outputs.

This matters because:
- ASR may already contain uncertainty
- translation may add a second layer of approximation
- bilingual export should preserve the original transcript when possible

---

## Recommended Architecture

### High-level architecture

1. **Desktop or local app**
   - captures local audio sources
   - shows live transcript and notes
   - allows language and output selection

2. **Audio processing layer**
   - buffering
   - VAD (voice activity detection)
   - chunking
   - optional denoise / echo handling

3. **Speech-to-text layer**
   - real-time ASR
   - timestamps per chunk
   - language detection or forced-language mode

4. **Speaker diarization layer**
   - identifies speaker turns
   - assigns stable speaker IDs where possible

5. **Notes and summarization layer**
   - latest rolling summary (last 3–5 minutes, refreshed continuously)
   - current meeting summary (cumulative, refreshed periodically)
   - speaker summaries
   - time-window summaries
   - final meeting recap (exported at end of session)
   - the layer must emit distinct snapshot types so the UI and storage can tell them apart

6. **Translation layer**
   - transcript translation
   - summary translation
   - bilingual formatting

7. **Storage layer**
   - transcript chunks
   - speaker segments
   - summary snapshots
   - translation snapshots
   - meeting metadata

8. **Export layer**
   - markdown
   - txt
   - json
   - optional docx/pdf later

---

## Suggested MVP Stack

The implementation should optimize for delivery speed and clarity, not theoretical perfection.

### Option A: Electron desktop app + Python backend
Recommended for MVP.

#### Frontend
- Electron
- React
- TypeScript

#### Local backend
- Python FastAPI or lightweight local service

#### Audio capture
- OS loopback / virtual audio device support
- microphone capture
- ffmpeg or native audio libraries where appropriate

#### ASR
Choose one path and keep abstraction clean:
- faster-whisper / Whisper-based local inference
- cloud ASR provider
- pluggable ASR interface

#### Diarization
- pyannote or equivalent diarization pipeline
- abstract behind a speaker service interface

#### Summarization and translation
- provider-agnostic summarization module
- provider-agnostic translation module
- incremental summarization over timestamped transcript chunks

#### Storage
- SQLite for MVP
- local files for transcript exports

---

## Language Support

The system must support **Chinese, English, and German** as first-class languages.

### Language requirements
- detect spoken language at session level and segment level where feasible
- transcribe Chinese, English, and German accurately enough for meeting use
- support mixed-language meetings
- allow summaries in Chinese, English, or German
- allow transcript translation between Chinese, English, and German
- allow summary translation between Chinese, English, and German

### Conversion requirements

The system should support mutual conversion among:
- Chinese -> English
- Chinese -> German
- English -> Chinese
- English -> German
- German -> Chinese
- German -> English

### UX requirements for language

The user should be able to choose:
- transcription language mode: auto / Chinese / English / German
- output summary language
- output transcript language
- whether to show original transcript only, translated transcript only, or both

### Engineering note

The implementation should separate these stages:
1. speech capture
2. transcription
3. speaker diarization
4. summarization
5. translation
6. export

Translation should not be tightly coupled to one ASR or one LLM provider.

---

## Suggested Data Model

### Meeting
- id
- title
- start_time
- end_time
- source_type
- detected_languages
- preferred_summary_language
- preferred_transcript_language
- notes

### TranscriptSegment
- id
- meeting_id
- start_time
- end_time
- text
- original_language
- translated_text_zh
- translated_text_en
- translated_text_de
- speaker_id
- confidence
- raw_asr_payload

### Speaker
- id
- meeting_id
- display_name
- inferred_label
- confidence
- is_local_user

### SummarySnapshot
- id
- meeting_id
- summary_type — one of:
  - `rolling_summary` (latest 3–5 minute window)
  - `cumulative_meeting_summary` (from session start to `time_end`)
  - `time_window_summary` (user-requested range)
  - `speaker_summary`
  - `final_summary` (produced at session end / export)
- time_start
- time_end
- language
- content
- source_segment_ids (optional, for traceability back to transcript chunks)
- is_latest (boolean, true for the most recent snapshot of a given type; used by the UI to render the "live" view)
- created_at

The schema must be able to hold multiple snapshots of each type for the same meeting, so historical rolling and cumulative snapshots can be retained and inspected.

### ActionItem
- id
- meeting_id
- assignee
- text
- confidence

---

## MVP Scope

The implementation should prioritize the following MVP features first.

### Must-have
- start / stop recording session
- choose online or offline mode
- choose audio source
- choose language settings
- real-time transcript panel
- approximate speaker diarization
- **latest rolling summary panel (live, last 3–5 minutes, with time range shown)**
- **current meeting summary panel (live, cumulative from session start)**
- both summary panels must support copy, expand, and save-to-notes actions
- final meeting summary (exported at session end)
- time-range summary (on user request)
- Chinese / English / German support
- transcript and summary translation among Chinese, English, and German
- export transcript and summary

### Nice-to-have
- rename speakers manually
- action item extraction
- chapter / topic segmentation
- searchable transcript
- side-by-side bilingual transcript view

### Defer
- deep Zoom/Teams/BBB SDK integrations
- meeting invite link parsing
- calendar sync
- cloud multi-user collaboration
- enterprise admin features

---

## UX Requirements

### MVP workflow

1. User opens the app
2. User chooses mode:
   - Online meeting mode
   - Offline / in-person mode
3. User selects input source:
   - system audio
   - microphone
   - both
4. User selects language settings:
   - auto detect / Chinese / English / German
   - summary output language
   - transcript output language
5. User clicks **Start meeting notes**
6. User sees the live session screen, which conceptually contains:
   - **Live Transcript** panel — timestamped speech with speaker labels
   - **Latest Rolling Summary** panel — short summary of the last 3–5 minutes, with covered time range, refreshing automatically
   - **Current Meeting Summary** panel — compact cumulative summary from session start, refreshing periodically
   - **Notes / Speakers / Quick Actions** panel — manual notes, speaker list, and action buttons
7. Each summary panel supports:
   - **copy** — copy text to clipboard
   - **expand** — open a larger view of the full snapshot
   - **save to notes** — append the snapshot into the user's notes for this meeting
8. User can request:
   - summarize last 5 minutes
   - summarize this speaker
   - show full transcript
   - translate transcript or summary
9. User clicks **End meeting**
10. User exports results, including transcript, rolling/cumulative snapshot history, and the final summary

### UX notes by mode

#### Online meeting mode
The user should not need to care which meeting platform is running.

#### Offline meeting mode
The user should be able to place the device on a desk or in a room and start capture with minimal setup.

### UX principle
The user should not need to care which meeting platform is running, and should not need to understand the internal transcription or translation pipeline.

---

## Engineering Risks

### Risk 1: OS-specific audio loopback

Different operating systems expose system audio differently.

Mitigation:
- design an audio adapter layer
- implement Windows first if necessary
- document loopback requirements clearly

### Risk 2: poor diarization in mixed audio

Remote participants may be hard to separate in online mode, and overlapping voices may be hard to separate in offline room recordings.

Mitigation:
- label speakers generically by default
- allow manual renaming
- expose confidence
- focus on usefulness over identity perfection
- recommend better microphone placement in offline mode

### Risk 3: latency vs accuracy tradeoff

More accurate models may be slower.

Mitigation:
- use streaming chunks
- separate live rough transcript from later refinement
- optionally reprocess at meeting end

### Risk 4: multilingual accuracy

Code-switching, accent variation, and domain-specific vocabulary can reduce quality in Chinese, English, and German.

Mitigation:
- support auto-detect and forced-language modes
- store original transcript before translation
- allow later retranslation and post-processing
- keep language modules replaceable

### Risk 5: privacy expectations

Users may not understand where audio is processed.

Mitigation:
- explicit privacy mode selection
- clear local vs cloud labels
- clear consent prompts

---

## Implementation Guidance

The implementation should work in phases.

### Phase 1: skeleton
Create:
- project structure
- desktop shell or local app shell
- audio capture abstraction
- transcript event pipeline
- local persistence
- language settings and translation abstraction

### Phase 2: transcription
Implement:
- audio chunking
- streaming transcription
- language detection or language selection
- transcript panel
- timestamped segment storage

### Phase 3: diarization
Implement:
- speaker turn segmentation
- stable speaker IDs
- speaker summary generation

### Phase 4: summaries and translation
Implement:
- latest rolling summary (3–5 minute window, continuous refresh) with covered-time-range display
- current meeting summary (cumulative, periodic refresh)
- side-by-side live summary UI (rolling + cumulative) with copy / expand / save-to-notes actions
- selected-range (time-window) summaries
- final meeting summary (at session end)
- action item extraction
- transcript translation
- summary translation (applies to rolling, cumulative, time-window, and final summaries)

### Phase 5: export and polish
Implement:
- markdown export
- txt export
- json export
- speaker rename UI
- bilingual output options
- error handling

---

## Acceptance Criteria

The MVP is acceptable if a normal user can:

1. Run the app during a Zoom, Teams, or BBB meeting without special platform integration
2. Run the app during an offline classroom or room discussion using the local microphone
3. Capture audible meeting content from the local device in either mode
4. See a transcript update during the session
5. Receive approximate speaker-separated transcript segments
6. See a **latest rolling summary** that refreshes during the session and clearly shows the covered time range
7. See a **current meeting summary** that refreshes periodically and reflects the session so far
8. Copy, expand, and save either live summary to notes during the session
9. Generate a summary for one speaker
10. Generate a summary for a selected time period
11. Get transcript and summary output in Chinese, English, or German
12. Translate transcript and summary among Chinese, English, and German
13. Export the full meeting transcript, saved summary snapshots, and the final meeting summary

---

## What the implementation should avoid

- Do not over-engineer around official meeting platform APIs in the first version
- Do not assume perfect speaker identity resolution
- Do not make bot-based integration the default architecture
- Do not block the MVP on enterprise permissions or marketplace approvals
- Do not tightly couple summarization logic to one ASR or one LLM provider
- Do not tightly couple translation logic to one provider

---

## Deliverables the implementation should produce

The project should include:

1. A runnable MVP repository
2. A README with local setup instructions
3. A clear architecture description
4. A modular audio/transcript/speaker/summarization/translation pipeline
5. A basic UI for live transcript and summary generation
6. Export capability for transcript and notes
7. Language settings for Chinese, English, and German

---

## Final Direction

Build the product as a **platform-agnostic local meeting assistant with two capture modes**.

The central product insight is:

> Compatibility is achieved not by integrating deeply with every meeting platform, but by capturing the meeting locally and treating the platform as just an audio source.

And the product extension is:

> The same core pipeline should support both online meetings from system audio and offline meetings from local microphone capture.

And the language principle is:

> The system should treat Chinese, English, and German as first-class output languages, while preserving the original transcript as the source of truth and generating translations as derived outputs.

This should be the main design principle for all implementation decisions.