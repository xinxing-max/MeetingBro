# Shared types

Canonical definitions of the contracts exchanged between the frontend and the local backend.
Keep `app/frontend/src/types.ts` and `app/backend/meetingbro/schemas.py` aligned with this document.

## Languages

Supported language codes: `"zh" | "en" | "de"`.

## Summary types

A `SummarySnapshot` is tagged with one of:

| `summary_type`               | Scope                                                  | Cadence                                |
|------------------------------|--------------------------------------------------------|----------------------------------------|
| `rolling_summary`            | Most recent 3–5 minutes of transcript                  | Continuous, ~60–90 s                   |
| `cumulative_meeting_summary` | From session start to `time_end`                       | Periodic, ~3–5 min                     |
| `time_window_summary`        | User-selected range                                    | On demand                              |
| `speaker_summary`            | A single speaker's remarks                             | On demand                              |
| `final_summary`              | Whole meeting recap                                    | Once, at session end                   |

## TranscriptSegment

```ts
{
  id: string
  meeting_id: string
  start_time: number          // seconds from session start
  end_time: number
  text: string
  original_language: "zh" | "en" | "de" | "unknown"
  speaker_id: string | null
  confidence: number          // 0..1
  translations: Partial<Record<"zh" | "en" | "de", string>>
}
```

## SummarySnapshot

```ts
{
  id: string
  meeting_id: string
  summary_type: "rolling_summary"
               | "cumulative_meeting_summary"
               | "time_window_summary"
               | "speaker_summary"
               | "final_summary"
  time_start: number          // seconds from session start
  time_end: number
  language: "zh" | "en" | "de"
  content: string
  source_segment_ids: string[]
  is_latest: boolean          // true for the most recent snapshot of this type
  translations: Partial<Record<"zh" | "en" | "de", string>>
  created_at: string          // ISO-8601
}
```

## Speaker

```ts
{
  id: string
  meeting_id: string
  display_name: string | null
  inferred_label: string      // e.g. "Speaker 1"
  confidence: number
  is_local_user: boolean
}
```

## Note

```ts
{
  id: string
  meeting_id: string
  content: string
  source_type: string | null  // e.g. "rolling_summary", "cumulative_meeting_summary"
  source_id: string | null    // id of the source snapshot
  created_at: string          // ISO-8601
}
```

## Session WebSocket events

Every event is JSON of the form `{ "type": "<event_type>", "payload": <object> }`.

| `type`               | Payload                                  |
|----------------------|------------------------------------------|
| `transcript_segment` | `TranscriptSegment`                      |
| `summary_snapshot`   | `SummarySnapshot`                        |
| `speaker_update`     | `Speaker`                                |
| `note_saved`         | `Note`                                   |
| `session_state`      | `{ state: "starting" \| "running" \| "paused" \| "ended", meeting_id: string }` |
| `error`              | `{ code: string, message: string }`      |

## Client-to-server WebSocket commands

| `type`      | Payload                                                         |
|-------------|-----------------------------------------------------------------|
| `save_note` | `{ content: string, source_type?: string, source_id?: string }` |
| `stop`      | `{}`                                                            |
