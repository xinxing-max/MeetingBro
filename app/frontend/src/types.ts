export type LanguageCode = "zh" | "en" | "de";

export type SummaryType =
  | "meeting_memory"
  | "rolling_summary"
  | "cumulative_meeting_summary"
  | "time_window_summary"
  | "speaker_summary"
  | "final_summary";

export interface TranscriptSegment {
  id: string;
  meeting_id: string;
  start_time: number;
  end_time: number;
  text: string;
  original_language: LanguageCode | "unknown";
  speaker_id: string | null;
  confidence: number;
  translations: Partial<Record<LanguageCode, string>>;
  created_at: string;
  emitted_at_elapsed_seconds?: number | null;
}

export interface SummarySnapshot {
  id: string;
  meeting_id: string;
  summary_type: SummaryType;
  time_start: number;
  time_end: number;
  language: LanguageCode;
  content: string;
  source_segment_ids: string[];
  is_latest: boolean;
  translations: Partial<Record<LanguageCode, string>>;
  created_at: string;
}

export interface Note {
  id: string;
  meeting_id: string;
  content: string;
  source_type: string | null;
  source_id: string | null;
  created_at: string;
}

export type SessionState = "starting" | "running" | "paused" | "ended";

export interface SessionStatePayload {
  state: SessionState;
  meeting_id: string;
  elapsed_seconds: number;
  source: string;
  live_translation_language: LanguageCode | null;
  retry_windows_total: number;
  retry_windows_improved: number;
  retry_windows_unchanged: number;
  retry_windows_diverged: number;
  last_backpressure_elapsed_seconds: number | null;
  mixed_microphone_gain: number | null;
  mixed_system_gain: number | null;
  mixed_effective_microphone_gain: number | null;
  mixed_auto_balance_enabled: boolean | null;
}

export type SessionEvent =
  | { type: "transcript_segment"; payload: TranscriptSegment }
  | { type: "transcript_translation"; payload: { segment_id: string; language: LanguageCode; text: string } }
  | { type: "transcript_preview"; payload: { segment: TranscriptSegment | null } }
  | { type: "summary_snapshot"; payload: SummarySnapshot }
  | { type: "note_saved"; payload: Note }
  | { type: "session_state"; payload: SessionStatePayload }
  | { type: "error"; payload: { code: string; message: string } };
