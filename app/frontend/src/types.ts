export type LanguageCode = "zh" | "en" | "de";

export type SummaryType =
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

export interface Speaker {
  id: string;
  meeting_id: string;
  display_name: string | null;
  inferred_label: string;
  confidence: number;
  is_local_user: boolean;
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

export type SessionEvent =
  | { type: "transcript_segment"; payload: TranscriptSegment }
  | { type: "summary_snapshot"; payload: SummarySnapshot }
  | { type: "speaker_update"; payload: Speaker }
  | { type: "note_saved"; payload: Note }
  | { type: "session_state"; payload: { state: SessionState; meeting_id: string } }
  | { type: "error"; payload: { code: string; message: string } };
