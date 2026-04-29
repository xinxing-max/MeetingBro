export type LanguageCode = "zh" | "en" | "de";

export type SummaryType =
  | "meeting_memory"
  | "rolling_summary"
  | "cumulative_meeting_summary"
  | "time_window_summary"
  | "speaker_summary"
  | "final_summary"
  | "chapter_list"
  | "action_item_list";

export interface TranscriptSegment {
  id: string;
  meeting_id: string;
  start_time: number;
  end_time: number;
  text: string;
  original_language: LanguageCode | "unknown";
  speaker_id: string | null;
  confidence: number;
  quality?: "ok" | "uncertain" | "low";
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
  time_seconds?: number | null;
  created_at: string;
}

export interface ExportMeetingResponse {
  meeting_id: string;
  export_dir: string;
  files: string[];
}

export type SessionState = "starting" | "running" | "paused" | "ended";

export interface SessionStatePayload {
  state: SessionState;
  meeting_id: string;
  elapsed_seconds: number;
  source: string;
  runtime_profile: string;
  audio_chunk_seconds: number | null;
  asr_accumulation_seconds: number | null;
  language_lock_enabled: boolean;
  live_translation_language: LanguageCode | null;
  retry_windows_total: number;
  retry_windows_improved: number;
  retry_windows_unchanged: number;
  retry_windows_diverged: number;
  last_backpressure_elapsed_seconds: number | null;
  asr_last_audio_seconds: number | null;
  asr_last_wall_seconds: number | null;
  asr_realtime_factor: number | null;
  asr_safeguard_active: boolean;
  asr_safeguard_reason: string | null;
  asr_safeguard_events: number;
  weak_rescue_attempts: number;
  weak_rescue_emitted: number;
  weak_rescue_buffer_seconds: number;
  summary_pending_count: number;
  translation_pending_count: number;
  translation_backlog_trim_total: number;
  audio_drop_total: number;
  audio_input_backlog_seconds: number;
  audio_input_queue_drop_total: number;
  fast_preview_enabled: boolean;
  fast_preview_attempts: number;
  fast_preview_emitted: number;
  fast_preview_skipped: number;
  fast_preview_last_audio_seconds: number | null;
  fast_preview_last_wall_seconds: number | null;
  fast_preview_realtime_factor: number | null;
  preview_stale_suppressed?: number;
  preview_alignment_compared?: number;
  preview_alignment_similarity_avg?: number | null;
  preview_alignment_similarity_last?: number | null;
  mixed_microphone_gain: number | null;
  mixed_system_gain: number | null;
  mixed_effective_microphone_gain: number | null;
  mixed_auto_balance_enabled: boolean | null;
}

export type SessionEvent =
  | { type: "transcript_segment"; payload: TranscriptSegment }
  | { type: "transcript_translation"; payload: { segment_id: string; language: LanguageCode; text: string } }
  | { type: "transcript_preview"; payload: { segment: TranscriptSegment | null; preview_backend?: string; preview_is_experimental?: boolean; preview_quality_note?: string | null } }
  | { type: "summary_snapshot"; payload: SummarySnapshot }
  | { type: "note_saved"; payload: Note }
  | { type: "session_state"; payload: SessionStatePayload }
  | { type: "error"; payload: { code: string; message: string } };
