from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


LanguageCode = Literal["zh", "en", "de"]
OriginalLanguage = Literal["zh", "en", "de", "unknown"]

SummaryType = Literal[
    "meeting_memory",
    "rolling_summary",
    "cumulative_meeting_summary",
    "time_window_summary",
    "speaker_summary",
    "final_summary",
]

SessionState = Literal["starting", "running", "paused", "ended"]


class TranscriptSegment(BaseModel):
    id: str
    meeting_id: str
    start_time: float
    end_time: float
    text: str
    original_language: OriginalLanguage
    speaker_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    translations: dict[LanguageCode, str] = Field(default_factory=dict)
    created_at: datetime
    emitted_at_elapsed_seconds: Optional[float] = None


class SummarySnapshot(BaseModel):
    id: str
    meeting_id: str
    summary_type: SummaryType
    time_start: float
    time_end: float
    language: LanguageCode
    content: str
    source_segment_ids: list[str] = Field(default_factory=list)
    is_latest: bool = False
    translations: dict[LanguageCode, str] = Field(default_factory=dict)
    created_at: datetime


class Speaker(BaseModel):
    id: str
    meeting_id: str
    display_name: Optional[str] = None
    inferred_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_local_user: bool = False


class Note(BaseModel):
    id: str
    meeting_id: str
    content: str
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    created_at: datetime


class SessionStatePayload(BaseModel):
    state: SessionState
    meeting_id: str
    elapsed_seconds: float = 0.0
    source: str = "mic"
    runtime_profile: str = "balanced"
    audio_chunk_seconds: Optional[float] = None
    asr_accumulation_seconds: Optional[float] = None
    language_lock_enabled: bool = False
    live_translation_language: Optional[LanguageCode] = None
    retry_windows_total: int = 0
    retry_windows_improved: int = 0
    retry_windows_unchanged: int = 0
    retry_windows_diverged: int = 0
    last_backpressure_elapsed_seconds: Optional[float] = None
    asr_last_audio_seconds: Optional[float] = None
    asr_last_wall_seconds: Optional[float] = None
    asr_realtime_factor: Optional[float] = None
    asr_safeguard_active: bool = False
    asr_safeguard_reason: Optional[str] = None
    asr_safeguard_events: int = 0
    weak_rescue_attempts: int = 0
    weak_rescue_emitted: int = 0
    weak_rescue_buffer_seconds: float = 0.0
    summary_pending_count: int = 0
    translation_pending_count: int = 0
    translation_backlog_trim_total: int = 0
    audio_drop_total: int = 0
    mixed_microphone_gain: Optional[float] = None
    mixed_system_gain: Optional[float] = None
    mixed_effective_microphone_gain: Optional[float] = None
    mixed_auto_balance_enabled: Optional[bool] = None


class TranscriptPreviewPayload(BaseModel):
    segment: Optional[TranscriptSegment] = None


class TranscriptTranslationPayload(BaseModel):
    segment_id: str
    language: LanguageCode
    text: str


class ErrorPayload(BaseModel):
    code: str
    message: str


class CreateNoteRequest(BaseModel):
    meeting_id: str
    content: str
    source_type: Optional[str] = None
    source_id: Optional[str] = None
