from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter, ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioSource  # noqa: E402
from meetingbro.schemas import TranscriptSegment  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import SCHEMA, Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402


class _Audio(AudioSource):
    @property
    def sample_rate(self): return 16000
    async def stream(self):
        if False:
            yield None


class _ASR(ASRAdapter):
    def transcribe(self, samples, sample_rate, *, forced_language=None, offset_seconds=0.0, initial_prompt=None, quality_preset="realtime"):
        return []


class _Summarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None): return ""


class _Translator(Translator):
    def translate(self, text, *, source_language, target_language): return text


def _verify_manager_quality() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify.db")
        try:
            manager = SessionManager(SessionConfig(audio_source=_Audio(), asr=_ASR(), summarizer=_Summarizer(), translator=_Translator(), storage=storage, forced_language="en", summary_language="en", asr_accumulation_seconds=0.1, silence_rms_threshold=0.0, pre_vad_enabled=False, rolling_interval_seconds=9999, memory_interval_seconds=9999, cumulative_interval_seconds=9999))
            cases = [
                ASRSegment(0.1, 0.3, "A", "en", 0.9, avg_logprob=-0.4, no_speech_prob=0.1, compression_ratio=1.3),
                ASRSegment(0.4, 0.6, "B", "en", 0.9, avg_logprob=-1.2, no_speech_prob=0.1, compression_ratio=1.3),
                ASRSegment(0.7, 0.9, "C", "en", 0.9, avg_logprob=-1.2, no_speech_prob=0.7, compression_ratio=1.3),
                ASRSegment(1.0, 1.2, "D", "en", 0.9, avg_logprob=-1.2, no_speech_prob=0.7, compression_ratio=2.5),
            ]
            assert [manager._compute_quality(case) for case in cases] == ["ok", "uncertain", "uncertain", "low"]
        finally:
            storage.close()


def _verify_storage_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "roundtrip.db")
        try:
            storage.insert_segment(TranscriptSegment(id="seg-low", meeting_id="m1", start_time=0.0, end_time=1.0, text="low", original_language="en", confidence=0.5, quality="low", created_at=datetime.now(tz=timezone.utc)))
            assert storage.list_segments("m1")[0].quality == "low"
        finally:
            storage.close()


def _verify_old_db_migration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "legacy.db"
        conn = sqlite3.connect(str(db_path)); conn.executescript(SCHEMA.replace("  quality TEXT NOT NULL DEFAULT 'ok',\n", "")); conn.commit(); conn.close()
        storage = Storage(db_path)
        try:
            storage.insert_segment(TranscriptSegment(id="seg-migrate", meeting_id="m2", start_time=0.0, end_time=1.0, text="migrate", original_language="en", confidence=0.8, quality="low", created_at=datetime.now(tz=timezone.utc)))
            assert storage.list_segments("m2")[0].quality == "low"
        finally:
            storage.close()


def main() -> int:
    _verify_manager_quality(); _verify_storage_round_trip(); _verify_old_db_migration()
    print("OK: segment quality derives, persists, and migrates correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())