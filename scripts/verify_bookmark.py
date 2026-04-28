from __future__ import annotations

import sys
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.audio.capture import AudioSource  # noqa: E402
from meetingbro.asr.base import ASRAdapter  # noqa: E402
from meetingbro.exporter import export_meeting  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import SCHEMA, Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402
from meetingbro.schemas import Note  # noqa: E402


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


def _manager(storage: Storage) -> SessionManager:
    return SessionManager(SessionConfig(audio_source=_Audio(), asr=_ASR(), summarizer=_Summarizer(), translator=_Translator(), storage=storage, forced_language="en", summary_language="en"))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        storage = Storage(root / "verify.db")
        try:
            manager = _manager(storage)
            manager._state.elapsed_seconds = 12.34
            bookmark = manager.save_note("test bookmark", source_type="bookmark")
            saved = storage.list_notes(manager._state.meeting_id)[0]
            ok1 = saved.source_type == "bookmark" and abs((saved.time_seconds or 0.0) - manager._state.elapsed_seconds) < 0.01
            print(f"[{'OK' if ok1 else 'FAIL'}] basic flow")

            manager._state.elapsed_seconds = 21.0
            empty_bookmark = manager.save_note("", source_type="bookmark")
            ok2 = empty_bookmark.content == "" and empty_bookmark.time_seconds == 21.0
            print(f"[{'OK' if ok2 else 'FAIL'}] empty label")

            note = manager.save_note("a regular note")
            ok3 = note.time_seconds is None
            print(f"[{'OK' if ok3 else 'FAIL'}] regular note unchanged")

            meeting_id = "export-test"
            storage.create_meeting(meeting_id)
            storage.insert_note(Note(id="b1", meeting_id=meeting_id, content="Important decision about Q2 roadmap", source_type="bookmark", time_seconds=34.0, created_at=datetime.now(tz=timezone.utc)))
            storage.insert_note(Note(id="b2", meeting_id=meeting_id, content="Action item discussed", source_type="bookmark", time_seconds=727.0, created_at=datetime.now(tz=timezone.utc)))
            response = export_meeting(storage, meeting_id=meeting_id, export_root=root / "exports", export_dir=root / "export")
            summary_text = (Path(response.export_dir) / "summary.md").read_text(encoding="utf-8")
            metadata = (Path(response.export_dir) / "metadata.json").read_text(encoding="utf-8")
            ok4 = "## Bookmarks" in summary_text and "[00:34] Important decision about Q2 roadmap" in summary_text and "[12:07] Action item discussed" in summary_text and '"bookmarks"' in metadata
            print(f"[{'OK' if ok4 else 'FAIL'}] export markdown")

            legacy = root / "legacy.db"
            conn = sqlite3.connect(str(legacy))
            conn.executescript(SCHEMA.replace("  time_seconds REAL,\n", ""))
            conn.commit()
            conn.close()
            migrated = Storage(legacy)
            try:
                migrated.insert_note(Note(id="bm", meeting_id="m2", content="legacy bookmark", source_type="bookmark", time_seconds=5.0, created_at=datetime.now(tz=timezone.utc)))
                round_trip = migrated.list_notes("m2")[0]
                ok5 = round_trip.time_seconds == 5.0
            finally:
                migrated.close()
            print(f"[{'OK' if ok5 else 'FAIL'}] database migration")
            print("\nMarkdown excerpt:\n## Bookmarks\n- [00:34] Important decision about Q2 roadmap\n- [12:07] Action item discussed")
            return 0 if all((ok1, ok2, ok3, ok4, ok5)) else 1
        finally:
            storage.close()


if __name__ == "__main__":
    raise SystemExit(main())