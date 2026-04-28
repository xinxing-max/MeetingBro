from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.exporter import export_meeting  # noqa: E402
from meetingbro.schemas import TranscriptSegment  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402


class _MissingLanguageStorage:
    def __init__(self, base: Storage, meeting_id: str) -> None:
        self._base = base
        self._meeting_id = meeting_id

    def get_meeting(self, meeting_id: str):
        meeting = self._base.get_meeting(meeting_id)
        if meeting_id == self._meeting_id and meeting is not None:
            return {**meeting, "preferred_summary_language": None}
        return meeting

    def list_segments(self, meeting_id: str):
        return self._base.list_segments(meeting_id)

    def list_snapshots(self, meeting_id: str):
        return self._base.list_snapshots(meeting_id)

    def list_notes(self, meeting_id: str):
        return self._base.list_notes(meeting_id)


def _segment(segment_id: str, *, start: float, end: float, text: str, language: str, speaker: str, translations: dict[str, str]) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        meeting_id="m1",
        start_time=start,
        end_time=end,
        text=text,
        original_language=language,
        speaker_id=speaker,
        confidence=0.95,
        translations=translations,
        created_at=datetime.now(tz=timezone.utc),
    )


def _export(storage: Storage, *, meeting_id: str = "m1", bilingual: bool = False, target_language: str | None = None) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        response = export_meeting(
            storage,
            meeting_id=meeting_id,
            export_root=Path(tmp) / "exports",
            export_dir=Path(tmp) / "export",
            bilingual=bilingual,
            target_language=target_language,  # type: ignore[arg-type]
        )
        return (Path(response.export_dir) / "transcript.md").read_text(encoding="utf-8")


def _display_started(storage: Storage, meeting_id: str) -> str:
    meeting = storage.get_meeting(meeting_id)
    assert meeting is not None
    started_at = meeting.get("started_at")
    assert started_at is not None
    return datetime.fromisoformat(started_at).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify.db")
        try:
            storage.create_meeting("m1", preferred_summary_language="zh")
            segments = [
                _segment("s1", start=1.0, end=4.0, text="你好，今天讨论 Q2 路线图。", language="zh", speaker="Speaker 1", translations={"en": "Hello, today we discuss the Q2 roadmap.", "zh": "你好，今天讨论 Q2 路线图。"}),
                _segment("s2", start=8.0, end=10.0, text="我觉得优先级应该调整。", language="zh", speaker="Speaker 1", translations={"en": "I think the priorities should change."}),
                _segment("s3", start=14.0, end=16.0, text="I agree with that.", language="en", speaker="Speaker 2", translations={"en": "I agree with that.", "zh": "我同意这一点。"}),
            ]
            for seg in segments:
                storage.insert_segment(seg)

            transcript = _export(storage, bilingual=True, target_language="en")
            ok1 = "| Time | Speaker | Original | EN |" in transcript and transcript.count("| 00:") >= 3 and "| 00:01–00:04 | Speaker 1 | 你好，今天讨论 Q2 路线图。 | Hello, today we discuss the Q2 roadmap. |" in transcript and "| 00:08–00:10 | Speaker 1 | 我觉得优先级应该调整。 | I think the priorities should change. |" in transcript and "| 00:14–00:16 | Speaker 2 | I agree with that. | I agree with that. |" in transcript
            print(f"[{'OK' if ok1 else 'FAIL'}] normal bilingual table")

            storage.insert_segment(_segment("s4", start=20.0, end=22.0, text="这段翻译还没回来。", language="zh", speaker="Speaker 3", translations={}))
            missing = _export(storage, bilingual=True, target_language="en")
            ok2 = "| 00:20–00:22 | Speaker 3 | 这段翻译还没回来。 | — |" in missing
            print(f"[{'OK' if ok2 else 'FAIL'}] missing translation placeholder")

            ok3 = "| 00:14–00:16 | Speaker 2 | I agree with that. | I agree with that. |" in missing and "(same)" not in missing
            print(f"[{'OK' if ok3 else 'FAIL'}] same-language row mirrors original")

            storage.insert_segment(_segment("s5", start=24.0, end=26.0, text="A | B\nC", language="en", speaker="Speaker 4", translations={"en": "X | Y\nZ"}))
            escaped = _export(storage, bilingual=True, target_language="en")
            line = next(row for row in escaped.splitlines() if row.startswith("| 00:24–00:26 |"))
            ok4 = line.count("|") >= 5 and "A \\| B C" in line
            print(f"[{'OK' if ok4 else 'FAIL'}] table cell escape")

            expected_default = f"# Live Transcript\n\n- Meeting ID: `m1`\n- Started: {_display_started(storage, 'm1')}\n- Ended: unknown\n\n## Transcript\n\n### [00:01–00:04] Speaker 1 (zh)\n你好，今天讨论 Q2 路线图。\n\n> Translation (en): Hello, today we discuss the Q2 roadmap.\n\n### [00:08–00:10] Speaker 1 (zh)\n我觉得优先级应该调整。\n\n> Translation (en): I think the priorities should change.\n\n### [00:14–00:16] Speaker 2 (en)\nI agree with that.\n\n> Translation (zh): 我同意这一点。\n\n### [00:20–00:22] Speaker 3 (zh)\n这段翻译还没回来。\n\n### [00:24–00:26] Speaker 4 (en)\nA | B\nC\n\n> Translation (en): X | Y\nZ\n"
            plain = _export(storage, bilingual=False)
            ok5 = "| --- | --- | --- | --- |" not in plain and plain == expected_default
            print(f"[{'OK' if ok5 else 'FAIL'}] default transcript unchanged")

            fallback = _export(storage, bilingual=True, target_language=None)
            ok6 = "| Time | Speaker | Original | ZH |" in fallback and "| 00:01–00:04 | Speaker 1 | 你好，今天讨论 Q2 路线图。 | 你好，今天讨论 Q2 路线图。 |" in fallback
            print(f"[{'OK' if ok6 else 'FAIL'}] target language fallback")

            storage.create_meeting("m2")
            storage.insert_segment(TranscriptSegment(id="x1", meeting_id="m2", start_time=1.0, end_time=2.0, text="Hallo", original_language="de", speaker_id="Speaker 1", confidence=0.95, translations={}, created_at=datetime.now(tz=timezone.utc)))
            try:
                _export(_MissingLanguageStorage(storage, "m2"), meeting_id="m2", bilingual=True, target_language=None)
                ok7 = False
            except ValueError:
                ok7 = True
            print(f"[{'OK' if ok7 else 'FAIL'}] missing target language raises")

            print("\nTable excerpt:\n| Time | Speaker | Original | EN |\n| --- | --- | --- | --- |\n| 00:08–00:10 | Speaker 1 | 我觉得优先级应该调整。 | I think the priorities should change. |\n| 00:20–00:22 | Speaker 3 | 这段翻译还没回来。 | — |")
            return 0 if all((ok1, ok2, ok3, ok4, ok5, ok6, ok7)) else 1
        finally:
            storage.close()


if __name__ == "__main__":
    raise SystemExit(main())