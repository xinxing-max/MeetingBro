from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "app/backend")

from meetingbro.exporter import export_meeting  # noqa: E402
from meetingbro.schemas import Note, SummarySnapshot, TranscriptSegment  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        storage = Storage(root / "meetingbro-test.db")
        meeting_id = "meeting-export-test"
        try:
            storage.create_meeting(meeting_id, preferred_summary_language="en")
            storage.insert_segment(
                TranscriptSegment(
                    id="seg-1",
                    meeting_id=meeting_id,
                    start_time=0.0,
                    end_time=3.2,
                    text="Hello from Alice.",
                    original_language="en",
                    speaker_id="speaker-1",
                    confidence=0.94,
                    translations={"zh": "Alice 说你好。"},
                    created_at=datetime.now(tz=timezone.utc),
                )
            )
            storage.insert_snapshot(
                SummarySnapshot(
                    id="sum-1",
                    meeting_id=meeting_id,
                    summary_type="cumulative_meeting_summary",
                    time_start=0.0,
                    time_end=3.2,
                    language="en",
                    content="## Meeting State\n- Alice greeted the group.",
                    source_segment_ids=["seg-1"],
                    is_latest=True,
                    created_at=datetime.now(tz=timezone.utc),
                )
            )
            storage.insert_note(
                Note(
                    id="note-1",
                    meeting_id=meeting_id,
                    content="Remember to follow up.",
                    source_type="manual",
                    source_id=None,
                    created_at=datetime.now(tz=timezone.utc),
                )
            )

            response = export_meeting(
                storage,
                meeting_id=meeting_id,
                export_root=root / "exports",
                export_dir=root / "chosen-export-folder",
                client_metadata={
                    "source": "mixed",
                    "runtime_profile": "balanced",
                    "summary_language": "en",
                },
            )
        finally:
            storage.close()

        export_dir = Path(response.export_dir)
        transcript = export_dir / "transcript.md"
        summary = export_dir / "summary.md"
        metadata = export_dir / "metadata.json"
        print(f"export_dir: {export_dir}")
        print(f"files:      {response.files}")

        if export_dir.name != "chosen-export-folder":
            print("FAIL: explicit export_dir was not respected")
            return 1

        if not (transcript.exists() and summary.exists() and metadata.exists()):
            print("FAIL: one or more export files are missing")
            return 1

        transcript_text = transcript.read_text(encoding="utf-8")
        summary_text = summary.read_text(encoding="utf-8")
        metadata_json = json.loads(metadata.read_text(encoding="utf-8"))
        ok = (
            "Hello from Alice." in transcript_text
            and "Translation (zh)" in transcript_text
            and "Meeting Board" in summary_text
            and "Remember to follow up." in summary_text
            and metadata_json["segment_count"] == 1
            and metadata_json["summary_count"] == 1
            and metadata_json["notes_count"] == 1
        )
        if ok:
            print("\nOK: meeting export creates transcript.md, summary.md, and metadata.json")
            return 0
        print("\nFAIL: export content did not match expectations")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
