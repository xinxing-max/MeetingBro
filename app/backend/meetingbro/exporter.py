from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .schemas import ExportMeetingResponse, Note, SummarySnapshot, TranscriptSegment
from .storage.db import Storage


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _display_time(iso: Optional[str]) -> str:
    if not iso:
        return "unknown"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return iso


def _safe_export_dir(root: Path) -> Path:
    now = datetime.now().astimezone()
    # Include milliseconds so repeated exports in the same second remain unique
    # without asking the user to rename the folder.
    stem = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}-{now.microsecond // 1000:03d}_meetingbro"
    candidate = root / stem
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _safe_requested_export_dir(path: Path) -> Path:
    candidate = path
    suffix = 2
    while candidate.exists():
        if candidate.is_dir() and not any(candidate.iterdir()):
            return candidate
        candidate = path.with_name(f"{path.name}_{suffix}")
        suffix += 1
    return candidate


def _speaker_label(seg: TranscriptSegment) -> str:
    return seg.speaker_id or "Speaker"


def _write_transcript(
    path: Path,
    *,
    meeting: dict[str, str | None],
    segments: list[TranscriptSegment],
    client_metadata: dict[str, Any],
) -> None:
    lines: list[str] = [
        "# Live Transcript",
        "",
        f"- Meeting ID: `{meeting['id']}`",
        f"- Started: {_display_time(meeting.get('started_at'))}",
        f"- Ended: {_display_time(meeting.get('ended_at'))}",
    ]
    if client_metadata.get("source"):
        lines.append(f"- Source: `{client_metadata['source']}`")
    if client_metadata.get("runtime_profile"):
        lines.append(f"- Runtime mode: `{client_metadata['runtime_profile']}`")
    lines.extend(["", "## Transcript", ""])

    if not segments:
        lines.append("_No transcript segments saved._")
    for seg in segments:
        time_range = f"{_format_seconds(seg.start_time)}–{_format_seconds(seg.end_time)}"
        lines.append(f"### [{time_range}] {_speaker_label(seg)} ({seg.original_language})")
        lines.append(seg.text.strip())
        for language, translation in sorted(seg.translations.items()):
            if translation.strip() and translation.strip() != seg.text.strip():
                lines.append("")
                lines.append(f"> Translation ({language}): {translation.strip()}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _latest_by_type(snapshots: list[SummarySnapshot]) -> dict[str, SummarySnapshot]:
    latest: dict[str, SummarySnapshot] = {}
    for snap in snapshots:
        current = latest.get(snap.summary_type)
        if current is None or snap.is_latest or snap.created_at >= current.created_at:
            latest[snap.summary_type] = snap
    return latest


def _json_list(content: str) -> list[dict[str, Any]] | None:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def _write_summary(
    path: Path,
    *,
    meeting: dict[str, str | None],
    snapshots: list[SummarySnapshot],
    notes: list[Note],
) -> None:
    latest = _latest_by_type(snapshots)
    sections = [
        ("Latest Rolling Summary", "rolling_summary"),
        ("Meeting Board", "cumulative_meeting_summary"),
        ("Final Summary", "final_summary"),
        ("Chapters", "chapter_list"),
        ("Action Items", "action_item_list"),
        ("Compressed Meeting Memory", "meeting_memory"),
    ]
    lines: list[str] = [
        "# Meeting Summary",
        "",
        f"- Meeting ID: `{meeting['id']}`",
        f"- Started: {_display_time(meeting.get('started_at'))}",
        f"- Ended: {_display_time(meeting.get('ended_at'))}",
        "",
    ]

    for title, summary_type in sections:
        snap = latest.get(summary_type)
        lines.append(f"## {title}")
        if snap is None:
            lines.append("_Not available._")
        elif summary_type == "chapter_list":
            chapters = _json_list(snap.content)
            if chapters is None:
                lines.append("_Invalid chapter payload._")
            elif not chapters:
                lines.append("_No chapters extracted._")
            else:
                for chapter in chapters:
                    lines.append(
                        f"- [{_format_seconds(float(chapter.get('time_start', 0.0)))}–{_format_seconds(float(chapter.get('time_end', 0.0)))}] "
                        f"**{str(chapter.get('title') or 'Untitled chapter').strip()}**"
                    )
                    summary = str(chapter.get("summary") or "").strip()
                    if summary:
                        lines.append(f"  {summary}")
        elif summary_type == "action_item_list":
            action_items = _json_list(snap.content)
            if action_items is None:
                lines.append("_Invalid action item payload._")
            elif not action_items:
                lines.append("_No action items extracted._")
            else:
                for item in action_items:
                    parts = [str(item.get("text") or "").strip()]
                    if item.get("assignee"):
                        parts.append(f"assignee: {item['assignee']}")
                    if item.get("due"):
                        parts.append(f"due: {item['due']}")
                    lines.append(f"- {' · '.join(part for part in parts if part)}")
        else:
            lines.append(f"_Range: {_format_seconds(snap.time_start)}–{_format_seconds(snap.time_end)} · language: {snap.language}_")
            lines.append("")
            lines.append(snap.content.strip())
        lines.append("")

    lines.append("## Saved Notes")
    if not notes:
        lines.append("_No saved notes._")
    for note in notes:
        source = f" · {note.source_type}" if note.source_type else ""
        lines.append(f"- {_display_time(note.created_at.isoformat())}{source}: {note.content.strip()}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def export_meeting(
    storage: Storage,
    *,
    meeting_id: str,
    export_root: Path,
    export_dir: Optional[Path] = None,
    client_metadata: Optional[dict[str, Any]] = None,
) -> ExportMeetingResponse:
    meeting = storage.get_meeting(meeting_id)
    if meeting is None:
        raise KeyError(meeting_id)
    client_metadata = {k: v for k, v in (client_metadata or {}).items() if v is not None}
    segments = storage.list_segments(meeting_id)
    snapshots = storage.list_snapshots(meeting_id)
    notes = storage.list_notes(meeting_id)

    if export_dir is None:
        export_root.mkdir(parents=True, exist_ok=True)
        export_dir = _safe_export_dir(export_root)
    else:
        export_dir = _safe_requested_export_dir(export_dir)
        export_dir.parent.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = export_dir / "transcript.md"
    summary_path = export_dir / "summary.md"
    metadata_path = export_dir / "metadata.json"

    _write_transcript(
        transcript_path,
        meeting=meeting,
        segments=segments,
        client_metadata=client_metadata,
    )
    _write_summary(summary_path, meeting=meeting, snapshots=snapshots, notes=notes)

    metadata = {
        "meeting_id": meeting_id,
        "exported_at": datetime.now().astimezone().isoformat(),
        "started_at": meeting.get("started_at"),
        "ended_at": meeting.get("ended_at"),
        "preferred_summary_language": meeting.get("preferred_summary_language"),
        "segment_count": len(segments),
        "summary_count": len(snapshots),
        "notes_count": len(notes),
        "files": ["transcript.md", "summary.md", "metadata.json"],
        **client_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return ExportMeetingResponse(
        meeting_id=meeting_id,
        export_dir=str(export_dir),
        files=["transcript.md", "summary.md", "metadata.json"],
    )
