"""End-to-end vertical-slice test.

Drives the full pipeline against a WAV file:
  WavFileSource -> FasterWhisperAdapter -> TranscriptSegments persisted & emitted
                -> rolling_summary + cumulative_meeting_summary + final_summary
                -> SQLite persistence

Uses compressed cadences (rolling every 2 s, cumulative every 5 s) so that a
short audio clip still exercises both live summary layers.

Usage:
  python scripts/e2e_vertical_slice.py
  python scripts/e2e_vertical_slice.py data/voice_test_statement_wikimedia.wav
  python scripts/e2e_vertical_slice.py data/mixed.wav --language auto
  python scripts/e2e_vertical_slice.py data/chinese.wav --language zh
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Allow running from repo root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402
from meetingbro.audio.capture import WavFileSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.llm import LLMSummarizer  # noqa: E402
from meetingbro.translation.passthrough import PassthroughTranslator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("e2e")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MeetingBro E2E against a WAV file.")
    parser.add_argument(
        "wav",
        nargs="?",
        default=str(ROOT / "data" / "sample_en.wav"),
        help="Path to a .wav file. Defaults to data/sample_en.wav.",
    )
    parser.add_argument(
        "--language",
        choices=("auto", "zh", "en", "de"),
        default="en",
        help=(
            "ASR language mode. Use 'auto' for multilingual/mixed audio. "
            "Default remains 'en' for the bundled English sample."
        ),
    )
    return parser.parse_args(argv[1:])


def _resolve_wav(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


async def main() -> int:
    args = _parse_args(sys.argv)
    wav = _resolve_wav(args.wav)
    forced_language = None if args.language == "auto" else args.language

    if not wav.exists():
        log.error("wav missing at %s", wav)
        return 2
    if wav.suffix.lower() != ".wav":
        log.error("expected a .wav file, got %s", wav)
        return 2

    log.info("using wav=%s language=%s", wav, args.language)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "e2e.db"
        storage = Storage(db_path)
        try:
            asr = FasterWhisperAdapter(model_size="tiny")
            config = SessionConfig(
                audio_source=WavFileSource(
                    wav, sample_rate=16_000, chunk_seconds=5.0, realtime=False
                ),
                asr=asr,
                summarizer=LLMSummarizer(),  # falls back to heuristic with no API key
                translator=PassthroughTranslator(),
                storage=storage,
                forced_language=forced_language,
                summary_language="en",
                rolling_window_seconds=30.0,
                rolling_interval_seconds=2.0,
                cumulative_interval_seconds=5.0,
                min_segments_for_rolling=1,
                min_segments_for_cumulative=1,
            )
            manager = SessionManager(config)

            collected: dict[str, list[dict]] = {
                "session_state": [],
                "transcript_segment": [],
                "summary_snapshot": [],
                "error": [],
            }

            async def collect() -> None:
                async for ev in manager.events():
                    collected.setdefault(ev.type, []).append(ev.payload)

            collector = asyncio.create_task(collect())
            await manager.start()

            # Wait until the run task ends (EOF on the WAV source) plus a short
            # grace period for final events.
            await asyncio.sleep(0.1)
            if manager._task is not None:
                await manager._task
            await manager.stop()

            collector.cancel()
            try:
                await collector
            except (asyncio.CancelledError, Exception):
                pass

            print()
            print("=== E2E summary ===")
            print(f"meeting_id: {manager.meeting_id}")
            for k, v in collected.items():
                print(f"event {k}: {len(v)}")

            segs = collected["transcript_segment"]
            if segs:
                print()
                print("first 3 transcript segments:")
                for s in segs[:3]:
                    print(f"  [{s['start_time']:.2f}-{s['end_time']:.2f}] ({s['original_language']}) {s['text']}")

            snaps = collected["summary_snapshot"]
            by_type: dict[str, list[dict]] = {}
            for snap in snaps:
                by_type.setdefault(snap["summary_type"], []).append(snap)

            print()
            print("summary snapshot types:", {k: len(v) for k, v in by_type.items()})
            for kind in ("rolling_summary", "cumulative_meeting_summary", "final_summary"):
                if kind in by_type:
                    latest = by_type[kind][-1]
                    print(
                        f"\n[{kind}] {latest['time_start']:.1f}-{latest['time_end']:.1f} "
                        f"lang={latest['language']}"
                    )
                    print(latest["content"])

            print()
            print("=== persistence check ===")
            db_segs = storage.list_segments(manager.meeting_id)
            db_snaps = storage.list_snapshots(manager.meeting_id)
            print(f"segments in db: {len(db_segs)}")
            print(f"snapshots in db: {len(db_snaps)}")
            snap_types_in_db = sorted({s.summary_type for s in db_snaps})
            print(f"snapshot types in db: {snap_types_in_db}")
            latest_flags = {
                s.summary_type: s.is_latest for s in db_snaps if s.is_latest
            }
            print(f"is_latest per type: {latest_flags}")

            note = manager.save_note(
                content="save-to-notes smoke-check",
                source_type="rolling_summary",
            )
            db_notes = storage.list_notes(manager.meeting_id)
            print(f"notes saved: {len(db_notes)} (latest id {note.id[:8]})")

            ok = (
                len(segs) >= 1
                and {"rolling_summary", "cumulative_meeting_summary"} & set(by_type)
                and len(db_segs) == len(segs)
                and len(db_snaps) >= 1
                and len(db_notes) == 1
            )
            if not ok:
                print("\nFAIL: E2E assertions not met")
                return 1

            print("\nOK: E2E vertical slice succeeded")
            return 0
        finally:
            storage.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
