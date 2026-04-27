"""End-to-end test for Windows WASAPI system-audio loopback capture.

Plays ``data/sample_en.wav`` through the default system output device while
simultaneously capturing the device's render stream via the new
``SystemAudioLoopbackSource``. The captured audio is fed through the full
pipeline (ASR -> transcript -> rolling/cumulative summaries -> persistence) to
prove that online-meeting mode works against real system audio, not just a file.

Run on Windows with audible default output. Headphones are fine — loopback
captures what the device renders, not what the room microphone hears.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402
from meetingbro.audio.loopback import SystemAudioLoopbackSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.llm import LLMSummarizer  # noqa: E402
from meetingbro.translation.passthrough import PassthroughTranslator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("e2e_loopback")


def _play_wav_loop(wav_path: Path, stop_event: threading.Event) -> None:
    """Play the WAV on the default output device repeatedly until asked to stop."""
    import sounddevice as sd
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    log.info("playback: loaded %s shape=%s sr=%d", wav_path.name, data.shape, sr)
    while not stop_event.is_set():
        sd.play(data, sr, blocking=False)
        # Block on the playback duration rather than sd.wait() so we can break
        # out quickly when the stop_event fires between loops.
        duration = len(data) / sr
        end = threading.Event()
        end.wait(timeout=duration + 0.25)
        if stop_event.is_set():
            sd.stop()
            break
    sd.stop()
    log.info("playback: stopped")


async def main() -> int:
    wav = ROOT / "data" / "sample_en.wav"
    if not wav.exists():
        log.error("sample wav missing at %s", wav)
        return 2

    if not sys.platform.startswith("win"):
        log.error("this test is Windows-only (WASAPI loopback)")
        return 2

    # Run the pipeline for a fixed capture window then tear everything down.
    capture_seconds = 20.0

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "e2e_loopback.db"
        storage = Storage(db_path)
        try:
            asr = FasterWhisperAdapter(model_size="tiny")
            loopback = SystemAudioLoopbackSource(
                sample_rate=16_000, chunk_seconds=3.0
            )
            config = SessionConfig(
                audio_source=loopback,
                asr=asr,
                summarizer=LLMSummarizer(),  # heuristic fallback with no API key
                translator=PassthroughTranslator(),
                storage=storage,
                forced_language="en",
                summary_language="en",
                rolling_window_seconds=30.0,
                rolling_interval_seconds=4.0,
                cumulative_interval_seconds=10.0,
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

            # Start playback in a background thread so the loopback source has
            # real system audio to capture.
            stop_playback = threading.Event()
            player = threading.Thread(
                target=_play_wav_loop,
                args=(wav, stop_playback),
                daemon=True,
            )
            player.start()

            log.info("capturing loopback for %.1f s", capture_seconds)
            await asyncio.sleep(capture_seconds)

            stop_playback.set()
            await manager.stop()
            player.join(timeout=3.0)

            collector.cancel()
            try:
                await collector
            except (asyncio.CancelledError, Exception):
                pass

            print()
            print("=== E2E loopback summary ===")
            print(f"meeting_id: {manager.meeting_id}")
            for k, v in collected.items():
                print(f"event {k}: {len(v)}")

            segs = collected["transcript_segment"]
            if segs:
                print()
                print("first 3 transcript segments:")
                for s in segs[:3]:
                    print(
                        f"  [{s['start_time']:.2f}-{s['end_time']:.2f}] "
                        f"({s['original_language']}) {s['text']}"
                    )

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

            ok = (
                len(segs) >= 1
                and len(db_segs) == len(segs)
                and not collected["error"]
            )
            if not ok:
                print("\nFAIL: loopback E2E assertions not met")
                return 1
            print("\nOK: loopback captured real system audio through the pipeline")
            return 0
        finally:
            storage.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
