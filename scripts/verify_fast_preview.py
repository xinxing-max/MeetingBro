from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter, ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402


SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.05
SAMPLES = (
    0.03
    * np.sin(
        2.0
        * np.pi
        * 440.0
        * np.arange(int(SAMPLE_RATE * CHUNK_SECONDS), dtype=np.float32)
        / SAMPLE_RATE
    )
).astype(np.float32)


class _PreviewASR(ASRAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(
        self,
        samples,
        sample_rate,
        *,
        forced_language=None,
        offset_seconds=0.0,
        initial_prompt=None,
        quality_preset="realtime",
    ):
        self.calls += 1
        duration = len(samples) / sample_rate
        return [
            ASRSegment(
                start_time=max(0.0, duration - 0.4),
                end_time=duration,
                text="hello",
                language="en",
                confidence=0.92,
            )
        ]


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


class _RealtimeSource(AudioSource):
    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def stream(self):
        start = 0.0
        for _ in range(12):
            yield AudioChunk(samples=SAMPLES.copy(), sample_rate=SAMPLE_RATE, start_time=start)
            start += CHUNK_SECONDS
            await asyncio.sleep(CHUNK_SECONDS)

    async def aclose(self) -> None:
        return


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "fast_preview.db")
        asr = _PreviewASR()
        manager = SessionManager(
            SessionConfig(
                audio_source=_RealtimeSource(),
                asr=asr,
                summarizer=_NoopSummarizer(),
                translator=_NoopTranslator(),
                storage=storage,
                forced_language="en",
                summary_language="en",
                asr_accumulation_seconds=60.0,
                fast_preview_enabled=True,
                fast_preview_interval_seconds=0.1,
                fast_preview_window_seconds=0.3,
                fast_preview_min_rms=0.001,
                fast_preview_max_backlog_seconds=10.0,
                pre_vad_enabled=False,
                silence_rms_threshold=1.0,
                rolling_interval_seconds=10_000,
                cumulative_interval_seconds=10_000,
                memory_interval_seconds=10_000,
                min_segments_for_rolling=10_000,
                min_segments_for_memory=10_000,
                min_segments_for_cumulative=10_000,
            )
        )
        collected: dict[str, list[dict]] = {}

        async def collect() -> None:
            async for event in manager.events():
                collected.setdefault(event.type, []).append(event.payload)

        collector = asyncio.create_task(collect())
        await manager.start()
        await asyncio.sleep(0.7)
        await manager.stop()
        await asyncio.wait_for(collector, timeout=1.0)

        storage_segments = storage.list_segments(manager.meeting_id)
        storage.close()

    previews = [
        payload.get("segment")
        for payload in collected.get("transcript_preview", [])
        if payload.get("segment") is not None
    ]
    committed = collected.get("transcript_segment", [])
    ok = (
        len(previews) >= 1
        and previews[-1]["text"] == "hello"
        and len(committed) == 0
        and len(storage_segments) == 0
        and asr.calls >= 1
    )
    print(
        f"previews={len(previews)} committed={len(committed)} "
        f"db_segments={len(storage_segments)} asr_calls={asr.calls}"
    )
    if not ok:
        print("FAIL: fast preview should emit temporary preview only")
        return 1
    print("OK: fast preview emits temporary transcript without persistence")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
