from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import time
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

TAIL_TEXT = "we should start with the agenda"
SAMPLE_RATE = 16_000


class _SpeechThenSilenceSource(AudioSource):
    def __init__(self, *, sleep_seconds: float = 0.05) -> None:
        self._sleep_seconds = sleep_seconds
        self._rng = np.random.default_rng(seed=41)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def stream(self):
        speech = self._rng.normal(loc=0.0, scale=0.04, size=SAMPLE_RATE // 2).astype(np.float32)
        silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        start = 0.0
        for chunk in [speech, speech, speech, *([silence] * 12)]:
            await asyncio.sleep(self._sleep_seconds)
            yield AudioChunk(samples=chunk.copy(), sample_rate=SAMPLE_RATE, start_time=start)
            start += 0.5


class _TailOnlyASR(ASRAdapter):
    def __init__(self, *, confidence: float) -> None:
        self.calls = 0
        self._confidence = confidence

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
        del samples, sample_rate, forced_language, offset_seconds, initial_prompt, quality_preset
        self.calls += 1
        time.sleep(0.02)
        return [
            ASRSegment(
                start_time=0.1,
                end_time=1.2,
                text=TAIL_TEXT,
                language="en",
                confidence=self._confidence,
            )
        ]


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


async def _run_case(*, confidence: float) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / f"verify_silence_commit_confidence_{confidence:.2f}.db")
        try:
            asr = _TailOnlyASR(confidence=confidence)
            manager = SessionManager(
                SessionConfig(
                    audio_source=_SpeechThenSilenceSource(),
                    asr=asr,
                    summarizer=_NoopSummarizer(),
                    translator=_NoopTranslator(),
                    storage=storage,
                    forced_language="en",
                    summary_language="en",
                    rolling_interval_seconds=10_000,
                    cumulative_interval_seconds=10_000,
                    memory_interval_seconds=10_000,
                    min_segments_for_rolling=10_000,
                    min_segments_for_memory=10_000,
                    min_segments_for_cumulative=10_000,
                    asr_accumulation_seconds=1.5,
                    silence_rms_threshold=0.002,
                    pre_vad_enabled=False,
                    silence_commit_min_confidence=0.75,
                    silence_commit_min_duration_seconds=0.6,
                )
            )

            collected: list[tuple[str, dict, bool]] = []
            stop_requested = False
            preview_seen = asyncio.Event()

            async def collect() -> None:
                async for ev in manager.events():
                    collected.append((ev.type, ev.payload, stop_requested))
                    if ev.type == "transcript_preview" and ev.payload.get("segment"):
                        preview_seen.set()

            collector = asyncio.create_task(collect())
            await manager.start()
            await asyncio.wait_for(preview_seen.wait(), timeout=2.0)
            await asyncio.sleep(0.25)
            stop_requested = True
            await manager.stop()
            collector.cancel()
            try:
                await collector
            except (asyncio.CancelledError, Exception):
                pass

            committed_before_stop = sum(
                1
                for event_type, payload, stopped in collected
                if event_type == "transcript_segment" and payload["text"].strip() == TAIL_TEXT and not stopped
            )
            committed_after_stop = sum(
                1
                for event_type, payload, stopped in collected
                if event_type == "transcript_segment" and payload["text"].strip() == TAIL_TEXT and stopped
            )
            return asr.calls, committed_before_stop, committed_after_stop
        finally:
            storage.close()


async def main() -> int:
    high_conf = await _run_case(confidence=0.92)
    low_conf = await _run_case(confidence=0.40)

    print(f"high confidence: asr_calls={high_conf[0]} before_stop={high_conf[1]} after_stop={high_conf[2]}")
    print(f"low confidence:  asr_calls={low_conf[0]} before_stop={low_conf[1]} after_stop={low_conf[2]}")

    ok = (
        high_conf == (1, 1, 0)
        and low_conf == (1, 0, 1)
    )
    if ok:
        print("\nOK: silence-boundary commit is gated by confidence")
        return 0
    print("\nFAIL: silence-boundary confidence gate regressed")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))