from __future__ import annotations

import asyncio
import logging
import math
import sys
import tempfile
import time
from dataclasses import dataclass
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

SAMPLE_RATE = 16_000
PROCESSING_SECONDS = 0.05


@dataclass(frozen=True)
class _LatencyResult:
    preview_delay_seconds: float
    committed_delay_seconds: float
    asr_calls: int


class _RealtimeSpeechSource(AudioSource):
    def __init__(self, *, chunk_seconds: float, total_seconds: float, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._total_seconds = total_seconds
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def stream(self):
        frames = max(1, int(self._sample_rate * self._chunk_seconds))
        total_chunks = max(1, math.ceil(self._total_seconds / self._chunk_seconds))
        t0 = 0.0
        samples = np.full(frames, 0.04, dtype=np.float32)
        for _ in range(total_chunks):
            if self._closed:
                return
            await asyncio.sleep(self._chunk_seconds)
            yield AudioChunk(samples=samples.copy(), sample_rate=self._sample_rate, start_time=t0)
            t0 += len(samples) / self._sample_rate

    async def aclose(self) -> None:
        self._closed = True


class _LatencyMockASR(ASRAdapter):
    def __init__(self, *, processing_seconds: float = PROCESSING_SECONDS) -> None:
        self._processing_seconds = processing_seconds
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
        del forced_language, offset_seconds, initial_prompt, quality_preset
        time.sleep(self._processing_seconds)
        self.calls += 1
        duration = len(samples) / sample_rate
        return [
            ASRSegment(
                start_time=0.1,
                end_time=max(0.2, duration - 0.1),
                text=f"Segment {self.calls}.",
                language="en",
                confidence=0.92,
            )
        ]


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


async def _wait_for_commit(commit_at: list[float | None], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if commit_at[0] is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for first committed segment")


def _assert_flush_threshold_cases() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "early_flush_thresholds.db")
        try:
            manager = SessionManager(
                SessionConfig(
                    audio_source=_RealtimeSpeechSource(chunk_seconds=1.0, total_seconds=1.0),
                    asr=_LatencyMockASR(),
                    summarizer=_NoopSummarizer(),
                    translator=_NoopTranslator(),
                    storage=storage,
                    forced_language="en",
                    summary_language="en",
                    asr_accumulation_seconds=1.5,
                    asr_early_flush_enabled=True,
                    asr_early_flush_min_seconds=0.8,
                    silence_rms_threshold=0.002,
                    pre_vad_enabled=False,
                )
            )
            single_voiced = manager._resolve_asr_flush_threshold(
                threshold=1.5,
                accum_chunk_rms=[0.04],
                current_chunk_rms=0.04,
            )
            voiced_then_silence = manager._resolve_asr_flush_threshold(
                threshold=1.5,
                accum_chunk_rms=[0.04, 0.0],
                current_chunk_rms=0.0,
            )
            voiced_then_voiced = manager._resolve_asr_flush_threshold(
                threshold=1.5,
                accum_chunk_rms=[0.04, 0.04],
                current_chunk_rms=0.04,
            )
            manager._cfg.asr_early_flush_enabled = False
            disabled = manager._resolve_asr_flush_threshold(
                threshold=1.5,
                accum_chunk_rms=[0.04, 0.0],
                current_chunk_rms=0.0,
            )
        finally:
            storage.close()

    assert single_voiced == 0.8, single_voiced
    assert voiced_then_silence == 0.8, voiced_then_silence
    assert voiced_then_voiced == 1.5, voiced_then_voiced
    assert disabled == 1.5, disabled


async def _measure_profile(*, early_flush_enabled: bool) -> _LatencyResult:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / ("early_flush_on.db" if early_flush_enabled else "early_flush_off.db"))
        try:
            asr = _LatencyMockASR()
            manager = SessionManager(
                SessionConfig(
                    audio_source=_RealtimeSpeechSource(chunk_seconds=1.0, total_seconds=4.0),
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
                    asr_early_flush_enabled=early_flush_enabled,
                    asr_early_flush_min_seconds=0.8,
                    silence_rms_threshold=0.0,
                    pre_vad_enabled=False,
                )
            )
            start_wall = time.monotonic()
            first_preview_at: list[float | None] = [None]
            first_commit_at: list[float | None] = [None]

            async def collect() -> None:
                async for ev in manager.events():
                    now = time.monotonic()
                    if ev.type == "transcript_preview" and ev.payload.get("segment") and first_preview_at[0] is None:
                        first_preview_at[0] = now
                    if ev.type == "transcript_segment" and first_commit_at[0] is None:
                        first_commit_at[0] = now

            collector = asyncio.create_task(collect())
            await manager.start()
            try:
                await _wait_for_commit(first_commit_at, timeout=12.0)
            finally:
                await manager.stop()
                try:
                    await asyncio.wait_for(collector, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    collector.cancel()
                    try:
                        await collector
                    except asyncio.CancelledError:
                        pass

            if first_preview_at[0] is None or first_commit_at[0] is None:
                raise AssertionError("profile did not produce both preview and committed events")

            return _LatencyResult(
                preview_delay_seconds=first_preview_at[0] - start_wall,
                committed_delay_seconds=first_commit_at[0] - start_wall,
                asr_calls=asr.calls,
            )
        finally:
            storage.close()


def _format_delta(before: float, after: float) -> str:
    delta = before - after
    percent = (delta / before * 100.0) if before > 0 else 0.0
    return f"{delta:.2f}s faster ({percent:.0f}% reduction)"


async def main() -> int:
    _assert_flush_threshold_cases()

    disabled = await _measure_profile(early_flush_enabled=False)
    enabled = await _measure_profile(early_flush_enabled=True)

    print("early flush disabled")
    print(f"  first preview:   {disabled.preview_delay_seconds:.2f}s")
    print(f"  first committed: {disabled.committed_delay_seconds:.2f}s")
    print(f"  ASR calls:       {disabled.asr_calls}")
    print("early flush enabled")
    print(f"  first preview:   {enabled.preview_delay_seconds:.2f}s")
    print(f"  first committed: {enabled.committed_delay_seconds:.2f}s")
    print(f"  ASR calls:       {enabled.asr_calls}")
    print(f"preview improvement:   {_format_delta(disabled.preview_delay_seconds, enabled.preview_delay_seconds)}")
    print(f"committed improvement: {_format_delta(disabled.committed_delay_seconds, enabled.committed_delay_seconds)}")

    ok = (
        enabled.preview_delay_seconds < disabled.preview_delay_seconds
        and enabled.committed_delay_seconds < disabled.committed_delay_seconds
    )
    if ok:
        print("\nOK: early flush reduces first preview and first committed latency")
        return 0
    print("\nFAIL: early flush did not improve first-result latency")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))