from __future__ import annotations

import asyncio
import logging
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
_SENTENCES = [
    "We should start with the agenda.",
    "The budget review follows after that.",
    "Then we can move to project risks.",
    "Finally we will confirm the next actions.",
]


@dataclass(frozen=True)
class _Profile:
    name: str
    chunk_seconds: float
    accumulation_seconds: float


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
        text = _SENTENCES[min(self.calls - 1, len(_SENTENCES) - 1)]
        return [
            ASRSegment(
                start_time=0.1,
                end_time=max(0.2, duration - 0.1),
                text=text,
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


async def _wait_for_commit(commit_at: list[Optional[float]], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if commit_at[0] is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for first committed segment")


async def _measure_profile(profile: _Profile) -> _LatencyResult:
    total_seconds = max(profile.accumulation_seconds * 2.0 + profile.chunk_seconds, profile.chunk_seconds * 4.0)
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / f"latency_{profile.name}.db")
        try:
            asr = _LatencyMockASR()
            manager = SessionManager(
                SessionConfig(
                    audio_source=_RealtimeSpeechSource(
                        chunk_seconds=profile.chunk_seconds,
                        total_seconds=total_seconds,
                    ),
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
                    asr_accumulation_seconds=profile.accumulation_seconds,
                    silence_rms_threshold=0.0,
                    pre_vad_enabled=False,
                )
            )
            start_wall = time.monotonic()
            first_preview_at: list[Optional[float]] = [None]
            first_commit_at: list[Optional[float]] = [None]

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
                await _wait_for_commit(first_commit_at, timeout=max(12.0, total_seconds + 4.0))
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
                raise AssertionError(f"profile {profile.name} did not produce both preview and committed events")

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
    legacy = _Profile(name="legacy", chunk_seconds=2.5, accumulation_seconds=2.5)
    tuned = _Profile(name="tuned", chunk_seconds=0.5, accumulation_seconds=1.5)

    legacy_result = await _measure_profile(legacy)
    tuned_result = await _measure_profile(tuned)

    print(f"legacy  chunk={legacy.chunk_seconds:.1f}s accum={legacy.accumulation_seconds:.1f}s")
    print(f"  first preview:   {legacy_result.preview_delay_seconds:.2f}s")
    print(f"  first committed: {legacy_result.committed_delay_seconds:.2f}s")
    print(f"  ASR calls:       {legacy_result.asr_calls}")
    print(f"tuned   chunk={tuned.chunk_seconds:.1f}s accum={tuned.accumulation_seconds:.1f}s")
    print(f"  first preview:   {tuned_result.preview_delay_seconds:.2f}s")
    print(f"  first committed: {tuned_result.committed_delay_seconds:.2f}s")
    print(f"  ASR calls:       {tuned_result.asr_calls}")
    print(f"preview improvement:   {_format_delta(legacy_result.preview_delay_seconds, tuned_result.preview_delay_seconds)}")
    print(f"committed improvement: {_format_delta(legacy_result.committed_delay_seconds, tuned_result.committed_delay_seconds)}")

    ok = (
        tuned_result.preview_delay_seconds < legacy_result.preview_delay_seconds
        and tuned_result.committed_delay_seconds < legacy_result.committed_delay_seconds
    )
    if ok:
        print("\nOK: tuned runtime defaults reduce first preview and first committed latency")
        return 0
    print("\nFAIL: tuned runtime defaults did not improve first-result latency")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
