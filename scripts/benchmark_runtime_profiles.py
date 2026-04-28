from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter, ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.audio.vad import SileroSpeechDetector, SpeechSpan  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

SAMPLE_RATE = 16_000
PROCESSING_SECONDS = 0.04
SPEECH_AMPLITUDE = 0.035
SILENCE_AMPLITUDE = 0.0
BASE_CHUNK_SECONDS = 0.2


@dataclass(frozen=True)
class _Profile:
    name: str
    chunk_seconds: float
    accumulation_seconds: float
    pre_vad_enabled: bool
    asr_early_flush_enabled: bool
    asr_early_flush_min_seconds: float
    silence_commit_min_confidence: float
    silence_commit_min_duration_seconds: float
    pre_vad_adaptive_trailing_silence_enabled: bool


@dataclass(frozen=True)
class _ProfileResult:
    preview_delay_seconds: float
    committed_delay_seconds: float
    preview_updates: int
    committed_segments: int
    asr_calls: int
    asr_safeguard_events: int
    weak_rescue_attempts: int
    weak_rescue_emitted: int
    error_codes: dict[str, int]


class _RealtimeUtteranceSource(AudioSource):
    def __init__(self, *, chunk_seconds: float, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._closed = False
        self._rng = np.random.default_rng(seed=20260428)
        self._samples = self._build_timeline()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def total_seconds(self) -> float:
        return len(self._samples) * self._chunk_seconds

    def _build_chunk(self, amplitude: float) -> np.ndarray:
        frames = max(1, int(self._sample_rate * self._chunk_seconds))
        if amplitude <= 0.0:
            return np.zeros(frames, dtype=np.float32)
        return self._rng.normal(loc=0.0, scale=amplitude, size=frames).astype(np.float32)

    def _build_timeline(self) -> list[np.ndarray]:
        # Two short utterances separated by silence. This is enough to exercise
        # first preview, first commit, preview stabilization, and silence-boundary commit.
        chunks: list[np.ndarray] = []
        chunks.extend(self._build_chunk(SPEECH_AMPLITUDE) for _ in range(3))
        chunks.extend(self._build_chunk(SILENCE_AMPLITUDE) for _ in range(2))
        chunks.extend(self._build_chunk(SPEECH_AMPLITUDE) for _ in range(4))
        chunks.extend(self._build_chunk(SILENCE_AMPLITUDE) for _ in range(4))
        return chunks

    async def stream(self):
        start_time = 0.0
        for samples in self._samples:
            if self._closed:
                return
            await asyncio.sleep(self._chunk_seconds)
            yield AudioChunk(samples=samples.copy(), sample_rate=self._sample_rate, start_time=start_time)
            start_time += len(samples) / self._sample_rate

    async def aclose(self) -> None:
        self._closed = True


class _ProfileMockASR(ASRAdapter):
    def __init__(self, *, processing_seconds: float = PROCESSING_SECONDS) -> None:
        self._processing_seconds = processing_seconds
        self.calls = 0
        self._responses = [
            (0.0, 0.7, "We should start with the agenda", 0.93),
            (0.0, 0.9, "We should start with the agenda.", 0.95),
            (0.9, 1.7, "Then we can move to project risks", 0.91),
            (0.9, 1.9, "Then we can move to project risks.", 0.94),
        ]

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
        time.sleep(self._processing_seconds)
        self.calls += 1
        index = min(self.calls - 1, len(self._responses) - 1)
        start_time, end_time, text, confidence = self._responses[index]
        if self.calls > len(self._responses):
            text = f"{text} Window {self.calls}."
        return [
            ASRSegment(
                start_time=start_time,
                end_time=end_time,
                text=text,
                language="en",
                confidence=confidence,
            )
        ]


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


@contextlib.contextmanager
def _force_rms_pre_vad() -> None:
    original_detect = SileroSpeechDetector.detect

    def _detect(self, samples, sample_rate):
        del sample_rate
        if samples.size == 0:
            return []
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < self._energy_rms_threshold:
            return []
        return [SpeechSpan(start_frame=0, end_frame=len(samples))]

    SileroSpeechDetector.detect = _detect
    try:
        yield
    finally:
        SileroSpeechDetector.detect = original_detect


async def _wait_for_runtime(source: _RealtimeUtteranceSource, *, timeout: float) -> None:
    await asyncio.sleep(min(timeout, source.total_seconds + 1.5))


def _build_profiles() -> list[_Profile]:
    return [
        _Profile(
            name="legacy",
            chunk_seconds=2.5,
            accumulation_seconds=2.5,
            pre_vad_enabled=False,
            asr_early_flush_enabled=False,
            asr_early_flush_min_seconds=0.8,
            silence_commit_min_confidence=1.1,
            silence_commit_min_duration_seconds=10.0,
            pre_vad_adaptive_trailing_silence_enabled=False,
        ),
        _Profile(
            name="tuned_base",
            chunk_seconds=0.5,
            accumulation_seconds=1.5,
            pre_vad_enabled=False,
            asr_early_flush_enabled=False,
            asr_early_flush_min_seconds=0.8,
            silence_commit_min_confidence=1.1,
            silence_commit_min_duration_seconds=10.0,
            pre_vad_adaptive_trailing_silence_enabled=False,
        ),
        _Profile(
            name="tuned_commit",
            chunk_seconds=0.5,
            accumulation_seconds=1.5,
            pre_vad_enabled=False,
            asr_early_flush_enabled=True,
            asr_early_flush_min_seconds=0.8,
            silence_commit_min_confidence=0.75,
            silence_commit_min_duration_seconds=0.6,
            pre_vad_adaptive_trailing_silence_enabled=False,
        ),
        _Profile(
            name="full_current",
            chunk_seconds=0.5,
            accumulation_seconds=1.5,
            pre_vad_enabled=True,
            asr_early_flush_enabled=True,
            asr_early_flush_min_seconds=0.8,
            silence_commit_min_confidence=0.75,
            silence_commit_min_duration_seconds=0.6,
            pre_vad_adaptive_trailing_silence_enabled=True,
        ),
    ]


async def _run_profile(profile: _Profile) -> _ProfileResult:
    timeout = max(12.0, profile.chunk_seconds * 12.0)
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / f"runtime_profile_{profile.name}.db")
        try:
            asr = _ProfileMockASR()
            source = _RealtimeUtteranceSource(chunk_seconds=profile.chunk_seconds)
            manager = SessionManager(
                SessionConfig(
                    audio_source=source,
                    audio_chunk_seconds=profile.chunk_seconds,
                    runtime_profile=profile.name,
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
                    silence_rms_threshold=0.002,
                    pre_vad_enabled=profile.pre_vad_enabled,
                    pre_vad_threshold=0.38,
                    pre_vad_energy_rms_threshold=0.005,
                    pre_vad_trailing_silence_seconds=0.45,
                    pre_vad_adaptive_trailing_silence_enabled=profile.pre_vad_adaptive_trailing_silence_enabled,
                    pre_vad_adaptive_fast_trailing_silence_seconds=0.30,
                    pre_vad_adaptive_max_realtime_factor=0.5,
                    asr_early_flush_enabled=profile.asr_early_flush_enabled,
                    asr_early_flush_min_seconds=profile.asr_early_flush_min_seconds,
                    silence_commit_min_confidence=profile.silence_commit_min_confidence,
                    silence_commit_min_duration_seconds=profile.silence_commit_min_duration_seconds,
                    language_lock_enabled=False,
                    weak_speech_rescue_enabled=True,
                    weak_speech_rescue_fast_window_seconds=2.5,
                )
            )

            start_wall = time.monotonic()
            first_preview_at: list[Optional[float]] = [None]
            first_commit_at: list[Optional[float]] = [None]
            preview_updates = 0
            committed_segments = 0
            error_codes: Counter[str] = Counter()

            async def collect() -> None:
                nonlocal preview_updates, committed_segments
                async for ev in manager.events():
                    now = time.monotonic()
                    if ev.type == "transcript_preview" and ev.payload.get("segment"):
                        preview_updates += 1
                        if first_preview_at[0] is None:
                            first_preview_at[0] = now
                    elif ev.type == "transcript_segment":
                        committed_segments += 1
                        if first_commit_at[0] is None:
                            first_commit_at[0] = now
                    elif ev.type == "error":
                        code = str(ev.payload.get("code") or "unknown")
                        error_codes[code] += 1

            collector = asyncio.create_task(collect())
            await manager.start()
            try:
                await _wait_for_runtime(source, timeout=timeout)
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

            return _ProfileResult(
                preview_delay_seconds=first_preview_at[0] - start_wall,
                committed_delay_seconds=first_commit_at[0] - start_wall,
                preview_updates=preview_updates,
                committed_segments=committed_segments,
                asr_calls=asr.calls,
                asr_safeguard_events=manager._state.asr_safeguard_events,
                weak_rescue_attempts=manager._state.weak_rescue_attempts,
                weak_rescue_emitted=manager._state.weak_rescue_emitted,
                error_codes=dict(error_codes),
            )
        finally:
            storage.close()


def _format_delta(before: float, after: float) -> str:
    delta = before - after
    percent = (delta / before * 100.0) if before > 0 else 0.0
    return f"{delta:.2f}s ({percent:.0f}% faster)"


def _print_results(rows: list[tuple[_Profile, _ProfileResult]]) -> None:
    print("profile\tchunk\taccum\tpreview\tcommit\tpreview_updates\tcommits\tasr_calls\tsafeguard\trescue\terrors")
    for profile, result in rows:
        rescue = f"{result.weak_rescue_emitted}/{result.weak_rescue_attempts}"
        errors = "—"
        if result.error_codes:
            errors = ",".join(f"{key}:{value}" for key, value in sorted(result.error_codes.items()))
        print(
            "\t".join(
                [
                    profile.name,
                    f"{profile.chunk_seconds:.1f}",
                    f"{profile.accumulation_seconds:.1f}",
                    f"{result.preview_delay_seconds:.2f}s",
                    f"{result.committed_delay_seconds:.2f}s",
                    str(result.preview_updates),
                    str(result.committed_segments),
                    str(result.asr_calls),
                    str(result.asr_safeguard_events),
                    rescue,
                    errors,
                ]
            )
        )

    baseline = rows[0][1]
    best_preview = min(rows, key=lambda item: item[1].preview_delay_seconds)
    best_commit = min(rows, key=lambda item: item[1].committed_delay_seconds)
    print()
    print(
        f"best preview: {best_preview[0].name} at {best_preview[1].preview_delay_seconds:.2f}s "
        f"vs legacy improvement {_format_delta(baseline.preview_delay_seconds, best_preview[1].preview_delay_seconds)}"
    )
    print(
        f"best commit:  {best_commit[0].name} at {best_commit[1].committed_delay_seconds:.2f}s "
        f"vs legacy improvement {_format_delta(baseline.committed_delay_seconds, best_commit[1].committed_delay_seconds)}"
    )


async def main() -> int:
    rows: list[tuple[_Profile, _ProfileResult]] = []
    with _force_rms_pre_vad():
        for profile in _build_profiles():
            rows.append((profile, await _run_profile(profile)))

    _print_results(rows)
    baseline = rows[0][1]
    best_preview = min(rows[1:], key=lambda item: item[1].preview_delay_seconds)
    best_commit = min(rows[1:], key=lambda item: item[1].committed_delay_seconds)
    ok = (
        best_preview[1].preview_delay_seconds < baseline.preview_delay_seconds
        and best_commit[1].committed_delay_seconds < baseline.committed_delay_seconds
    )
    if ok:
        print("\nOK: runtime profile matrix shows post-legacy configs improving first-result latency")
        return 0
    print("\nFAIL: runtime profile matrix did not show any improvement over legacy")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
