from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource, WavFileSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

DEFAULT_WAV = ROOT / "data" / "sample_en.wav"
MIN_BENCH_SECONDS = 60.0


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


class _ArrivalProbeSource(AudioSource):
    def __init__(self, inner: AudioSource) -> None:
        self._inner = inner
        self.first_chunk_wall: float | None = None
        self.chunk_arrival_times: dict[float, float] = {}

    @property
    def sample_rate(self) -> int:
        return self._inner.sample_rate

    async def stream(self):
        async for chunk in self._inner.stream():
            now = time.monotonic()
            if self.first_chunk_wall is None:
                self.first_chunk_wall = now
            self.chunk_arrival_times[chunk.start_time] = now
            yield chunk

    def drain_drops(self) -> int:
        return self._inner.drain_drops()

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass
class _TrialResult:
    accumulation_s: float
    latency_p50: float
    latency_p95: float
    latency_max: float
    total_segments: int
    total_chars: int
    dropped_filler: int
    dropped_repetition: int
    total_wall_time: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end transcript latency across accumulation windows.")
    parser.add_argument("--wav", default=str(DEFAULT_WAV), help="Path to source wav file (default: data/sample_en.wav)")
    parser.add_argument("--values", default="1.5,2.0,2.2,2.5", help="Comma-separated accumulation values in seconds")
    parser.add_argument("--whisper-size", default="small", help="Faster-Whisper model size (default: small)")
    return parser.parse_args()


def _parse_values(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--values must contain at least one accumulation value")
    return values


def _prepare_wav(path: Path) -> Path:
    info = sf.info(str(path))
    if info.duration >= MIN_BENCH_SECONDS:
        return path
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    target_frames = int(MIN_BENCH_SECONDS * sample_rate)
    repeats = max(1, int(np.ceil(target_frames / len(samples))))
    extended = np.tile(samples, repeats)[:target_frames]
    tmp = tempfile.NamedTemporaryFile(prefix="meetingbro-bench-", suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, extended, sample_rate)
    return Path(tmp.name)


async def _run_trial(adapter: FasterWhisperAdapter, *, wav_path: Path, accumulation_s: float) -> _TrialResult:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "benchmark.db")
        source = _ArrivalProbeSource(WavFileSource(wav_path, sample_rate=16_000, chunk_seconds=0.5, realtime=True))
        manager = SessionManager(SessionConfig(
            audio_source=source,
            asr=adapter,
            summarizer=_NoopSummarizer(),
            translator=_NoopTranslator(),
            storage=storage,
            summary_language="en",
            audio_chunk_seconds=0.5,
            asr_accumulation_seconds=accumulation_s,
            rolling_interval_seconds=10000,
            memory_interval_seconds=10000,
            cumulative_interval_seconds=10000,
            min_segments_for_rolling=10000,
            min_segments_for_memory=10000,
            min_segments_for_cumulative=10000,
        ))
        latencies: list[float] = []
        total_chars = 0
        repetition_drops = 0
        bench_start = time.monotonic()

        async def collect() -> None:
            nonlocal total_chars, repetition_drops
            async for event in manager.events():
                if event.type == "transcript_segment":
                    base_wall = source.first_chunk_wall or bench_start
                    emit_wall = time.monotonic()
                    latencies.append(emit_wall - base_wall - event.payload["end_time"])
                    total_chars += len(event.payload["text"])
                elif event.type == "error" and event.payload.get("code") == "asr_repetition":
                    repetition_drops += 1

        collector = asyncio.create_task(collect())
        try:
            await manager.start()
            if manager._task is not None:
                await manager._task
            await manager.stop()
        finally:
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass
            storage.close()

        p50 = statistics.median(latencies) if latencies else float("nan")
        p95 = float(np.percentile(latencies, 95)) if latencies else float("nan")
        max_latency = max(latencies) if latencies else float("nan")
        return _TrialResult(
            accumulation_s=accumulation_s,
            latency_p50=p50,
            latency_p95=p95,
            latency_max=max_latency,
            total_segments=len(latencies),
            total_chars=total_chars,
            dropped_filler=manager._state.filler_filtered_total,
            dropped_repetition=repetition_drops,
            total_wall_time=time.monotonic() - bench_start,
        )


def _fmt(value: float) -> str:
    return f"{value:.2f}" if value == value else "n/a"


def _print_table(results: list[_TrialResult]) -> None:
    print("| accumulation_s | p50 latency | p95 latency | max latency | segments | total_chars | filler_drops | repetition_drops | total_wall_time |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        print(
            f"| {result.accumulation_s:.1f} | {_fmt(result.latency_p50)} | {_fmt(result.latency_p95)} | {_fmt(result.latency_max)} | "
            f"{result.total_segments} | {result.total_chars} | {result.dropped_filler} | {result.dropped_repetition} | {_fmt(result.total_wall_time)} |"
        )


def _print_summary(results: list[_TrialResult]) -> None:
    by_value = {result.accumulation_s: result for result in results}
    lines: list[str] = []
    if 2.5 in by_value and 2.0 in by_value:
        base = by_value[2.5]
        candidate = by_value[2.0]
        lines.append(f"Median latency changes by {candidate.latency_p50 - base.latency_p50:+.2f}s going from 2.5 -> 2.0.")
        if base.total_chars > 0:
            lines.append(f"Total transcript chars: 2.0 produces {candidate.total_chars / base.total_chars * 100:.1f}% of 2.5.")
        lines.append(f"Segment count: 2.0 produces {candidate.total_segments - base.total_segments:+d} segments versus 2.5.")
    fastest = min(results, key=lambda item: item.latency_p50)
    slowest = max(results, key=lambda item: item.latency_p50)
    lines.append(f"Lowest median latency in this run: {fastest.accumulation_s:.1f}s at {_fmt(fastest.latency_p50)}s; highest: {slowest.accumulation_s:.1f}s at {_fmt(slowest.latency_p50)}s.")
    filler_span = f"Filler drops ranged from {min(r.dropped_filler for r in results)} to {max(r.dropped_filler for r in results)} across the tested windows."
    lines.append(filler_span)
    print()
    for line in lines[:5]:
        print(line)


async def _main() -> int:
    args = _parse_args()
    wav_path = Path(args.wav).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    values = _parse_values(args.values)
    prepared_wav = _prepare_wav(wav_path)
    temp_generated = prepared_wav != wav_path
    adapter = FasterWhisperAdapter(model_size=args.whisper_size, device="cpu", compute_type="int8")
    try:
        warmup_source = WavFileSource(prepared_wav, sample_rate=16_000, chunk_seconds=5.0, realtime=False)
        async for warmup_chunk in warmup_source.stream():
            adapter.transcribe(warmup_chunk.samples, warmup_chunk.sample_rate, quality_preset="realtime")
            break
        await _run_trial(adapter, wav_path=prepared_wav, accumulation_s=values[0])
        results: list[_TrialResult] = []
        for value in values:
            results.append(await _run_trial(adapter, wav_path=prepared_wav, accumulation_s=value))
        _print_table(results)
        _print_summary(results)
    finally:
        if temp_generated and prepared_wav.exists():
            prepared_wav.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))