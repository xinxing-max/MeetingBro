from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

SAMPLE_RATE = 16_000
VOICED_CHUNK = np.full(SAMPLE_RATE // 50, 0.02, dtype=np.float32)
SILENT_CHUNK = np.full(SAMPLE_RATE // 50, 0.0001, dtype=np.float32)
TICK_SECONDS = 0.02
SLOW_SECONDS = 0.05
SILENT_SECONDS = 0.15
ALL_SILENT_SECONDS = 0.45
DROP_WINDOW_SECONDS = 0.30
DROP_BURST_COUNT = 50


class _NoopASR(ASRAdapter):
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
        return []


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


class _BaseSource(AudioSource):
    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def aclose(self) -> None:
        self._closed = True


class _PauseRecoverSource(_BaseSource):
    async def stream(self):
        start_time = 0.0
        for _ in range(3):
            if self._closed:
                return
            yield AudioChunk(samples=VOICED_CHUNK.copy(), sample_rate=self.sample_rate, start_time=start_time)
            start_time += len(VOICED_CHUNK) / self.sample_rate
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.25)
        if self._closed:
            return
        yield AudioChunk(samples=VOICED_CHUNK.copy(), sample_rate=self.sample_rate, start_time=start_time)


class _AllSilentSource(_BaseSource):
    async def stream(self):
        start_time = 0.0
        for _ in range(60):
            if self._closed:
                return
            yield AudioChunk(samples=SILENT_CHUNK.copy(), sample_rate=self.sample_rate, start_time=start_time)
            start_time += len(SILENT_CHUNK) / self.sample_rate
            await asyncio.sleep(0.01)
        for _ in range(30):
            if self._closed:
                return
            yield AudioChunk(samples=VOICED_CHUNK.copy(), sample_rate=self.sample_rate, start_time=start_time)
            start_time += len(VOICED_CHUNK) / self.sample_rate
            await asyncio.sleep(0.03)


class _DropBurstSource(_BaseSource):
    def __init__(self) -> None:
        super().__init__()
        self._burst_deadline: float | None = None

    async def stream(self):
        loop = asyncio.get_running_loop()
        self._burst_deadline = loop.time() + 0.22
        start_time = 0.0
        while not self._closed and loop.time() < self._burst_deadline + 0.24:
            yield AudioChunk(samples=VOICED_CHUNK.copy(), sample_rate=self.sample_rate, start_time=start_time)
            start_time += len(VOICED_CHUNK) / self.sample_rate
            await asyncio.sleep(0.01)

    def drain_drops(self) -> int:
        if self._burst_deadline is None or self._closed:
            return 0
        if asyncio.get_running_loop().time() < self._burst_deadline:
            return 8
        return 0


class _CrashSource(_BaseSource):
    async def stream(self):
        yield AudioChunk(samples=VOICED_CHUNK.copy(), sample_rate=self.sample_rate, start_time=0.0)
        await asyncio.sleep(0.01)
        raise RuntimeError("device disconnected")


def _error_count(collected: dict[str, list[dict]], code: str) -> int:
    return sum(1 for payload in collected.get("error", []) if payload.get("code") == code)


def _error_messages(collected: dict[str, list[dict]], code: str) -> list[str]:
    return [payload.get("message", "") for payload in collected.get("error", []) if payload.get("code") == code]


async def _wait_for(predicate: Callable[[], bool], *, timeout: float, label: str) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {label}")


async def _run_manager(
    source: AudioSource,
    *,
    wait_for: Callable[[dict[str, list[dict]]], bool] | None,
    timeout: float,
    settle_seconds: float = 0.0,
    run_seconds: float | None = None,
) -> dict[str, list[dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify_watchdog.db")
        try:
            manager = SessionManager(
                SessionConfig(
                    audio_source=source,
                    asr=_NoopASR(),
                    summarizer=_NoopSummarizer(),
                    translator=_NoopTranslator(),
                    storage=storage,
                    forced_language="en",
                    summary_language="en",
                    watchdog_enabled=True,
                    watchdog_tick_seconds=TICK_SECONDS,
                    watchdog_no_chunk_warning_seconds=SLOW_SECONDS,
                    watchdog_no_chunk_error_seconds=SILENT_SECONDS,
                    watchdog_all_silent_seconds=ALL_SILENT_SECONDS,
                    watchdog_drop_burst_count=DROP_BURST_COUNT,
                    watchdog_drop_burst_window_seconds=DROP_WINDOW_SECONDS,
                    rolling_interval_seconds=10_000,
                    cumulative_interval_seconds=10_000,
                    memory_interval_seconds=10_000,
                    min_segments_for_rolling=10_000,
                    min_segments_for_memory=10_000,
                    min_segments_for_cumulative=10_000,
                    asr_accumulation_seconds=60.0,
                    pre_vad_enabled=False,
                    silence_rms_threshold=0.002,
                )
            )
            collected: dict[str, list[dict]] = {}

            async def collect() -> None:
                async for event in manager.events():
                    collected.setdefault(event.type, []).append(event.payload)

            collector = asyncio.create_task(collect())
            await manager.start()
            try:
                if run_seconds is not None:
                    await asyncio.sleep(run_seconds)
                elif wait_for is not None:
                    await _wait_for(lambda: wait_for(collected), timeout=timeout, label="scenario completion")
                if settle_seconds > 0:
                    await asyncio.sleep(settle_seconds)
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
            return collected
        finally:
            storage.close()


async def test_chunk_pause_recovery() -> tuple[bool, str]:
    collected = await _run_manager(
        _PauseRecoverSource(),
        wait_for=lambda events: _error_count(events, "audio_recovered") >= 1,
        timeout=1.5,
    )
    ok = (
        _error_count(collected, "audio_source_slow") == 1
        and _error_count(collected, "audio_source_silent") == 1
        and _error_count(collected, "audio_recovered") == 1
        and any(payload.get("recovered_code") == "audio_source_silent" for payload in collected.get("error", []) if payload.get("code") == "audio_recovered")
    )
    detail = (
        f"slow={_error_count(collected, 'audio_source_slow')} "
        f"silent={_error_count(collected, 'audio_source_silent')} "
        f"recovered={_error_count(collected, 'audio_recovered')}"
    )
    return ok, detail


async def test_all_silent() -> tuple[bool, str]:
    collected = await _run_manager(
        _AllSilentSource(),
        wait_for=None,
        timeout=1.5,
        run_seconds=1.6,
    )
    ok = (
        _error_count(collected, "audio_all_silent") == 1
        and sum(1 for payload in collected.get("error", []) if payload.get("code") == "audio_recovered" and payload.get("recovered_code") == "audio_all_silent") == 1
    )
    detail = f"all_silent={_error_count(collected, 'audio_all_silent')} recovered={_error_count(collected, 'audio_recovered')}"
    return ok, detail


async def test_drop_burst() -> tuple[bool, str]:
    collected = await _run_manager(
        _DropBurstSource(),
        wait_for=lambda events: _error_count(events, "audio_recovered") >= 1,
        timeout=1.5,
    )
    ok = (
        _error_count(collected, "audio_drops_sustained") == 1
        and sum(1 for payload in collected.get("error", []) if payload.get("code") == "audio_recovered" and payload.get("recovered_code") == "audio_drops_sustained") == 1
    )
    detail = (
        f"audio_drops_sustained={_error_count(collected, 'audio_drops_sustained')} "
        f"audio_drop={_error_count(collected, 'audio_drop')} recovered={_error_count(collected, 'audio_recovered')}"
    )
    return ok, detail


async def test_source_crash() -> tuple[bool, str]:
    collected = await _run_manager(
        _CrashSource(),
        wait_for=lambda events: _error_count(events, "audio_source_crashed") >= 1,
        timeout=1.5,
    )
    ok = (
        _error_count(collected, "audio_source_crashed") == 1
        and any("device disconnected" in message for message in _error_messages(collected, "audio_source_crashed"))
    )
    detail = f"audio_source_crashed={_error_count(collected, 'audio_source_crashed')}"
    return ok, detail


async def main() -> int:
    scenarios = [
        ("chunk pause recovery", test_chunk_pause_recovery),
        ("all silent", test_all_silent),
        ("drop burst", test_drop_burst),
        ("source crash", test_source_crash),
    ]
    failed = False
    for name, scenario in scenarios:
        try:
            ok, detail = await scenario()
        except Exception as exc:
            ok = False
            detail = str(exc)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
