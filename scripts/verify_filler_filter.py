from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter, ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.schemas import TranscriptSegment  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

SAMPLE_RATE = 16000


class _Source(AudioSource):
    def __init__(self, n_chunks: int, *, chunk_seconds: float = 0.2) -> None:
        self._n = n_chunks
        self._chunk_samples = int(SAMPLE_RATE * chunk_seconds)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def stream(self):
        samples = np.full(self._chunk_samples, 0.05, dtype=np.float32)
        for index in range(self._n):
            yield AudioChunk(samples=samples.copy(), sample_rate=SAMPLE_RATE, start_time=index * (self._chunk_samples / SAMPLE_RATE))


class _ASR(ASRAdapter):
    def __init__(self, batches: list[list[ASRSegment]]) -> None:
        self._batches = batches
        self.calls = 0

    def transcribe(self, samples, sample_rate, *, forced_language=None, offset_seconds=0.0, initial_prompt=None, quality_preset="realtime"):
        del samples, sample_rate, forced_language, offset_seconds, initial_prompt, quality_preset
        batch = self._batches[self.calls] if self.calls < len(self._batches) else []
        self.calls += 1
        return batch


class _Summarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class _Translator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


def _seg(text: str, language: str) -> ASRSegment:
    return ASRSegment(start_time=0.0, end_time=0.2, text=text, language=language, confidence=0.9)


async def _run_case(batches: list[list[ASRSegment]], *, filler_filter_enabled: bool = True, accumulation_seconds: float = 0.1, chunk_seconds: float = 0.2):
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify.db")
        try:
            manager = SessionManager(SessionConfig(
                audio_source=_Source(max(1, len(batches)), chunk_seconds=chunk_seconds),
                asr=_ASR(batches),
                summarizer=_Summarizer(),
                translator=_Translator(),
                storage=storage,
                forced_language="en",
                summary_language="en",
                rolling_interval_seconds=10000,
                cumulative_interval_seconds=10000,
                memory_interval_seconds=10000,
                min_segments_for_rolling=10000,
                min_segments_for_cumulative=10000,
                min_segments_for_memory=10000,
                asr_accumulation_seconds=accumulation_seconds,
                silence_rms_threshold=0.0,
                pre_vad_enabled=False,
                filler_filter_enabled=filler_filter_enabled,
            ))
            events: list[dict] = []

            async def collect() -> None:
                async for event in manager.events():
                    if event.type == "transcript_segment":
                        events.append(event.payload)

            collector = asyncio.create_task(collect())
            await manager.start()
            await asyncio.sleep(0.5)
            await manager.stop()
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass
            stored = storage.list_segments(manager._state.meeting_id)
            return len(events), len(stored), manager._state.filler_filtered_total
        finally:
            storage.close()


async def main() -> int:
    results: list[bool] = []
    cases = [
        ("zh filler", [_seg("嗯", "zh")], True, 0, 0, 1),
        ("en filler punct", [_seg("Um.", "en")], True, 0, 0, 1),
        ("de filler", [_seg("ähm", "de")], True, 0, 0, 1),
        ("mixed kept", [_seg("嗯，所以我建议", "zh")], True, 1, 1, 0),
        ("cross language kept", [_seg("um", "zh")], True, 1, 1, 0),
        ("uppercase drop", [_seg("UM", "en")], True, 0, 0, 1),
        ("punctuation drop", [_seg("，。", "zh")], True, 0, 0, 1),
        ("disabled passes", [_seg("嗯", "zh")], False, 1, 1, 0),
    ]
    for label, batch, enabled, expected_events, expected_stored, expected_counter in cases:
        events, stored, counter = await _run_case([batch], filler_filter_enabled=enabled)
        ok = (events, stored, counter) == (expected_events, expected_stored, expected_counter)
        print(f"[{'OK' if ok else 'FAIL'}] {label}")
        results.append(ok)

    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify-counter.db")
        try:
            manager = SessionManager(SessionConfig(audio_source=_Source(1), asr=_ASR([[]]), summarizer=_Summarizer(), translator=_Translator(), storage=storage, forced_language="en", summary_language="en", rolling_interval_seconds=10000, cumulative_interval_seconds=10000, memory_interval_seconds=10000, min_segments_for_rolling=10000, min_segments_for_cumulative=10000, min_segments_for_memory=10000, asr_accumulation_seconds=0.1, silence_rms_threshold=0.0, pre_vad_enabled=False))
            storage.create_meeting(manager._state.meeting_id, preferred_summary_language="en")
            for text, language in [("嗯", "zh"), ("Um.", "en"), ("ähm", "de"), ("，。", "zh"), ("UM", "en")]:
                if manager._is_pure_filler(text, language):
                    manager._state.filler_filtered_total += 1
            for index, text in enumerate(["keep one.", "still discussing.", "um test."]):
                await manager._persist_and_emit_segment(TranscriptSegment(id=f"manual-{index}", meeting_id=manager._state.meeting_id, start_time=float(index), end_time=float(index) + 0.2, text=text, original_language="en" if index < 2 else "zh", confidence=0.9, created_at=datetime.now(tz=timezone.utc)))
            events = len(manager._state.segments)
            stored = len(storage.list_segments(manager._state.meeting_id))
            counter = manager._state.filler_filtered_total
        finally:
            storage.close()
    ok = events == 3 and stored == 3 and counter == 5
    print(f"[{'OK' if ok else 'FAIL'}] counter accuracy")
    results.append(ok)

    events, stored, counter = await _run_case([[ _seg("嗯", "zh") ]], accumulation_seconds=10.0)
    ok = events == 0 and stored == 0 and counter == 1
    print(f"[{'OK' if ok else 'FAIL'}] final flush filler")
    results.append(ok)

    print(f"\nCounter example: a 5-segment session with 2 fillers had filler_filtered_total=2")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))