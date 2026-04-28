from __future__ import annotations

import asyncio
import json
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
from meetingbro.summarization.llm import LLMSummarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

SAMPLE_RATE = 16_000
VOICED = np.full(SAMPLE_RATE // 10, 0.02, dtype=np.float32)


class _Source(AudioSource):
    def __init__(self, chunks: int) -> None:
        self._chunks = chunks
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def stream(self):
        start = 0.0
        for _ in range(self._chunks):
            if self._closed:
                return
            yield AudioChunk(samples=VOICED.copy(), sample_rate=self.sample_rate, start_time=start)
            start += len(VOICED) / self.sample_rate
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        self._closed = True


class _ASR(ASRAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, samples, sample_rate, *, forced_language=None, offset_seconds=0.0, initial_prompt=None, quality_preset="realtime"):
        self.calls += 1
        if self.calls > 1:
            return []
        return [ASRSegment(0.0, len(samples) / sample_rate, "Roadmap updated today.", "en", 0.95)]


class _Translator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


class _CompatClient:
    def __init__(self, response):
        self.response = response

    def chat(self, **kwargs):
        del kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FinalizeSummarizer(LLMSummarizer):
    def __init__(self, response) -> None:
        super().__init__()
        self._compatible_client = _CompatClient(response)
        self.finalize_calls = 0
        self.recorded_segments_count = 0
        self.recorded_segment_ids: list[str] = []
        self.recorded_meeting_memory = None

    def _ensure_client(self):
        return "openai_compatible"

    def finalize_meeting(self, *args, **kwargs):
        self.finalize_calls += 1
        segments = list(args[0]) if args else []
        self.recorded_segments_count = len(segments)
        self.recorded_segment_ids = [segment.id for segment in segments]
        self.recorded_meeting_memory = kwargs.get("meeting_memory")
        return super().finalize_meeting(*args, **kwargs)


async def _run_long_meeting(summarizer: _FinalizeSummarizer, *, meeting_memory: str | None):
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify_finalize_long.db")
        try:
            manager = SessionManager(SessionConfig(
                audio_source=_Source(0),
                asr=_ASR(),
                summarizer=summarizer,
                translator=_Translator(),
                storage=storage,
                forced_language="en",
                summary_language="en",
                summary_tail_seconds=120.0,
            ))
            storage.create_meeting(manager._state.meeting_id, preferred_summary_language="en")
            manager._state.segments = [
                TranscriptSegment(
                    id=f"seg-{index}",
                    meeting_id=manager._state.meeting_id,
                    start_time=index * 30.0,
                    end_time=index * 30.0 + 20.0,
                    text=f"Segment {index}",
                    original_language="en",
                    confidence=0.95,
                    created_at=datetime.now(tz=timezone.utc),
                )
                for index in range(60)
            ]
            manager._state.elapsed_seconds = manager._state.segments[-1].end_time
            manager._state.latest_meeting_memory = meeting_memory
            await manager._emit_final()
            return storage.list_snapshots(manager._state.meeting_id), summarizer.recorded_segments_count, summarizer.recorded_segment_ids, summarizer.recorded_meeting_memory
        finally:
            storage.close()


async def _run(chunks: int, summarizer: _FinalizeSummarizer):
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify_finalize.db")
        try:
            manager = SessionManager(SessionConfig(
                audio_source=_Source(chunks),
                asr=_ASR(),
                summarizer=summarizer,
                translator=_Translator(),
                storage=storage,
                forced_language="en",
                summary_language="en",
                vocabulary_hint="Anthropic, MeetingBro",
                rolling_interval_seconds=10_000,
                cumulative_interval_seconds=10_000,
                memory_interval_seconds=10_000,
                min_segments_for_rolling=10_000,
                min_segments_for_cumulative=10_000,
                min_segments_for_memory=10_000,
                asr_accumulation_seconds=60.0,
                pre_vad_enabled=False,
                silence_rms_threshold=0.0,
            ))
            events: list[dict] = []

            async def collect() -> None:
                async for event in manager.events():
                    if event.type == "error":
                        events.append(event.payload)

            collector = asyncio.create_task(collect())
            await manager.start()
            await asyncio.sleep(0.35 if chunks else 0.1)
            await manager.stop()
            meeting_id = manager._state.meeting_id
            snapshots = storage.list_snapshots(meeting_id)
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass
            return snapshots, events, summarizer.finalize_calls
        finally:
            storage.close()


async def main() -> int:
    failed = False
    happy = json.dumps({
        "chapters": [
            {"title": "Roadmap review", "time_start": 0.0, "time_end": 12.0, "summary": "The team reviewed roadmap priorities."},
            {"title": "Deadline alignment", "time_start": 12.0, "time_end": 24.0, "summary": "They aligned on the release deadline."},
        ],
        "action_items": [
            {"text": "Share updated roadmap", "assignee": "Alice", "due": "2026-05-01"},
            {"text": "Confirm dependencies", "assignee": None, "due": None},
            {"text": "Book follow-up", "assignee": "Bob", "due": "2026-05-03"},
        ],
        "final_summary": "The team reviewed the roadmap, aligned on priorities, confirmed the deadline, discussed dependencies, and agreed on a follow-up.",
    })
    snapshots, errors, calls = await _run(8, _FinalizeSummarizer(happy))
    latest = {snap.summary_type: snap for snap in snapshots if snap.is_latest}
    chapters = json.loads(latest["chapter_list"].content)
    actions = json.loads(latest["action_item_list"].content)
    ok = {'final_summary', 'chapter_list', 'action_item_list'} <= set(latest) and len(chapters) == 2 and len(actions) == 3 and calls == 1
    print(f"[{'OK' if ok else 'FAIL'}] happy path")
    failed = failed or not ok

    snapshots, errors, _ = await _run(8, _FinalizeSummarizer(RuntimeError("LLM unavailable")))
    latest = {snap.summary_type: snap for snap in snapshots if snap.is_latest}
    ok = any(err.get('code') == 'finalize_failed' for err in errors) and 'final_summary' in latest and 'chapter_list' not in latest and 'action_item_list' not in latest
    print(f"[{'OK' if ok else 'FAIL'}] fallback path")
    failed = failed or not ok

    snapshots, errors, _ = await _run(8, _FinalizeSummarizer("not valid json{{"))
    latest = {snap.summary_type: snap for snap in snapshots if snap.is_latest}
    ok = any(err.get('code') == 'finalize_failed' for err in errors) and 'final_summary' in latest and 'chapter_list' not in latest and 'action_item_list' not in latest
    print(f"[{'OK' if ok else 'FAIL'}] malformed json")
    failed = failed or not ok

    snapshots, errors, calls = await _run(0, _FinalizeSummarizer(happy))
    ok = not snapshots and not errors and calls == 0
    print(f"[{'OK' if ok else 'FAIL'}] empty meeting")
    failed = failed or not ok
    long_memory = "## Topics\n- Q2 roadmap\n## Decisions\n- Allocate 2 engineers"
    snapshots, segment_count, segment_ids, recorded_memory = await _run_long_meeting(_FinalizeSummarizer(happy), meeting_memory=long_memory)
    latest = {snap.summary_type: snap for snap in snapshots if snap.is_latest}
    ok = {'final_summary', 'chapter_list', 'action_item_list'} <= set(latest) and segment_count < 60 and segment_count == len(segment_ids) and all(int(segment_id.split('-')[-1]) >= 55 for segment_id in segment_ids) and recorded_memory == long_memory
    print(f"[{'OK' if ok else 'FAIL'}] long meeting + memory (finalize segments={segment_count} / full=60)")
    failed = failed or not ok

    summarizer = _FinalizeSummarizer(happy)
    snapshots, segment_count, _, recorded_memory = await _run_long_meeting(summarizer, meeting_memory=None)
    latest = {snap.summary_type: snap for snap in snapshots if snap.is_latest}
    ok = {'final_summary', 'chapter_list', 'action_item_list'} <= set(latest) and segment_count < 60 and recorded_memory is None and summarizer.finalize_calls == 1
    print(f"[{'OK' if ok else 'FAIL'}] long meeting + empty memory (finalize segments={segment_count} / full=60)")
    failed = failed or not ok
    print(f"chapters[0]: {chapters[0]}")
    print(f"action_items[0]: {actions[0]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))