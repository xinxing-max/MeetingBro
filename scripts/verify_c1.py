"""verify_c1.py — confirm that rolling/cumulative summaries never overlap.

Creates a slow mock LLM summarizer (sleeps 2 s per call) and fires many rapid
ASR-style segment updates.  At most one rolling and one cumulative task should
be in-flight at any moment.
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, "app/backend")

from meetingbro.session.manager import SessionManager, SessionConfig, _State
from meetingbro.schemas import TranscriptSegment


# ── Helpers ─────────────────────────────────────────────────────────────────

class SlowSummarizer:
    """Simulates a slow LLM that takes 2 s per summarise call."""

    def __init__(self):
        self.call_count = 0
        self.concurrent_peak = 0
        self._active = 0
        self._lock = asyncio.Lock()  # not used — peak tracked via simple int

    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        self._active += 1
        self.call_count += 1
        if self._active > self.concurrent_peak:
            self.concurrent_peak = self._active
        time.sleep(2)  # blocks the executor thread
        self._active -= 1
        return f"summary of {len(segments)} segments"


def make_segment(i: int) -> TranscriptSegment:
    return TranscriptSegment(
        id=str(i),
        meeting_id="test-meeting",
        start_time=float(i * 10),
        end_time=float(i * 10 + 9),
        text=f"segment {i}",
        original_language="en",
        confidence=0.9,
    )


# ── Test ─────────────────────────────────────────────────────────────────────

async def run_test():
    slow_summarizer = SlowSummarizer()

    # Build a minimal SessionManager with mocked everything except the summarizer.
    storage = MagicMock()
    storage.insert_segment = MagicMock()
    storage.insert_snapshot = MagicMock()
    storage.create_meeting = MagicMock()
    storage.end_meeting = MagicMock()

    translator = MagicMock()
    translator.translate = MagicMock(return_value="")

    class FakeASR:
        def transcribe(self, *a, **kw):
            return []

    class FakeAudioSource:
        @property
        def sample_rate(self):
            return 16_000

        async def stream(self):
            # Yield 20 "chunks" quickly so many summary tasks get scheduled.
            for i in range(20):
                yield type("C", (), {
                    "samples": __import__("numpy").zeros(400, dtype="float32") + 0.1,
                    "sample_rate": 16_000,
                    "start_time": float(i * 10),
                })()
                await asyncio.sleep(0)  # yield to event loop

        async def aclose(self):
            pass

        def drain_drops(self):
            return 0

    class FakeTranslator:
        def translate(self, *a, **kw):
            return ""

    cfg = SessionConfig(
        audio_source=FakeAudioSource(),
        asr=FakeASR(),
        summarizer=slow_summarizer,
        translator=FakeTranslator(),
        storage=storage,
        rolling_interval_seconds=1.0,   # fire aggressively
        cumulative_interval_seconds=1.0,
        min_segments_for_rolling=0,
        min_segments_for_cumulative=0,
        asr_accumulation_seconds=1.0,
        silence_rms_threshold=0.0,  # never skip
    )

    mgr = SessionManager(cfg)
    # Manually populate segments so summaries have something to work with.
    mgr._state.segments = [make_segment(i) for i in range(10)]
    mgr._state.elapsed_seconds = 200.0  # far past all intervals

    # Run several simultaneous _maybe_emit_rolling calls and check peak concurrency.
    tasks = [asyncio.create_task(mgr._maybe_emit_rolling()) for _ in range(10)]
    await asyncio.gather(*tasks)
    # Let any remaining executor work finish.
    await asyncio.sleep(3)

    tasks2 = [asyncio.create_task(mgr._maybe_emit_cumulative()) for _ in range(10)]
    await asyncio.gather(*tasks2)
    await asyncio.sleep(3)

    print(f"SlowSummarizer call_count    : {slow_summarizer.call_count}")
    print(f"SlowSummarizer concurrent_peak: {slow_summarizer.concurrent_peak}")

    # With the in-flight guard, only 1 call should make it through per type per cycle.
    assert slow_summarizer.concurrent_peak <= 1, (
        f"FAIL: concurrent_peak={slow_summarizer.concurrent_peak} — in-flight guard not working!"
    )
    print("PASS: at most 1 concurrent summary call at any time.")


if __name__ == "__main__":
    asyncio.run(run_test())
