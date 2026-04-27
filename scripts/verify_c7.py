"""verify_c7.py — confirm stop() exits cleanly within 15 s when ASR blocks for 8 s.

A mock ASR that sleeps 8 s in the executor is wired in.  We start the session,
wait 1 s for the ASR call to begin, then call stop().  The whole stop() must
complete in ≤ 15 s and must not discard the partially-transcribed segment.
"""
from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import MagicMock

sys.path.insert(0, "app/backend")

import numpy as np

from meetingbro.session.manager import SessionManager, SessionConfig
from meetingbro.asr.base import ASRSegment


# ── Mock ASR: blocks for 8 s then returns one segment ────────────────────────

class SlowASR:
    def transcribe(self, samples, sample_rate, *, forced_language=None, offset_seconds=0.0, initial_prompt=None):
        time.sleep(8)  # simulate slow model on a large audio block
        return [ASRSegment(
            start_time=0.6,
            end_time=5.0,
            text="blocked segment",
            language="en",
            confidence=0.9,
        )]


# ── Fake audio source: emits exactly one loud chunk ──────────────────────────

class OneChunkSource:
    @property
    def sample_rate(self):
        return 16_000

    async def stream(self):
        # Emit one loud chunk — ASR will block for 8 s processing it.
        yield type("C", (), {
            "samples": np.ones(40_000, dtype=np.float32) * 0.1,
            "sample_rate": 16_000,
            "start_time": 0.0,
        })()
        # Generator finishes here; the outer loop will exit after ASR returns.

    async def aclose(self):
        pass  # _stopped flag drives the exit; aclose is a no-op here

    def drain_drops(self):
        return 0


# ── Test ─────────────────────────────────────────────────────────────────────

async def run_test():
    slow_asr = SlowASR()

    storage = MagicMock()
    storage.insert_segment = MagicMock()
    storage.insert_snapshot = MagicMock()
    storage.create_meeting = MagicMock()
    storage.end_meeting = MagicMock()

    class FakeSummarizer:
        def summarize(self, *a, **kw):
            return "summary"

    class FakeTranslator:
        def translate(self, *a, **kw):
            return ""

    cfg = SessionConfig(
        audio_source=OneChunkSource(),
        asr=slow_asr,
        summarizer=FakeSummarizer(),
        translator=FakeTranslator(),
        storage=storage,
        rolling_interval_seconds=9999.0,
        cumulative_interval_seconds=9999.0,
        memory_interval_seconds=9999.0,
        asr_accumulation_seconds=2.4,
        silence_rms_threshold=0.0,
        asr_overlap_seconds=0.5,
    )

    mgr = SessionManager(cfg)

    # Start the session pipeline properly so mgr._task is set.
    await mgr.start()

    # Wait for ASR to start (the chunk is emitted immediately, ASR blocks for 8 s).
    await asyncio.sleep(1)

    # Time how long stop() takes.
    t0 = time.monotonic()
    await mgr.stop()
    elapsed = time.monotonic() - t0

    print(f"stop() elapsed: {elapsed:.1f} s")
    print(f"segments collected: {len(mgr._state.segments)}")

    assert elapsed < 14.0, f"FAIL: stop() took {elapsed:.1f} s — exceeded deadline (ASR was 8 s, should finish well under 14 s)"
    print("PASS: stop() completed within 15 s.")

    # The ASR finished (returned after 8 s) and the segment should have been collected.
    # (The task checks _stopped after the executor returns, then breaks the loop.)
    assert len(mgr._state.segments) >= 1, (
        "FAIL: expected at least 1 segment from the 8 s ASR call — segment was discarded"
    )
    print(f"PASS: segment not lost ({len(mgr._state.segments)} segment(s) collected).")


if __name__ == "__main__":
    asyncio.run(run_test())
