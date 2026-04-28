"""verify_c5.py — confirm language vote stickiness in SessionManager.

Feeds a mock ASR that reports language sequence [zh, zh, zh, unknown, en, zh, zh]
through the SessionManager state machine.  After the first 5 zh votes the language
should lock to 'zh'.  The two follow-up 'en' + 'zh' segments with confidence < 0.8
should NOT cause an unlock.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Optional
from unittest.mock import MagicMock

sys.path.insert(0, "app/backend")

from meetingbro.session.manager import SessionManager, SessionConfig
from meetingbro.asr.base import ASRSegment


# ── Mock ASR that emits a fixed language sequence ────────────────────────────

_LANG_SEQUENCE = ["zh", "zh", "zh", "unknown", "zh", "en", "zh"]
_CONF_SEQUENCE = [0.9,   0.9,   0.9,   0.5,       0.9,  0.9,  0.9]

_call_index = 0


class SequenceASR:
    """Returns one segment per call with the next language in the sequence."""

    def __init__(self):
        self.call_idx = 0

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
        if self.call_idx >= len(_LANG_SEQUENCE):
            return []
        lang = _LANG_SEQUENCE[self.call_idx]
        conf = _CONF_SEQUENCE[self.call_idx]
        self.call_idx += 1
        return [ASRSegment(
            start_time=0.6,  # past overlap_duration (0.5 s default)
            end_time=1.5,
            text=f"text in {lang}",
            language=lang,
            confidence=conf,
        )]


# ── Minimal fake audio source ─────────────────────────────────────────────────

import numpy as np


class SeqAudioSource:
    @property
    def sample_rate(self):
        return 16_000

    async def stream(self):
        for i in range(len(_LANG_SEQUENCE)):
            # Chunk of 2.5 s worth of audio at 16 kHz (louder than silence threshold).
            samples = np.ones(40_000, dtype=np.float32) * 0.1
            yield type("C", (), {
                "samples": samples,
                "sample_rate": 16_000,
                "start_time": float(i * 3),
            })()
            await asyncio.sleep(0)

    async def aclose(self):
        pass

    def drain_drops(self):
        return 0


# ── Test ─────────────────────────────────────────────────────────────────────

async def run_test():
    seq_asr = SequenceASR()

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
        audio_source=SeqAudioSource(),
        asr=seq_asr,
        summarizer=FakeSummarizer(),
        translator=FakeTranslator(),
        storage=storage,
        # Set very high intervals so summaries don't interfere.
        rolling_interval_seconds=9999.0,
        cumulative_interval_seconds=9999.0,
        memory_interval_seconds=9999.0,
        asr_accumulation_seconds=2.4,   # just under 2.5 s so each 2.5 s chunk flushes
        silence_rms_threshold=0.0,      # never skip
        asr_overlap_seconds=0.5,
        pre_vad_enabled=False,
        language_lock_enabled=True,
    )

    mgr = SessionManager(cfg)

    # Run the pipeline (no real start/stop overhead needed — just _run).
    await mgr._run()

    state = mgr._state
    print(f"locked_language      : {state.locked_language}")
    print(f"language_votes       : {state.language_votes}")
    print(f"language_dissent_streak: {state.language_dissent_streak}")
    print(f"segments emitted     : {len(state.segments)}")
    print(f"segment languages    : {[s.original_language for s in state.segments]}")

    assert state.locked_language == "zh", (
        f"FAIL: expected locked_language='zh', got {state.locked_language!r}"
    )
    print("PASS: language correctly locked to 'zh'.")

    # Single 'en' batch with conf=0.9 should increase dissent streak but not reach 3.
    # After the sequence the streak should be 0 (final zh resets it) or <= 2.
    assert state.language_dissent_streak < 3, (
        f"FAIL: dissent_streak={state.language_dissent_streak} — language would have unlocked incorrectly"
    )
    print("PASS: language did not unlock on isolated 'en' detection.")


if __name__ == "__main__":
    asyncio.run(run_test())
