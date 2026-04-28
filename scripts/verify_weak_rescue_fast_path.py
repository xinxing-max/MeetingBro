from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, "app/backend")

from meetingbro.asr.base import ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioChunk  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402


class WeakAudioSource:
    @property
    def sample_rate(self):
        return 16_000

    async def stream(self):
        for i in range(3):
            noise = np.random.default_rng(seed=100 + i).normal(loc=0.0, scale=0.006, size=16_000).astype(np.float32)
            yield AudioChunk(samples=noise, sample_rate=16_000, start_time=float(i))
            await asyncio.sleep(0)

    async def aclose(self):
        pass

    def drain_drops(self):
        return 0


class MockASR:
    def __init__(self):
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
        del samples, sample_rate, forced_language, offset_seconds, initial_prompt, quality_preset
        self.calls += 1
        return [
            ASRSegment(
                start_time=0.0,
                end_time=1.0,
                text="quiet rescued speech.",
                language="en",
                confidence=0.85,
            )
        ]


class FakeSummarizer:
    def summarize(self, *a, **kw):
        return ""


class FakeTranslator:
    def translate(self, *a, **kw):
        return ""


async def main():
    storage = MagicMock()
    storage.insert_segment = MagicMock()
    storage.insert_snapshot = MagicMock()
    storage.create_meeting = MagicMock()
    storage.end_meeting = MagicMock()
    asr = MockASR()

    cfg = SessionConfig(
        audio_source=WeakAudioSource(),
        asr=asr,
        summarizer=FakeSummarizer(),
        translator=FakeTranslator(),
        storage=storage,
        forced_language="en",
        summary_language="en",
        rolling_interval_seconds=9999.0,
        cumulative_interval_seconds=9999.0,
        memory_interval_seconds=9999.0,
        pre_vad_enabled=True,
        pre_vad_conditioning_enabled=False,
        pre_vad_threshold=0.99,
        pre_vad_energy_rms_threshold=0.1,
        asr_accumulation_seconds=2.0,
        silence_rms_threshold=0.0,
        weak_speech_rescue_enabled=True,
        weak_speech_rescue_rms_min=0.001,
        weak_speech_rescue_rms_max=0.02,
        weak_speech_rescue_fast_rms_max=0.01,
        weak_speech_rescue_fast_window_seconds=2.5,
        weak_speech_rescue_window_seconds=6.0,
        weak_speech_rescue_cooldown_seconds=0.0,
    )

    mgr = SessionManager(cfg)
    await mgr._run()

    print(f"ASR calls              : {asr.calls}")
    print(f"segments emitted       : {len(mgr._state.segments)}")
    print(f"weak rescue attempts   : {mgr._state.weak_rescue_attempts}")
    print(f"weak rescue emitted    : {mgr._state.weak_rescue_emitted}")

    assert mgr._state.weak_rescue_attempts >= 1, "expected at least one rescue attempt"
    assert mgr._state.weak_rescue_emitted >= 1, "expected rescue to emit a segment"
    assert any(seg.text == "quiet rescued speech." for seg in mgr._state.segments)
    print("PASS: weak-speech rescue fast-path recovered short quiet audio.")


if __name__ == "__main__":
    asyncio.run(main())