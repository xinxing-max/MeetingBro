from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, "app/backend")

from meetingbro.asr.base import ASRSegment  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402


_LANG_SEQUENCE = ["zh", "zh", "zh", "unknown", "zh", "en", "zh"]
_CONF_SEQUENCE = [0.9, 0.9, 0.9, 0.5, 0.9, 0.9, 0.9]


class SequenceASR:
    def __init__(self):
        self.call_idx = 0
        self.forced_language_args: list[str | None] = []

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
        del samples, sample_rate, offset_seconds, initial_prompt, quality_preset
        self.forced_language_args.append(forced_language)
        if self.call_idx >= len(_LANG_SEQUENCE):
            return []
        lang = _LANG_SEQUENCE[self.call_idx]
        conf = _CONF_SEQUENCE[self.call_idx]
        self.call_idx += 1
        return [
            ASRSegment(
                start_time=0.6,
                end_time=1.5,
                text=f"text in {lang}",
                language=lang,
                confidence=conf,
            )
        ]


class SeqAudioSource:
    @property
    def sample_rate(self):
        return 16_000

    async def stream(self):
        for i in range(len(_LANG_SEQUENCE)):
            samples = np.ones(40_000, dtype=np.float32) * 0.1
            yield type(
                "C",
                (),
                {
                    "samples": samples,
                    "sample_rate": 16_000,
                    "start_time": float(i * 3),
                },
            )()
            await asyncio.sleep(0)

    async def aclose(self):
        pass

    def drain_drops(self):
        return 0


class FakeSummarizer:
    def summarize(self, *a, **kw):
        return "summary"


class FakeTranslator:
    def translate(self, *a, **kw):
        return ""


async def _run_case(*, forced_language: str | None, language_lock_enabled: bool | None = None):
    seq_asr = SequenceASR()
    storage = MagicMock()
    storage.insert_segment = MagicMock()
    storage.insert_snapshot = MagicMock()
    storage.create_meeting = MagicMock()
    storage.end_meeting = MagicMock()

    kwargs = {}
    if language_lock_enabled is not None:
        kwargs["language_lock_enabled"] = language_lock_enabled

    cfg = SessionConfig(
        audio_source=SeqAudioSource(),
        asr=seq_asr,
        summarizer=FakeSummarizer(),
        translator=FakeTranslator(),
        storage=storage,
        forced_language=forced_language,
        rolling_interval_seconds=9999.0,
        cumulative_interval_seconds=9999.0,
        memory_interval_seconds=9999.0,
        asr_accumulation_seconds=2.4,
        silence_rms_threshold=0.0,
        asr_overlap_seconds=0.5,
        pre_vad_enabled=False,
        **kwargs,
    )
    mgr = SessionManager(cfg)
    await mgr._run()
    return mgr._state, seq_asr.forced_language_args


async def main() -> int:
    auto_state, auto_forced = await _run_case(forced_language=None)
    locked_state, locked_forced = await _run_case(forced_language=None, language_lock_enabled=True)
    forced_state, forced_args = await _run_case(forced_language="en")

    print(f"auto locked_language   : {auto_state.locked_language}")
    print(f"auto forced args       : {auto_forced}")
    print(f"locked locked_language : {locked_state.locked_language}")
    print(f"locked forced args     : {locked_forced}")
    print(f"forced locked_language : {forced_state.locked_language}")
    print(f"forced args            : {forced_args}")

    ok = (
        auto_state.locked_language is None
        and all(arg is None for arg in auto_forced)
        and locked_state.locked_language == "zh"
        and forced_state.locked_language is None
        and all(arg == "en" for arg in forced_args)
    )
    if ok:
        print("\nOK: language lock defaults off, can be enabled explicitly, and is bypassed in forced-language mode")
        return 0
    print("\nFAIL: default language-lock behavior regressed")
    return 1


if __name__ == "__main__":
    asyncio.run(main())
