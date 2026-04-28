from __future__ import annotations

import sys
import asyncio
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, "app/backend")

from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.main import _chunk_seconds_for_profile, _runtime_settings_from_profile  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.session.profiles import normalize_runtime_profile, runtime_profile_defaults  # noqa: E402


class _SilentSource(AudioSource):
    @property
    def sample_rate(self) -> int:
        return 16_000

    async def stream(self):
        yield AudioChunk(samples=np.zeros(1600, dtype=np.float32), sample_rate=16_000, start_time=0.0)

    async def aclose(self) -> None:
        pass


class _NoopASR:
    def transcribe(self, *args, **kwargs):
        return []


class _NoopSummarizer:
    def summarize(self, *args, **kwargs):
        return ""


class _NoopTranslator:
    def translate(self, *args, **kwargs):
        return ""


def _storage() -> MagicMock:
    storage = MagicMock()
    storage.insert_segment = MagicMock()
    storage.insert_snapshot = MagicMock()
    storage.create_meeting = MagicMock()
    storage.end_meeting = MagicMock()
    storage.update_meeting_summary_language = MagicMock()
    return storage


async def main() -> int:
    assert normalize_runtime_profile("low-latency") == "low_latency"
    assert normalize_runtime_profile("does-not-exist") == "balanced"

    balanced = runtime_profile_defaults("balanced")
    low_latency = runtime_profile_defaults("low_latency")
    robust = runtime_profile_defaults("robust")
    multilingual = runtime_profile_defaults("multilingual")
    single_language = runtime_profile_defaults("single_language")
    low_latency_settings = _runtime_settings_from_profile("low_latency")
    robust_settings = _runtime_settings_from_profile("robust")

    assert low_latency["asr_accumulation_seconds"] < balanced["asr_accumulation_seconds"]
    assert robust["asr_accumulation_seconds"] > balanced["asr_accumulation_seconds"]
    assert multilingual["language_lock_enabled"] is False
    assert single_language["language_lock_enabled"] is True
    assert _chunk_seconds_for_profile("robust") == robust["chunk_seconds"]
    assert low_latency_settings["asr_accumulation_seconds"] < robust_settings["asr_accumulation_seconds"]

    manager = SessionManager(
        SessionConfig(
            audio_source=_SilentSource(),
            audio_chunk_seconds=float(balanced["chunk_seconds"]),
            runtime_profile="balanced",
            asr=_NoopASR(),
            summarizer=_NoopSummarizer(),
            translator=_NoopTranslator(),
            storage=_storage(),
        )
    )
    manager.update_runtime_settings(
        forced_language=None,
        runtime_profile="single_language",
        runtime_settings={
            "audio_chunk_seconds": float(single_language["chunk_seconds"]),
            "asr_accumulation_seconds": float(single_language["asr_accumulation_seconds"]),
            "language_lock_enabled": bool(single_language["language_lock_enabled"]),
        },
    )

    payload = manager._session_state_payload(state="running")
    print(f"profile: {payload.runtime_profile}")
    print(f"chunk:   {payload.audio_chunk_seconds}")
    print(f"accum:   {payload.asr_accumulation_seconds}")
    print(f"lock:    {payload.language_lock_enabled}")

    ok = (
        payload.runtime_profile == "single_language"
        and payload.language_lock_enabled is True
        and payload.asr_accumulation_seconds == single_language["asr_accumulation_seconds"]
    )
    if ok:
        print("\nOK: runtime profiles normalize, map to expected settings, and update session state")
        return 0
    print("\nFAIL: runtime profile behavior regressed")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
