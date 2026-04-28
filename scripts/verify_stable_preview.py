from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter, ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

CONFIRMED_TEXT = "We should start now."
FOLLOW_UP_TEXT = "Next topic is budget planning."


class _SilentAudioSource(AudioSource):
    def __init__(self, *, n_chunks: int, sample_rate: int = 16_000) -> None:
        self._n = n_chunks
        self._sr = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sr

    async def stream(self):
        rng = np.random.default_rng(seed=7)
        for i in range(self._n):
            samples = rng.normal(scale=0.05, size=self._sr).astype(np.float32)
            yield AudioChunk(samples=samples, sample_rate=self._sr, start_time=float(i))


class _PreviewMockASR(ASRAdapter):
    def __init__(self) -> None:
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
        self.calls += 1
        if self.calls == 1:
            return [
                ASRSegment(
                    start_time=0.2,
                    end_time=1.6,
                    text=CONFIRMED_TEXT,
                    language="en",
                    confidence=0.9,
                )
            ]
        return [
            ASRSegment(
                start_time=0.2,
                end_time=1.8,
                text=FOLLOW_UP_TEXT,
                language="en",
                confidence=0.88,
            )
        ]


class _NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None):
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify_stable_preview.db")
        try:
            manager = SessionManager(
                SessionConfig(
                    audio_source=_SilentAudioSource(n_chunks=6),
                    asr=_PreviewMockASR(),
                    summarizer=_NoopSummarizer(),
                    translator=_NoopTranslator(),
                    storage=storage,
                    forced_language="en",
                    summary_language="en",
                    rolling_interval_seconds=10_000,
                    cumulative_interval_seconds=10_000,
                    memory_interval_seconds=10_000,
                    min_segments_for_rolling=10_000,
                    min_segments_for_memory=10_000,
                    min_segments_for_cumulative=10_000,
                    asr_accumulation_seconds=2.5,
                    silence_rms_threshold=0.0,
                    pre_vad_enabled=False,
                )
            )
            collected: list[tuple[str, dict]] = []

            async def collect() -> None:
                async for ev in manager.events():
                    collected.append((ev.type, ev.payload))

            collector = asyncio.create_task(collect())
            await manager.start()
            await asyncio.sleep(1.5)
            await manager.stop()
            collector.cancel()
            try:
                await collector
            except (asyncio.CancelledError, Exception):
                pass

            committed = [payload for event_type, payload in collected if event_type == "transcript_segment"]
            previews = [payload["segment"] for event_type, payload in collected if event_type == "transcript_preview" and payload.get("segment")]

            confirmed_commits = [seg for seg in committed if seg["text"].strip() == CONFIRMED_TEXT]
            follow_up_commits = [seg for seg in committed if seg["text"].strip() == FOLLOW_UP_TEXT]
            confirmed_preview_before_commit = False
            confirmed_commit_index = next(
                (idx for idx, (event_type, payload) in enumerate(collected) if event_type == "transcript_segment" and payload["text"].strip() == CONFIRMED_TEXT),
                None,
            )
            if confirmed_commit_index is not None:
                confirmed_preview_before_commit = any(
                    event_type == "transcript_preview"
                    and payload.get("segment")
                    and payload["segment"]["text"].strip() == CONFIRMED_TEXT
                    for event_type, payload in collected[:confirmed_commit_index]
                )

            follow_up_preview_before_commit = False
            follow_up_commit_index = next(
                (idx for idx, (event_type, payload) in enumerate(collected) if event_type == "transcript_segment" and payload["text"].strip() == FOLLOW_UP_TEXT),
                None,
            )
            if follow_up_commit_index is not None:
                follow_up_preview_before_commit = any(
                    event_type == "transcript_preview"
                    and payload.get("segment")
                    and payload["segment"]["text"].strip() == FOLLOW_UP_TEXT
                    for event_type, payload in collected[:follow_up_commit_index]
                )

            print(f"transcript_segment events: {len(committed)}")
            print(f"preview events with text: {len(previews)}")
            print(f"confirmed sentence commits: {len(confirmed_commits)}")
            print(f"follow-up sentence commits: {len(follow_up_commits)}")
            print(f"confirmed preview observed before commit: {confirmed_preview_before_commit}")
            print(f"follow-up preview observed before commit: {follow_up_preview_before_commit}")

            ok = (
                len(confirmed_commits) == 1
                and len(follow_up_commits) == 1
                and confirmed_preview_before_commit
                and follow_up_preview_before_commit
            )
            if ok:
                print("\nOK: stable preview delays the last complete segment until the next window or stop")
                return 0
            print("\nFAIL: stable preview delayed-commit behavior regressed")
            return 1
        finally:
            storage.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
