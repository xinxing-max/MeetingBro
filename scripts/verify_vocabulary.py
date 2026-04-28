from __future__ import annotations

import asyncio
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

VOCABULARY = "Anthropic, MeetingBro"


class _SyntheticAudioSource(AudioSource):
    def __init__(self, n_chunks: int, sample_rate: int = 16_000) -> None:
        self._n, self._sr = n_chunks, sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sr

    async def stream(self):
        rng = np.random.default_rng(seed=7)
        for i in range(self._n):
            yield AudioChunk(rng.normal(scale=0.05, size=self._sr).astype(np.float32), self._sr, float(i))


class RecordingMockASR(ASRAdapter):
    def __init__(self) -> None:
        self.initial_prompts: list[str | None] = []
        self.calls = 0

    def transcribe(self, samples, sample_rate, *, forced_language=None, offset_seconds=0.0, initial_prompt=None, quality_preset="realtime"):
        self.calls += 1
        self.initial_prompts.append(initial_prompt)
        return [ASRSegment(0.1, 0.8, f"Segment {self.calls} mentions Anthropic and MeetingBro.", "en", 0.92)]


class RecordingMockSummarizer(Summarizer):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        self.calls.append((kind, vocabulary))
        return f"{kind}: {len(segments)} segment(s)"


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "verify.db")
        asr = RecordingMockASR()
        summarizer = RecordingMockSummarizer()
        try:
            manager = SessionManager(SessionConfig(
                audio_source=_SyntheticAudioSource(5),
                asr=asr,
                summarizer=summarizer,
                translator=_NoopTranslator(),
                storage=storage,
                forced_language="en",
                summary_language="en",
                vocabulary_hint=VOCABULARY,
                asr_accumulation_seconds=0.5,
                rolling_interval_seconds=1.0,
                memory_interval_seconds=1.0,
                cumulative_interval_seconds=1.0,
                min_segments_for_rolling=1,
                min_segments_for_memory=1,
                min_segments_for_cumulative=1,
                silence_rms_threshold=0.0,
                pre_vad_enabled=False,
            ))
            await manager.start()
            await asyncio.sleep(1.5)
            await manager.stop()
        finally:
            storage.close()

    saw_asr_vocabulary = any(prompt == VOCABULARY for prompt in asr.initial_prompts)
    saw_summary_vocabulary = any(vocabulary == VOCABULARY for _, vocabulary in summarizer.calls)
    print(f"ASR calls: {asr.calls}")
    print(f"Summary calls: {len(summarizer.calls)}")
    print(f"ASR initial prompts: {asr.initial_prompts}")
    print(f"Summary vocabulary calls: {summarizer.calls}")
    if saw_asr_vocabulary and saw_summary_vocabulary:
        print("\nOK: vocabulary hint reached ASR and summary generation")
        return 0
    print("\nFAIL: vocabulary hint did not reach all expected consumers")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))