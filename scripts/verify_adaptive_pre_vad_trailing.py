from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.audio.vad import PreVadSegmenter, SpeechSpan  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.2


class _Detector:
    def detect(self, samples, sample_rate):
        del sample_rate
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        if rms < 0.01:
            return []
        return [SpeechSpan(start_frame=0, end_frame=len(samples))]


class _Source(AudioSource):
    @property
    def sample_rate(self):
        return SAMPLE_RATE

    async def stream(self):
        if False:
            yield None


class _ASR(ASRAdapter):
    def transcribe(self, samples, sample_rate, **kwargs):
        del samples, sample_rate, kwargs
        return []


class _Summarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class _Translator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


def _build_manager(*, adaptive_enabled: bool) -> SessionManager:
    tmp = tempfile.TemporaryDirectory()
    storage = Storage(Path(tmp.name) / "x.db")
    manager = SessionManager(
        SessionConfig(
            audio_source=_Source(),
            asr=_ASR(),
            summarizer=_Summarizer(),
            translator=_Translator(),
            storage=storage,
            forced_language="en",
            summary_language="en",
            pre_vad_trailing_silence_seconds=0.45,
            pre_vad_adaptive_trailing_silence_enabled=adaptive_enabled,
            pre_vad_adaptive_fast_trailing_silence_seconds=0.30,
            pre_vad_adaptive_max_realtime_factor=0.5,
        )
    )
    manager._verify_tmp = tmp
    manager._verify_storage = storage
    return manager


def _measure_emit_time(trailing_silence_seconds: float) -> float | None:
    detector = _Detector()
    segmenter = PreVadSegmenter(
        sample_rate=SAMPLE_RATE,
        detector=detector,
        enabled=True,
        trailing_silence_seconds=0.45,
        max_segment_seconds=8.0,
    )
    speech = np.random.default_rng(seed=61).normal(
        loc=0.0,
        scale=0.04,
        size=int(SAMPLE_RATE * CHUNK_SECONDS),
    ).astype(np.float32)
    silence = np.zeros(int(SAMPLE_RATE * CHUNK_SECONDS), dtype=np.float32)
    elapsed = 0.0
    for samples in [speech, speech, silence, silence, silence, silence]:
        segmenter.set_trailing_silence_seconds(trailing_silence_seconds)
        out = segmenter.push(
            AudioChunk(samples=samples.copy(), sample_rate=SAMPLE_RATE, start_time=elapsed)
        )
        elapsed += CHUNK_SECONDS
        if out:
            return elapsed
    return None


def main() -> int:
    adaptive_off = _build_manager(adaptive_enabled=False)
    adaptive_on = _build_manager(adaptive_enabled=True)
    try:
        off_trailing = adaptive_off._resolve_pre_vad_trailing_silence_seconds()
        on_trailing = adaptive_on._resolve_pre_vad_trailing_silence_seconds()
        off_emit = _measure_emit_time(off_trailing)
        on_emit = _measure_emit_time(on_trailing)

        print(f"adaptive off trailing: {off_trailing:.2f}s")
        print(f"adaptive on trailing : {on_trailing:.2f}s")
        print(f"adaptive off emit    : {off_emit:.2f}s" if off_emit is not None else "adaptive off emit    : None")
        print(f"adaptive on emit     : {on_emit:.2f}s" if on_emit is not None else "adaptive on emit     : None")

        ok = (
            abs(off_trailing - 0.45) < 1e-6
            and on_trailing < off_trailing
            and on_trailing <= 0.30
            and off_emit is not None
            and on_emit is not None
            and on_emit < off_emit
        )
        if ok:
            print("\nOK: adaptive pre-VAD trailing silence emits utterances earlier when ASR is healthy")
            return 0
        print("\nFAIL: adaptive pre-VAD trailing silence did not shorten utterance emission timing")
        return 1
    finally:
        adaptive_off._verify_storage.close()
        adaptive_off._verify_tmp.cleanup()
        adaptive_on._verify_storage.close()
        adaptive_on._verify_tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())