"""verify_suspicious_segments.py — confirm suspicious ASR segment triage.

Runs a small suite of handcrafted ASR outputs through
SessionManager._classify_asr_segment() and prints whether they are kept,
retried, or dropped. The goal is to keep obviously bad hallucination-like
segments out while routing borderline outputs into a second-pass retry.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "app/backend")

from meetingbro.asr.base import ASRSegment
from meetingbro.session.manager import SessionConfig, SessionManager


class DummyAudioSource:
    @property
    def sample_rate(self):
        return 16_000

    async def stream(self):
        if False:
            yield None

    async def aclose(self):
        return None

    def drain_drops(self):
        return 0


class DummyASR:
    pass


class DummySummarizer:
    pass


class DummyTranslator:
    pass


class DummyStorage:
    pass


def make_manager() -> SessionManager:
    return SessionManager(
        SessionConfig(
            audio_source=DummyAudioSource(),
            asr=DummyASR(),
            summarizer=DummySummarizer(),
            translator=DummyTranslator(),
            storage=DummyStorage(),
        )
    )


def run_test() -> None:
    manager = make_manager()
    cases: list[tuple[str, ASRSegment, str]] = [
        (
            "short silence hallucination",
            ASRSegment(
                start_time=0.0,
                end_time=0.20,
                text="thank you very much",
                language="en",
                confidence=0.20,
                avg_logprob=-1.4,
                no_speech_prob=0.82,
                compression_ratio=2.6,
            ),
            "drop",
        ),
        (
            "borderline noisy short utterance",
            ASRSegment(
                start_time=0.0,
                end_time=0.90,
                text="project artemis",
                language="en",
                confidence=0.35,
                avg_logprob=-1.0,
                no_speech_prob=0.68,
                compression_ratio=1.4,
            ),
            "retry",
        ),
        (
            "normal short utterance",
            ASRSegment(
                start_time=0.0,
                end_time=1.20,
                text="hello team",
                language="en",
                confidence=0.88,
                avg_logprob=-0.2,
                no_speech_prob=0.05,
                compression_ratio=1.2,
            ),
            "keep",
        ),
        (
            "normal chinese short utterance",
            ASRSegment(
                start_time=0.0,
                end_time=1.10,
                text="大家下午好",
                language="zh",
                confidence=0.86,
                avg_logprob=-0.25,
                no_speech_prob=0.04,
                compression_ratio=1.1,
            ),
            "keep",
        ),
        (
            "mixed-language normal utterance",
            ASRSegment(
                start_time=0.0,
                end_time=1.80,
                text="今天我们 review 一下 PR",
                language="zh",
                confidence=0.81,
                avg_logprob=-0.35,
                no_speech_prob=0.08,
                compression_ratio=1.3,
            ),
            "keep",
        ),
    ]

    failures: list[str] = []
    for name, seg, expected in cases:
        actual = manager._classify_asr_segment(seg)
        print(f"{name:30} -> {actual.upper():5} (expected {expected.upper()})")
        if actual != expected:
            failures.append(name)

    if failures:
        raise AssertionError(f"FAIL: unexpected suspicious-segment decisions for: {', '.join(failures)}")

    print("PASS: suspicious segment triage decisions match expected cases.")


if __name__ == "__main__":
    run_test()