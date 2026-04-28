from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter, ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.exporter import export_meeting  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402


SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.05
SAMPLES = (
    0.03
    * np.sin(
        2.0
        * np.pi
        * 440.0
        * np.arange(int(SAMPLE_RATE * CHUNK_SECONDS), dtype=np.float32)
        / SAMPLE_RATE
    )
).astype(np.float32)


class _StaticASR(ASRAdapter):
    def __init__(self, *, text: str | None = None, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
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
        if self.error is not None:
            raise self.error
        if not self.text:
            return []
        duration = len(samples) / sample_rate
        return [
            ASRSegment(
                start_time=max(0.0, duration - 0.4),
                end_time=duration,
                text=self.text,
                language=forced_language or "en",
                confidence=0.92,
            )
        ]


class _CountingSummarizer(Summarizer):
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        self.calls += 1
        return ""


class _NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


class _RealtimeSource(AudioSource):
    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def stream(self):
        start = 0.0
        for _ in range(12):
            yield AudioChunk(samples=SAMPLES.copy(), sample_rate=SAMPLE_RATE, start_time=start)
            start += CHUNK_SECONDS
            await asyncio.sleep(CHUNK_SECONDS)

    async def aclose(self) -> None:
        return


@dataclass
class _CaseResult:
    previews: list[dict]
    committed: list[dict]
    db_segments: int
    transcript_text: str
    summary_text: str
    fast_preview_skipped: int
    shared_calls: int
    preview_calls: int
    summarizer_calls: int


async def _run_case(
    *,
    shared_asr: _StaticASR,
    preview_asr: _StaticASR | None,
    formal_asr_rtf: float | None = None,
) -> _CaseResult:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "fast_preview.db")
        summarizer = _CountingSummarizer()
        manager = SessionManager(
            SessionConfig(
                audio_source=_RealtimeSource(),
                asr=shared_asr,
                preview_asr=preview_asr,
                summarizer=summarizer,
                translator=_NoopTranslator(),
                storage=storage,
                forced_language="en",
                summary_language="en",
                asr_accumulation_seconds=60.0,
                fast_preview_enabled=True,
                fast_preview_interval_seconds=0.1,
                fast_preview_window_seconds=0.3,
                fast_preview_min_rms=0.001,
                fast_preview_max_backlog_seconds=10.0,
                pre_vad_enabled=False,
                silence_rms_threshold=1.0,
                rolling_interval_seconds=10_000,
                cumulative_interval_seconds=10_000,
                memory_interval_seconds=10_000,
                min_segments_for_rolling=10_000,
                min_segments_for_memory=10_000,
                min_segments_for_cumulative=10_000,
            )
        )
        manager._state.asr_realtime_factor = formal_asr_rtf
        collected: dict[str, list[dict]] = {}

        async def collect() -> None:
            async for event in manager.events():
                collected.setdefault(event.type, []).append(event.payload)

        collector = asyncio.create_task(collect())
        await manager.start()
        await asyncio.sleep(0.7)
        await manager.stop()
        await asyncio.wait_for(collector, timeout=1.0)

        storage_segments = storage.list_segments(manager.meeting_id)
        export = export_meeting(
            storage,
            meeting_id=manager.meeting_id,
            export_root=Path(tmp) / "exports",
            export_dir=Path(tmp) / "export",
        )
        transcript_text = (Path(export.export_dir) / "transcript.md").read_text(encoding="utf-8")
        summary_text = (Path(export.export_dir) / "summary.md").read_text(encoding="utf-8")
        storage.close()

    previews = [
        payload.get("segment")
        for payload in collected.get("transcript_preview", [])
        if payload.get("segment") is not None
    ]
    committed = collected.get("transcript_segment", [])
    return _CaseResult(
        previews=previews,
        committed=committed,
        db_segments=len(storage_segments),
        transcript_text=transcript_text,
        summary_text=summary_text,
        fast_preview_skipped=manager._state.fast_preview_skipped,
        shared_calls=shared_asr.calls,
        preview_calls=0 if preview_asr is None else preview_asr.calls,
        summarizer_calls=summarizer.calls,
    )


def _is_temporary_only(result: _CaseResult, expected_text: str) -> bool:
    return (
        len(result.previews) >= 1
        and result.previews[-1]["text"] == expected_text
        and len(result.committed) == 0
        and result.db_segments == 0
        and result.summarizer_calls == 0
        and "_No transcript segments saved._" in result.transcript_text
        and expected_text not in result.transcript_text
        and expected_text not in result.summary_text
    )


async def main() -> int:
    dedicated_shared = _StaticASR(text=None)
    dedicated_preview = _StaticASR(text="preview dedicated")
    dedicated = await _run_case(
        shared_asr=dedicated_shared,
        preview_asr=dedicated_preview,
    )
    ok_dedicated = (
        _is_temporary_only(dedicated, "preview dedicated")
        and dedicated.shared_calls == 0
        and dedicated.preview_calls >= 1
    )
    print(
        "dedicated:",
        f"previews={len(dedicated.previews)}",
        f"committed={len(dedicated.committed)}",
        f"db_segments={dedicated.db_segments}",
        f"fast_preview_skipped={dedicated.fast_preview_skipped}",
        f"shared_calls={dedicated.shared_calls}",
        f"preview_calls={dedicated.preview_calls}",
        f"summarizer_calls={dedicated.summarizer_calls}",
    )

    fallback_shared = _StaticASR(text="preview fallback")
    failing_preview = _StaticASR(error=RuntimeError("preview backend unavailable"))
    fallback = await _run_case(
        shared_asr=fallback_shared,
        preview_asr=failing_preview,
    )
    ok_fallback = (
        _is_temporary_only(fallback, "preview fallback")
        and fallback.shared_calls >= 1
        and fallback.preview_calls >= 1
    )
    print(
        "fallback:",
        f"previews={len(fallback.previews)}",
        f"committed={len(fallback.committed)}",
        f"db_segments={fallback.db_segments}",
        f"fast_preview_skipped={fallback.fast_preview_skipped}",
        f"shared_calls={fallback.shared_calls}",
        f"preview_calls={fallback.preview_calls}",
        f"summarizer_calls={fallback.summarizer_calls}",
    )

    overloaded_shared = _StaticASR(text=None)
    overloaded_preview = _StaticASR(text="should not run")
    overloaded = await _run_case(
        shared_asr=overloaded_shared,
        preview_asr=overloaded_preview,
        formal_asr_rtf=0.90,
    )
    ok_overloaded = (
        len(overloaded.previews) == 0
        and len(overloaded.committed) == 0
        and overloaded.db_segments == 0
        and overloaded.fast_preview_skipped >= 1
        and overloaded.shared_calls == 0
        and overloaded.preview_calls == 0
        and overloaded.summarizer_calls == 0
    )
    print(
        "formal-rtf-guard:",
        f"previews={len(overloaded.previews)}",
        f"committed={len(overloaded.committed)}",
        f"db_segments={overloaded.db_segments}",
        f"fast_preview_skipped={overloaded.fast_preview_skipped}",
        f"shared_calls={overloaded.shared_calls}",
        f"preview_calls={overloaded.preview_calls}",
        f"summarizer_calls={overloaded.summarizer_calls}",
    )

    if not ok_dedicated:
        print("FAIL: dedicated preview backend should stay temporary and bypass shared ASR")
        return 1
    if not ok_fallback:
        print("FAIL: preview backend fallback should stay temporary and use shared ASR")
        return 1
    if not ok_overloaded:
        print("FAIL: dedicated preview should be skipped when formal ASR RTF exceeds the preview threshold")
        return 1
    print("OK: fast preview stays temporary with dedicated backend and shared fallback, and skips on formal ASR overload")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))