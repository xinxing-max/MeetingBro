from __future__ import annotations

import asyncio
import io
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.base import ASRAdapter, ASRSegment  # noqa: E402
from meetingbro.audio.capture import AudioChunk, AudioSource  # noqa: E402
from meetingbro.exporter import export_meeting  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionEvent, SessionManager  # noqa: E402
from meetingbro.schemas import TranscriptSegment  # noqa: E402
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
    preview_payloads: list[dict]
    committed: list[dict]
    db_segments: int
    transcript_text: str
    summary_text: str
    fast_preview_skipped: int
    preview_continued_during_formal: int
    shared_calls: int
    preview_calls: int
    summarizer_calls: int


async def _run_case(
    *,
    shared_asr: _StaticASR,
    preview_asr: _StaticASR | None,
    formal_asr_rtf: float | None = None,
    preview_asr_backend_name: str = "unknown",
    preview_asr_fallback_on_error: bool = True,
    seed_formal_pending: bool = False,
) -> _CaseResult:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "fast_preview.db")
        summarizer = _CountingSummarizer()
        manager = SessionManager(
            SessionConfig(
                audio_source=_RealtimeSource(),
                asr=shared_asr,
                preview_asr=preview_asr,
                preview_asr_backend_name=preview_asr_backend_name,
                preview_asr_fallback_on_error=preview_asr_fallback_on_error,
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
        if seed_formal_pending:
            # The fast-preview loop only needs pending formal work to be truthy.
            # Use a low-confidence/short preview candidate rather than a pending
            # segment so the session finalizer never persists this synthetic sentinel.
            manager._state.preview_candidate_segment = TranscriptSegment(
                id=str(uuid.uuid4()),
                meeting_id=manager.meeting_id,
                start_time=0.0,
                end_time=0.1,
                text="synthetic formal pending sentinel",
                original_language="en",
                confidence=0.0,
                created_at=datetime.now(),
            )
        await asyncio.sleep(0.7)
        if seed_formal_pending:
            manager._state.preview_candidate_segment = None
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

    preview_payloads: list[dict] = collected.get("transcript_preview", [])
    previews = [
        payload.get("segment")
        for payload in preview_payloads
        if payload.get("segment") is not None
    ]
    committed = collected.get("transcript_segment", [])
    return _CaseResult(
        previews=previews,
        preview_payloads=preview_payloads,
        committed=committed,
        db_segments=len(storage_segments),
        transcript_text=transcript_text,
        summary_text=summary_text,
        fast_preview_skipped=manager._state.fast_preview_skipped,
        preview_continued_during_formal=manager._state.preview_continued_during_formal,
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
    overloaded = await _run_case(
        shared_asr=overloaded_shared,
        preview_asr=None,
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

    # ------------------------------------------------------------------
    # Qwen3 backend env-builder tests (no real model required)
    # ------------------------------------------------------------------

    import os as _os
    from meetingbro.asr.qwen3_asr_adapter import Qwen3ASRAdapter  # noqa: E402

    # (a) Selecting qwen3 without model dir raises RuntimeError.
    _env_before = _os.environ.copy()
    _os.environ["MEETINGBRO_PREVIEW_ASR_BACKEND"] = "qwen3"
    _os.environ.pop("MEETINGBRO_PREVIEW_QWEN3_MODEL_DIR", None)
    try:
        from meetingbro.main import _build_preview_asr as _bpa
        _bpa()
        print("FAIL: qwen3 backend without model dir should raise RuntimeError")
        return 1
    except RuntimeError as _e:
        if "MEETINGBRO_PREVIEW_QWEN3_MODEL_DIR" in str(_e):
            print("OK: qwen3 backend without model dir raised expected RuntimeError")
        else:
            print(f"FAIL: unexpected RuntimeError message: {_e}")
            return 1
    finally:
        _os.environ.clear()
        _os.environ.update(_env_before)

    # (b) Constructing Qwen3ASRAdapter with a non-existent model dir fails clearly.
    try:
        Qwen3ASRAdapter(model_dir="/nonexistent/path/does/not/exist")
        print("FAIL: Qwen3ASRAdapter with missing model dir should raise RuntimeError")
        return 1
    except (RuntimeError, ImportError) as _e:
        # RuntimeError = sherpa-onnx missing or model dir missing; both are fine.
        print(f"OK: Qwen3ASRAdapter with bad model dir raised {type(_e).__name__}: {_e}")

    # (c) Qwen3 adapter, when selected, plugs into preview path and stays temporary
    #     (tested via the fake _StaticASR class which already models that).
    #     The dedicated + fallback cases above already cover the "temporary-only"
    #     contract.  Re-confirm here by passing a _StaticASR as preview_asr so we
    #     exercise the same SessionManager code path that a real Qwen3ASRAdapter
    #     would use.
    qwen3_fake_preview = _StaticASR(text="qwen3 preview text")
    qwen3_result = await _run_case(
        shared_asr=_StaticASR(text=None),
        preview_asr=qwen3_fake_preview,
        preview_asr_backend_name="qwen3",
    )
    ok_qwen3_temporary = _is_temporary_only(qwen3_result, "qwen3 preview text")

    # Verify preview payload metadata fields.
    non_null_payloads = [p for p in qwen3_result.preview_payloads if p.get("segment") is not None]
    ok_qwen3_meta = (
        len(non_null_payloads) >= 1
        and all(p.get("preview_backend") == "qwen3" for p in non_null_payloads)
        and all(p.get("preview_is_experimental") is True for p in non_null_payloads)
        and all(p.get("preview_quality_note") == "experimental_fast_preview" for p in non_null_payloads)
    )
    # Committed segments must NOT carry preview metadata.
    ok_no_meta_in_committed = all(
        "preview_backend" not in p and "preview_is_experimental" not in p
        for p in qwen3_result.committed
    )
    print(
        "qwen3-fake-backend:",
        f"previews={len(qwen3_result.previews)}",
        f"committed={len(qwen3_result.committed)}",
        f"db_segments={qwen3_result.db_segments}",
        f"preview_calls={qwen3_result.preview_calls}",
        f"non_null_payloads={len(non_null_payloads)}",
        f"meta_ok={ok_qwen3_meta}",
    )
    if not ok_qwen3_temporary:
        print("FAIL: qwen3 preview should be temporary-only and not committed or persisted")
        return 1
    if not ok_qwen3_meta:
        sample = non_null_payloads[0] if non_null_payloads else {}
        print(f"FAIL: qwen3 preview payload missing or wrong metadata fields. Sample: {sample}")
        return 1
    if not ok_no_meta_in_committed:
        print("FAIL: committed transcript_segment payloads should not contain preview metadata")
        return 1
    print("OK: qwen3 preview (via fake adapter) is temporary-only, not persisted, metadata correct")

    # Verify that non-qwen3 (faster_whisper) preview is NOT marked experimental.
    fw_fake_preview = _StaticASR(text="fw preview text")
    fw_result = await _run_case(
        shared_asr=_StaticASR(text=None),
        preview_asr=fw_fake_preview,
        preview_asr_backend_name="faster_whisper",
    )
    fw_non_null = [p for p in fw_result.preview_payloads if p.get("segment") is not None]
    ok_fw_not_experimental = (
        len(fw_non_null) >= 1
        and all(p.get("preview_backend") == "faster_whisper" for p in fw_non_null)
        and all(p.get("preview_is_experimental") is False for p in fw_non_null)
    )
    print(
        "faster-whisper-preview-meta:",
        f"non_null_payloads={len(fw_non_null)}",
        f"not_experimental={ok_fw_not_experimental}",
    )
    if not ok_fw_not_experimental:
        print("FAIL: faster_whisper preview payload should not be marked experimental")
        return 1
    print("OK: faster_whisper preview payload correctly not marked experimental")

    # (d) Fallback works when Qwen3 adapter raises.
    qwen3_fallback_shared = _StaticASR(text="qwen3 fallback shared")
    qwen3_failing_preview = _StaticASR(error=RuntimeError("sherpa-onnx decode error"))
    qwen3_fallback = await _run_case(
        shared_asr=qwen3_fallback_shared,
        preview_asr=qwen3_failing_preview,
    )
    ok_qwen3_fallback = (
        _is_temporary_only(qwen3_fallback, "qwen3 fallback shared")
        and qwen3_fallback.shared_calls >= 1
        and qwen3_fallback.preview_calls >= 1
    )
    print(
        "qwen3-fallback:",
        f"previews={len(qwen3_fallback.previews)}",
        f"committed={len(qwen3_fallback.committed)}",
        f"db_segments={qwen3_fallback.db_segments}",
        f"shared_calls={qwen3_fallback.shared_calls}",
        f"preview_calls={qwen3_fallback.preview_calls}",
    )
    if not ok_qwen3_fallback:
        print("FAIL: SessionManager should fall back to shared ASR when qwen3 adapter raises")
        return 1
    print("OK: SessionManager falls back to shared preview path when qwen3 adapter raises")

    # (e) Optional no-fallback mode: Qwen3 failures should not borrow the formal
    # shared ASR path, preserving formal Whisper capacity under preview errors.
    qwen3_no_fallback_shared = _StaticASR(text="should not be used")
    qwen3_no_fallback_preview = _StaticASR(error=RuntimeError("qwen3 transient failure"))
    qwen3_no_fallback = await _run_case(
        shared_asr=qwen3_no_fallback_shared,
        preview_asr=qwen3_no_fallback_preview,
        preview_asr_backend_name="qwen3",
        preview_asr_fallback_on_error=False,
    )
    ok_qwen3_no_fallback = (
        len(qwen3_no_fallback.previews) == 0
        and len(qwen3_no_fallback.committed) == 0
        and qwen3_no_fallback.db_segments == 0
        and qwen3_no_fallback.shared_calls == 0
        and qwen3_no_fallback.preview_calls >= 1
        and qwen3_no_fallback.summarizer_calls == 0
    )
    print(
        "qwen3-no-fallback:",
        f"previews={len(qwen3_no_fallback.previews)}",
        f"committed={len(qwen3_no_fallback.committed)}",
        f"db_segments={qwen3_no_fallback.db_segments}",
        f"shared_calls={qwen3_no_fallback.shared_calls}",
        f"preview_calls={qwen3_no_fallback.preview_calls}",
    )
    if not ok_qwen3_no_fallback:
        print("FAIL: qwen3 no-fallback mode should skip preview without using shared ASR")
        return 1
    print("OK: qwen3 no-fallback mode skips preview without borrowing shared ASR")

    # (f) Qwen3 live preview should continue while formal text is pending.
    # This is the key UX split: Qwen keeps the live subtitle moving while
    # Whisper/formal work settles behind it.
    qwen3_pending_shared = _StaticASR(text=None)
    qwen3_pending_preview = _StaticASR(text="qwen3 live while formal pending")
    qwen3_pending = await _run_case(
        shared_asr=qwen3_pending_shared,
        preview_asr=qwen3_pending_preview,
        preview_asr_backend_name="qwen3",
        preview_asr_fallback_on_error=False,
        seed_formal_pending=True,
    )
    ok_qwen3_pending = (
        _is_temporary_only(qwen3_pending, "qwen3 live while formal pending")
        and qwen3_pending.shared_calls == 0
        and qwen3_pending.preview_calls >= 1
        and qwen3_pending.preview_continued_during_formal >= 1
    )
    print(
        "qwen3-pending-formal-live:",
        f"previews={len(qwen3_pending.previews)}",
        f"committed={len(qwen3_pending.committed)}",
        f"shared_calls={qwen3_pending.shared_calls}",
        f"preview_calls={qwen3_pending.preview_calls}",
        f"continued={qwen3_pending.preview_continued_during_formal}",
    )
    if not ok_qwen3_pending:
        print("FAIL: qwen3 dedicated preview should continue while formal text is pending")
        return 1
    print("OK: qwen3 dedicated preview continues while formal text is pending")

    # (g) Non-Qwen dedicated preview remains conservative while formal text is pending.
    # This prevents another Whisper-family preview backend from competing with the
    # formal Whisper lane in the old way.
    non_qwen_pending_shared = _StaticASR(text=None)
    non_qwen_pending_preview = _StaticASR(text="should not preview while pending")
    non_qwen_pending = await _run_case(
        shared_asr=non_qwen_pending_shared,
        preview_asr=non_qwen_pending_preview,
        preview_asr_backend_name="faster_whisper",
        seed_formal_pending=True,
    )
    ok_non_qwen_pending = (
        len(non_qwen_pending.previews) == 0
        and non_qwen_pending.preview_calls == 0
        and non_qwen_pending.preview_continued_during_formal == 0
    )
    print(
        "non-qwen-pending-formal-guard:",
        f"previews={len(non_qwen_pending.previews)}",
        f"preview_calls={non_qwen_pending.preview_calls}",
        f"continued={non_qwen_pending.preview_continued_during_formal}",
    )
    if not ok_non_qwen_pending:
        print("FAIL: non-qwen preview should still pause while formal text is pending")
        return 1
    print("OK: non-qwen preview still pauses while formal text is pending")

    # ------------------------------------------------------------------
    # Local unit tests for script filter and filler suppression
    # (no SessionManager, no model required)
    # ------------------------------------------------------------------

    from meetingbro.asr.qwen3_asr_adapter import _wrong_script, _FILLER_SET, _is_garbage  # noqa: E402

    _script_cases: list[tuple[str, str | None, bool, str]] = [
        # (text, forced_language, expect_wrong, label)
        # --- en: reject Arabic-dominant
        ("مرحبا بالعالم كيف حال", "en", True, "en rejects Arabic-dominant"),
        # --- en: reject CJK-dominant
        ("你好世界今天天气真好", "en", True, "en rejects CJK-dominant"),
        # --- en: allow Latin
        ("Hello world, this is fine.", "en", False, "en allows Latin"),
        # --- de: reject Arabic-dominant
        ("مرحبا بالعالم كيف حال", "de", True, "de rejects Arabic-dominant"),
        # --- de: reject CJK-dominant
        ("你好世界今天天气真好", "de", True, "de rejects CJK-dominant"),
        # --- de: allow Latin
        ("Guten Morgen, wie geht es Ihnen?", "de", False, "de allows Latin"),
        # --- zh: reject Arabic-dominant
        ("مرحبا بالعالم كيف حال", "zh", True, "zh rejects Arabic-dominant"),
        # --- zh: reject Cyrillic-dominant
        ("Привет мир как дела", "zh", True, "zh rejects Cyrillic-dominant"),
        # --- zh: reject pure Latin (no CJK at all)
        ("hello world", "zh", True, "zh rejects pure Latin"),
        # --- zh: allow CJK-heavy
        ("你好世界今天天气真好", "zh", False, "zh allows CJK-heavy"),
        # --- None: pass through (Arabic)
        ("مرحبا بالعالم", None, False, "auto/None passes through Arabic"),
        # --- None: pass through (CJK)
        ("你好世界", None, False, "auto/None passes through CJK"),
        # --- en: reject Cyrillic-dominant
        ("Привет мир как дела сегодня", "en", True, "en rejects Cyrillic-dominant"),
    ]

    script_failures = 0
    for text, lang, expect, label in _script_cases:
        got = _wrong_script(text, lang)
        status = "OK" if got == expect else "FAIL"
        if got != expect:
            script_failures += 1
        print(f"  script-filter [{status}] {label}: wrong_script={got} (expected {expect})")

    if script_failures:
        print(f"FAIL: {script_failures} script-filter case(s) did not match expected result")
        return 1
    print("OK: all script-filter cases passed")

    # Filler suppression checks.
    _filler_cases: list[tuple[str, bool, str]] = [
        ("嗯。", True, "Chinese filler '嗯。' should be in filler set"),
        ("嗯", True, "Chinese filler '嗯' should be in filler set"),
        ("啊", True, "Chinese filler '啊' should be in filler set"),
        ("啊。", True, "Chinese filler '啊。' should be in filler set"),
        ("um", True, "English filler 'um' should be in filler set"),
        ("uh", True, "English filler 'uh' should be in filler set"),
        ("ah", True, "English filler 'ah' should be in filler set"),
        ("hello", False, "'hello' should NOT be in filler set"),
        ("你好", False, "'你好' should NOT be in filler set"),
    ]

    filler_failures = 0
    for text, expect_suppressed, label in _filler_cases:
        got = text.lower() in _FILLER_SET
        status = "OK" if got == expect_suppressed else "FAIL"
        if got != expect_suppressed:
            filler_failures += 1
        print(f"  filler [{status}] {label}: in_set={got} (expected {expect_suppressed})")

    if filler_failures:
        print(f"FAIL: {filler_failures} filler suppression case(s) failed")
        return 1
    print("OK: all filler suppression cases passed")

    # Garbage-filter checks.
    _garbage_cases: list[tuple[str, bool, str]] = [
        ("aaaaaaaaa", True, "single char repeated → garbage"),
        ("........", True, "punctuation repeated → garbage"),
        ("x", True, "single non-CJK char → garbage"),
        ("hello hello hello hello", True, "word repeated 3+ times → garbage"),
        ("你好", False, "valid CJK → not garbage"),
        ("Hello, how are you?", False, "normal English → not garbage"),
        ("Good morning everyone", False, "normal English phrase → not garbage"),
        ("嗯嗯嗯嗯嗯嗯嗯", True, "CJK char repeated → garbage"),
    ]

    garbage_failures = 0
    for text, expect_garbage, label in _garbage_cases:
        got = _is_garbage(text)
        status = "OK" if got == expect_garbage else "FAIL"
        if got != expect_garbage:
            garbage_failures += 1
        print(f"  garbage [{status}] {label}: is_garbage={got} (expected {expect_garbage})")

    if garbage_failures:
        print(f"FAIL: {garbage_failures} garbage-filter case(s) failed")
        return 1
    print("OK: all garbage-filter cases passed")

    # ------------------------------------------------------------------
    # Qwen3ASRAdapter.transcribe() unit tests — fake recognizer, no model
    # Verify: (1) confidence is 0.60, (2) _is_garbage is active in transcribe,
    # (3) filler suppression fires before garbage check.
    # ------------------------------------------------------------------

    import types as _types  # noqa: F401
    import numpy as _np_local

    class _FakeResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeStream:
        def __init__(self, text: str) -> None:
            self.result = _FakeResult(text)

        def accept_waveform(self, sr: int, samples) -> None:
            pass

    class _FakeRecognizer:
        def __init__(self, text: str) -> None:
            self._text = text

        def create_stream(self) -> _FakeStream:
            return _FakeStream(self._text)

        def decode_stream(self, stream: _FakeStream) -> None:
            pass

    def _make_adapter(suppress_fillers: bool = True, filter_language_script: bool = True):
        """Build a Qwen3ASRAdapter bypassing __init__ model-dir checks."""
        adapter = object.__new__(Qwen3ASRAdapter)
        adapter._suppress_fillers = suppress_fillers
        adapter._filter_language_script = filter_language_script
        adapter._recognizer = None  # replaced per-test below
        return adapter

    _samples_1s = _np_local.zeros(16_000, dtype=_np_local.float32)

    # (1) confidence is 0.60 for normal speech text
    _a1 = _make_adapter()
    _a1._recognizer = _FakeRecognizer("Hello world how are you today")
    _r1 = _a1.transcribe(_samples_1s, 16_000)
    if not _r1:
        print("FAIL: transcribe() returned empty for normal text (confidence test)")
        return 1
    _got_conf = _r1[0].confidence
    if _got_conf != 0.60:
        print(f"FAIL: expected confidence=0.60, got {_got_conf}")
        return 1
    print(f"OK: Qwen3ASRAdapter.transcribe() confidence={_got_conf} (expected 0.60)")

    # (2) _is_garbage is active: single-char-repeated text returns []
    _a2 = _make_adapter()
    _a2._recognizer = _FakeRecognizer("aaaaaaaaaa")
    _r2 = _a2.transcribe(_samples_1s, 16_000)
    if _r2:
        print(f"FAIL: garbage text 'aaaaaaaaaa' should be filtered, got {_r2}")
        return 1
    print("OK: garbage text 'aaaaaaaaaa' correctly filtered by _is_garbage inside transcribe()")

    # (3) filler suppression fires before garbage check — 'um' returns []
    _a3 = _make_adapter()
    _a3._recognizer = _FakeRecognizer("um")
    _r3 = _a3.transcribe(_samples_1s, 16_000)
    if _r3:
        print(f"FAIL: filler 'um' should be suppressed, got {_r3}")
        return 1
    print("OK: filler 'um' correctly suppressed inside transcribe()")

    # (4) wrong-script filter fires: CJK text for en returns []
    _a4 = _make_adapter()
    _a4._recognizer = _FakeRecognizer("你好世界今天天气真好")
    _r4 = _a4.transcribe(_samples_1s, 16_000, forced_language="en")
    if _r4:
        print(f"FAIL: CJK text for lang=en should be filtered by wrong-script, got {_r4}")
        return 1
    print("OK: CJK text for lang=en correctly filtered by wrong-script inside transcribe()")

    print("OK: all Qwen3ASRAdapter.transcribe() unit tests passed")

    # ------------------------------------------------------------------
    # _looks_incomplete_preview unit tests
    # ------------------------------------------------------------------

    from meetingbro.asr.qwen3_asr_adapter import _looks_incomplete_preview  # noqa: E402

    _incomplete_cases: list[tuple[str, bool, str]] = [
        # Should be suppressed (incomplete)
        ("Nein, wer k\u00f6nnte es...", True, "trailing ASCII ellipsis"),
        ("This is incomplete\u2026", True, "trailing Unicode ellipsis \u2026"),
        ("...", True, "only ASCII ellipsis"),
        ("\u3002\u3002\u3002", True, "CJK triple-period"),
        ("Warte mal \u2014", True, "trailing em-dash"),
        ("Something \u2013", True, "trailing en-dash"),
        ("Let me think-", True, "trailing ASCII dash"),
        # Should NOT be suppressed (complete)
        ("\u8fd9\u662f\u4e00\u4e2a\u5b8c\u6574\u53e5\u5b50\u3002", False, "complete Chinese sentence"),
        ("This is a complete sentence.", False, "complete English sentence"),
        ("Guten Morgen, wie geht es Ihnen?", False, "complete German sentence"),
    ]

    incomplete_failures = 0
    for text, expect_suppressed, label in _incomplete_cases:
        got = _looks_incomplete_preview(text)
        status = "OK" if got == expect_suppressed else "FAIL"
        if got != expect_suppressed:
            incomplete_failures += 1
        print(f"  incomplete-preview [{status}] {label}: suppressed={got} (expected {expect_suppressed})")

    if incomplete_failures:
        print(f"FAIL: {incomplete_failures} incomplete-preview case(s) failed")
        return 1
    print("OK: all _looks_incomplete_preview cases passed")

    # Verify that trailing-ellipsis text now PASSES through transcribe() so
    # the preview lane can show growing/partial text in real-time.  The
    # incomplete-preview filter was intentionally moved OUT of transcribe() so
    # Qwen's partial output is visible as it accumulates.
    _a_ell = _make_adapter()
    _a_ell._recognizer = _FakeRecognizer("Nein, wer k\u00f6nnte es...")
    _r_ell = _a_ell.transcribe(_samples_1s, 16_000)
    if not _r_ell:
        print("FAIL: trailing-ellipsis text should now pass through transcribe() for real-time preview")
        return 1
    print("OK: trailing-ellipsis text now passes through transcribe() (partial text shown in preview)")

    _a_complete = _make_adapter()
    _a_complete._recognizer = _FakeRecognizer("This is a complete sentence.")
    _r_complete = _a_complete.transcribe(_samples_1s, 16_000)
    if not _r_complete:
        print("FAIL: complete sentence should NOT be filtered in transcribe()")
        return 1
    print("OK: complete sentence correctly passes through transcribe()")

    print("OK: all _looks_incomplete_preview transcribe() integration tests passed")

    # ------------------------------------------------------------------
    # Stale-preview suppression + alignment diagnostics unit tests
    # (direct calls to SessionManager internals; no audio loop needed)
    # ------------------------------------------------------------------

    import tempfile as _tempfile
    from meetingbro.schemas import TranscriptSegment as _TS

    def _make_segment(
        *,
        start_time: float,
        end_time: float,
        text: str = "dummy",
        confidence: float = 0.92,
        language: str = "en",
    ) -> _TS:
        return _TS(
            id=str(uuid.uuid4()),
            meeting_id="test-meeting",
            start_time=start_time,
            end_time=end_time,
            text=text,
            original_language=language,  # type: ignore[arg-type]
            confidence=confidence,
            created_at=datetime.now(),
        )

    def _make_manager() -> SessionManager:
        with _tempfile.TemporaryDirectory() as _tmp:
            _storage = Storage(Path(_tmp) / "diag.db")
            _m = SessionManager(
                SessionConfig(
                    audio_source=_RealtimeSource(),
                    asr=_StaticASR(text=None),
                    summarizer=_CountingSummarizer(),
                    translator=_NoopTranslator(),
                    storage=_storage,
                    forced_language="en",
                    summary_language="en",
                    asr_accumulation_seconds=60.0,
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
            # Close storage before temp dir is deleted (Windows file-lock).
            _storage.close()
        return _m

    # Test A: _run_alignment_diagnostics with overlapping preview + formal
    _m_align = _make_manager()
    _preview_seg = _make_segment(start_time=0.0, end_time=2.0, text="Hello world")
    _formal_seg = _make_segment(start_time=0.5, end_time=2.5, text="Hello world today")
    _m_align._state.recent_preview_segments.append(_preview_seg)
    _m_align._run_alignment_diagnostics(_formal_seg)
    _ok_align_compared = _m_align._state.preview_alignment_compared == 1
    _ok_align_ratio = (
        _m_align._state.preview_alignment_similarity_last is not None
        and 0.0 < _m_align._state.preview_alignment_similarity_last <= 1.0
    )
    _ok_align_pruned = len(_m_align._state.recent_preview_segments) == 0
    print(
        "alignment-diagnostics:",
        f"compared={_m_align._state.preview_alignment_compared}",
        f"similarity_last={_m_align._state.preview_alignment_similarity_last:.3f}"
        if _m_align._state.preview_alignment_similarity_last is not None else "similarity_last=None",
        f"pruned_ok={_ok_align_pruned}",
    )
    if not _ok_align_compared:
        print("FAIL: alignment compared counter should be 1 after overlapping formal/preview")
        return 1
    if not _ok_align_ratio:
        print(f"FAIL: similarity_last should be (0,1], got {_m_align._state.preview_alignment_similarity_last}")
        return 1
    if not _ok_align_pruned:
        print("FAIL: overlapping preview segment should be pruned from ring buffer after alignment")
        return 1
    print("OK: alignment diagnostics: counter incremented, ratio in range, overlapping segment pruned")

    # Test B: future non-overlapping preview is NOT compared and NOT pruned
    _m_noalign = _make_manager()
    _future_preview = _make_segment(start_time=8.0, end_time=9.0, text="Future preview")
    _later_formal = _make_segment(start_time=5.0, end_time=7.0, text="Later formal")
    _m_noalign._state.recent_preview_segments.append(_future_preview)
    _m_noalign._run_alignment_diagnostics(_later_formal)
    _ok_noalign_zero = _m_noalign._state.preview_alignment_compared == 0
    _ok_noalign_kept = len(_m_noalign._state.recent_preview_segments) == 1
    _ok_noalign_unconfirmed_zero = _m_noalign._state.preview_unconfirmed_after_formal == 0
    print(
        "alignment-no-overlap:",
        f"compared={_m_noalign._state.preview_alignment_compared}",
        f"kept={_ok_noalign_kept}",
        f"unconfirmed={_m_noalign._state.preview_unconfirmed_after_formal}",
    )
    if not _ok_noalign_zero:
        print("FAIL: alignment counter should remain 0 for non-overlapping segments")
        return 1
    if not _ok_noalign_kept:
        print("FAIL: future non-overlapping preview segment should NOT be pruned from ring buffer")
        return 1
    if not _ok_noalign_unconfirmed_zero:
        print("FAIL: future non-overlapping preview should NOT be marked unconfirmed")
        return 1
    print("OK: alignment diagnostics: non-overlapping segment not compared and not pruned")

    # Test C: old non-overlapping preview is marked unconfirmed once formal has passed it
    _m_unconfirmed = _make_manager()
    _orphan_preview = _make_segment(start_time=0.0, end_time=1.0, text="orphan preview")
    _later_formal_2 = _make_segment(start_time=5.0, end_time=7.0, text="later formal")
    _m_unconfirmed._state.recent_preview_segments.append(_orphan_preview)
    _m_unconfirmed._run_alignment_diagnostics(_later_formal_2)
    _ok_unconfirmed_count = _m_unconfirmed._state.preview_unconfirmed_after_formal == 1
    _ok_unconfirmed_text = _m_unconfirmed._state.preview_unconfirmed_last_text == "orphan preview"
    _ok_unconfirmed_pruned = len(_m_unconfirmed._state.recent_preview_segments) == 0
    print(
        "preview-unconfirmed-after-formal:",
        f"count={_m_unconfirmed._state.preview_unconfirmed_after_formal}",
        f"last={_m_unconfirmed._state.preview_unconfirmed_last_text!r}",
        f"pruned={_ok_unconfirmed_pruned}",
    )
    if not _ok_unconfirmed_count:
        print("FAIL: old non-overlapping preview should increment unconfirmed counter")
        return 1
    if not _ok_unconfirmed_text:
        print("FAIL: unconfirmed last text should be recorded for diagnostics")
        return 1
    if not _ok_unconfirmed_pruned:
        print("FAIL: unconfirmed preview should be pruned from ring buffer to avoid double counting")
        return 1
    print("OK: preview unconfirmed diagnostic increments and prunes old non-overlapping preview")

    # Test D: critical transcript_segment makes room by dropping queued non-critical event
    _m_critical = _make_manager()
    _m_critical._event_queue = asyncio.Queue(maxsize=1)
    _m_critical._event_queue.put_nowait(SessionEvent(type="transcript_preview", payload={"segment": None}))
    _critical_seg = _make_segment(start_time=0.0, end_time=1.0, text="critical formal")
    await _m_critical._emit("transcript_segment", _critical_seg)
    _critical_event = _m_critical._event_queue.get_nowait()
    _ok_critical_kept = _critical_event.type == "transcript_segment"
    _ok_critical_drop_count = _m_critical._queue_drop_count == 1
    print(
        "critical-transcript-event:",
        f"event={_critical_event.type}",
        f"drop_count={_m_critical._queue_drop_count}",
    )
    if not _ok_critical_kept:
        print("FAIL: transcript_segment should replace queued non-critical event when queue is full")
        return 1
    if not _ok_critical_drop_count:
        print("FAIL: dropping queued non-critical event should increment drop counter")
        return 1
    print("OK: critical transcript_segment is not dropped when event queue is full")

    # Test E: stale suppression ? preview segment end_time ? last_emitted_end_time + tolerance
    _m_stale = _make_manager()
    _stale_preview = _make_segment(start_time=0.0, end_time=2.0, text="stale preview")
    _m_stale._state.last_emitted_end_time = 2.0  # formal has covered up to 2.0 s
    # end_time=2.0 ≤ 2.0 + 0.30 → should be suppressed; emit with segment=null
    await _m_stale._emit_transcript_preview(_stale_preview)
    _ok_suppressed = _m_stale._state.preview_stale_suppressed == 1
    # The null event must be in the queue (queue is non-empty because we awaited the put).
    try:
        _stale_event = _m_stale._event_queue.get_nowait()
        _ok_null_emitted = (
            _stale_event.type == "transcript_preview"
            and _stale_event.payload.get("segment") is None
        )
    except Exception as _qex:
        _ok_null_emitted = False
        print(f"  (queue read error: {_qex})")
    # Ring buffer must NOT have the stale segment (stale → resolved=None → not appended).
    _ok_stale_not_in_ring = len(_m_stale._state.recent_preview_segments) == 0
    print(
        "stale-suppression:",
        f"suppressed={_m_stale._state.preview_stale_suppressed}",
        f"null_event_emitted={_ok_null_emitted}",
        f"not_in_ring={_ok_stale_not_in_ring}",
    )
    if not _ok_suppressed:
        print("FAIL: stale_suppressed counter should be 1 after suppressing stale preview")
        return 1
    if not _ok_null_emitted:
        print("FAIL: suppressed preview should emit a null-segment transcript_preview event")
        return 1
    if not _ok_stale_not_in_ring:
        print("FAIL: stale segment should NOT be added to ring buffer")
        return 1
    print("OK: stale preview correctly suppressed, null event emitted, counter incremented, not in ring")

    # Test D: non-stale preview — end_time > last_emitted_end_time + tolerance → not suppressed
    _m_nonstale = _make_manager()
    _fresh_preview = _make_segment(start_time=3.0, end_time=5.0, text="fresh preview")
    _m_nonstale._state.last_emitted_end_time = 2.0  # 5.0 > 2.0 + 0.30 → not stale
    await _m_nonstale._emit_transcript_preview(_fresh_preview)
    _ok_not_suppressed = _m_nonstale._state.preview_stale_suppressed == 0
    try:
        _fresh_event = _m_nonstale._event_queue.get_nowait()
        _ok_nonnull = (
            _fresh_event.type == "transcript_preview"
            and _fresh_event.payload.get("segment") is not None
        )
    except Exception as _qex2:
        _ok_nonnull = False
        print(f"  (queue read error: {_qex2})")
    _ok_added_to_ring = len(_m_nonstale._state.recent_preview_segments) == 1
    print(
        "fresh-preview-pass-through:",
        f"suppressed={_m_nonstale._state.preview_stale_suppressed}",
        f"nonnull={_ok_nonnull}",
        f"ring_buffer_len={len(_m_nonstale._state.recent_preview_segments)}",
    )
    if not _ok_not_suppressed:
        print("FAIL: fresh preview should NOT increment stale_suppressed counter")
        return 1
    if not _ok_nonnull:
        print("FAIL: fresh preview should emit non-null segment event")
        return 1
    if not _ok_added_to_ring:
        print("FAIL: fresh preview segment should be added to ring buffer")
        return 1
    print("OK: fresh preview passes through, non-null event emitted, added to ring buffer")

    print("OK: all stale-suppression and alignment diagnostics unit tests passed")

    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
