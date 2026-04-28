"""Benchmark preview-backend contention using the real SessionManager pipeline.

This benchmark replays local WAV input in realtime through:

  WavFileSource(realtime=True) -> SessionManager -> faster-whisper

It compares four benchmark-only configurations without changing product logic:

  preview-off      : fast_preview_enabled=False (formal-ASR-only baseline)
  shared           : fast preview enabled, no dedicated preview_asr
  dedicated-tiny   : preview_asr=tiny, multilingual=True, beam_size=1
  dedicated-base   : preview_asr=base, multilingual=True, beam_size=1
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import glob
import json
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402
from meetingbro.audio.capture import WavFileSource  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.session.profiles import DEFAULT_RUNTIME_PROFILE, runtime_profile_defaults  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402


DEFAULT_WAV = ROOT / "data" / "sample_en.wav"
MEDIAN_FIELDS = [
    "first_preview_latency_seconds",
    "first_formal_latency_seconds",
    "formal_asr_realtime_factor_avg",
    "formal_asr_realtime_factor_max",
    "audio_input_backlog_seconds_avg",
    "audio_input_backlog_seconds_max",
    "audio_input_queue_drop_total",
    "fast_preview_attempts",
    "fast_preview_emitted",
    "fast_preview_skipped",
    "committed_transcript_segment_count",
    "preview_mismatch_rate",
    "pipeline_realtime_factor",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    fast_preview_enabled: bool
    preview_model_size: str | None
    preview_multilingual: bool
    preview_beam_size: int
    env_equivalent: str


SCENARIOS: dict[str, Scenario] = {
    "preview-off": Scenario(
        name="preview-off",
        fast_preview_enabled=False,
        preview_model_size=None,
        preview_multilingual=False,
        preview_beam_size=0,
        env_equivalent="fast_preview_enabled=False",
    ),
    "shared": Scenario(
        name="shared",
        fast_preview_enabled=True,
        preview_model_size=None,
        preview_multilingual=False,
        preview_beam_size=0,
        env_equivalent="MEETINGBRO_PREVIEW_WHISPER_SIZE=shared",
    ),
    "dedicated-tiny": Scenario(
        name="dedicated-tiny",
        fast_preview_enabled=True,
        preview_model_size="tiny",
        preview_multilingual=True,
        preview_beam_size=1,
        env_equivalent=(
            "MEETINGBRO_PREVIEW_WHISPER_SIZE=tiny "
            "MEETINGBRO_PREVIEW_WHISPER_MULTILINGUAL=true "
            "MEETINGBRO_PREVIEW_WHISPER_BEAM_SIZE=1"
        ),
    ),
    "dedicated-base": Scenario(
        name="dedicated-base",
        fast_preview_enabled=True,
        preview_model_size="base",
        preview_multilingual=True,
        preview_beam_size=1,
        env_equivalent=(
            "MEETINGBRO_PREVIEW_WHISPER_SIZE=base "
            "MEETINGBRO_PREVIEW_WHISPER_MULTILINGUAL=true "
            "MEETINGBRO_PREVIEW_WHISPER_BEAM_SIZE=1"
        ),
    ),
}


class NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None, vocabulary=None):
        return ""


class NoopTranslator(Translator):
    def translate(self, text, *, source_language, target_language):
        return text


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in patterns:
        has_glob = any(ch in item for ch in "*?[")
        if has_glob:
            pattern = item if Path(item).is_absolute() else str(ROOT / item)
            matches = [Path(p) for p in glob.glob(pattern)]
        else:
            matches = []
        if not matches:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            matches = [path]
        for path in matches:
            path = path.resolve()
            if path.suffix.lower() == ".wav" and path.exists():
                out.append(path)
    return sorted(dict.fromkeys(out))


def _wav_duration_seconds(path: Path) -> float:
    import soundfile as sf

    with sf.SoundFile(str(path), mode="r") as handle:
        return len(handle) / float(handle.samplerate)


def _normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def _preview_matches_committed(preview: str, committed: str, *, threshold: float) -> bool:
    left = _normalize_text(preview)
    right = _normalize_text(committed)
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    return difflib.SequenceMatcher(a=left, b=right).ratio() >= threshold


def _preview_mismatch_stats(previews: list[str], committed: list[str], *, threshold: float) -> dict[str, Any]:
    if not previews:
        return {
            "preview_events": 0,
            "matched_preview_events": 0,
            "mismatched_preview_events": 0,
            "preview_mismatch_rate": None,
        }
    matched = 0
    for preview in previews:
        if any(_preview_matches_committed(preview, text, threshold=threshold) for text in committed):
            matched += 1
    mismatched = len(previews) - matched
    return {
        "preview_events": len(previews),
        "matched_preview_events": matched,
        "mismatched_preview_events": mismatched,
        "preview_mismatch_rate": mismatched / max(1, len(previews)),
    }


def _format_value(value: float | int | str | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _median(values: list[float | int | None]) -> float | int | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return statistics.median(filtered)


def _build_formal_asr(args: argparse.Namespace) -> FasterWhisperAdapter:
    return FasterWhisperAdapter(
        model_size=args.formal_model_size,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.formal_beam_size,
        cpu_threads=_env_int("MEETINGBRO_WHISPER_CPU_THREADS", 0),
        num_workers=_env_int("MEETINGBRO_WHISPER_NUM_WORKERS", 1),
        vad_threshold=_env_float("MEETINGBRO_WHISPER_VAD_THRESHOLD", 0.3),
        vad_min_speech_ms=_env_int("MEETINGBRO_WHISPER_VAD_MIN_SPEECH_MS", 100),
        vad_min_silence_ms=_env_int("MEETINGBRO_WHISPER_VAD_MIN_SILENCE_MS", 300),
        vad_speech_pad_ms=_env_int("MEETINGBRO_WHISPER_VAD_SPEECH_PAD_MS", 400),
        multilingual=_env_bool("MEETINGBRO_WHISPER_MULTILINGUAL", False),
        language_detection_threshold=_env_float("MEETINGBRO_WHISPER_LANGUAGE_DETECTION_THRESHOLD", 0.5),
        language_detection_segments=_env_int("MEETINGBRO_WHISPER_LANGUAGE_DETECTION_SEGMENTS", 1),
    )


def _build_preview_asr(args: argparse.Namespace, scenario: Scenario) -> FasterWhisperAdapter | None:
    if scenario.preview_model_size is None:
        return None
    return FasterWhisperAdapter(
        model_size=scenario.preview_model_size,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=scenario.preview_beam_size,
        cpu_threads=_env_int("MEETINGBRO_PREVIEW_WHISPER_CPU_THREADS", 0),
        num_workers=_env_int("MEETINGBRO_PREVIEW_WHISPER_NUM_WORKERS", 1),
        vad_threshold=_env_float(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_THRESHOLD",
            _env_float("MEETINGBRO_WHISPER_VAD_THRESHOLD", 0.3),
        ),
        vad_min_speech_ms=_env_int(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_MIN_SPEECH_MS",
            _env_int("MEETINGBRO_WHISPER_VAD_MIN_SPEECH_MS", 100),
        ),
        vad_min_silence_ms=_env_int(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_MIN_SILENCE_MS",
            _env_int("MEETINGBRO_WHISPER_VAD_MIN_SILENCE_MS", 300),
        ),
        vad_speech_pad_ms=_env_int(
            "MEETINGBRO_PREVIEW_WHISPER_VAD_SPEECH_PAD_MS",
            _env_int("MEETINGBRO_WHISPER_VAD_SPEECH_PAD_MS", 400),
        ),
        multilingual=scenario.preview_multilingual,
        language_detection_threshold=_env_float("MEETINGBRO_PREVIEW_WHISPER_LANGUAGE_DETECTION_THRESHOLD", 0.5),
        language_detection_segments=_env_int("MEETINGBRO_PREVIEW_WHISPER_LANGUAGE_DETECTION_SEGMENTS", 1),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark shared vs dedicated preview ASR backends.")
    parser.add_argument("wav", nargs="*", default=[str(DEFAULT_WAV)], help="WAV file(s) or repo-relative glob(s). Default: data/sample_en.wav")
    parser.add_argument("--scenario", choices=("all", *SCENARIOS.keys()), default="all", help="Which preview backend configuration to run. Default: all")
    parser.add_argument("--runtime-profile", default=DEFAULT_RUNTIME_PROFILE)
    parser.add_argument("--language", choices=("auto", "zh", "en", "de"), default="auto")
    parser.add_argument("--formal-model-size", default=os.environ.get("MEETINGBRO_WHISPER_SIZE", "medium"))
    parser.add_argument("--formal-beam-size", type=int, default=_env_int("MEETINGBRO_WHISPER_BEAM_SIZE", 3))
    parser.add_argument("--device", default=os.environ.get("MEETINGBRO_WHISPER_DEVICE", "cpu"))
    parser.add_argument("--compute-type", default=os.environ.get("MEETINGBRO_WHISPER_COMPUTE_TYPE", "int8"))
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--preview-match-threshold", type=float, default=0.75)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--discard-first", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args(argv[1:])


async def _run_one(path: Path, args: argparse.Namespace, scenario: Scenario, *, run_index: int | None, run_type: str) -> dict[str, Any]:
    profile = runtime_profile_defaults(args.runtime_profile)
    forced_language = None if args.language == "auto" else args.language
    audio_duration = _wav_duration_seconds(path)
    chunk_seconds = float(profile.get("chunk_seconds", 0.5))

    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / f"benchmark-preview-{scenario.name}-{run_type}-{run_index or 0}.db")
        try:
            formal_asr = _build_formal_asr(args)
            preview_asr = _build_preview_asr(args, scenario)
            manager = SessionManager(
                SessionConfig(
                    audio_source=WavFileSource(path, sample_rate=16_000, chunk_seconds=chunk_seconds, realtime=True),
                    audio_chunk_seconds=chunk_seconds,
                    runtime_profile=args.runtime_profile,
                    asr=formal_asr,
                    preview_asr=preview_asr,
                    summarizer=NoopSummarizer(),
                    translator=NoopTranslator(),
                    storage=storage,
                    forced_language=forced_language,
                    summary_language="en",
                    rolling_interval_seconds=999999.0,
                    memory_interval_seconds=999999.0,
                    cumulative_interval_seconds=999999.0,
                    min_segments_for_rolling=999999,
                    min_segments_for_memory=999999,
                    min_segments_for_cumulative=999999,
                    asr_accumulation_seconds=float(profile.get("asr_accumulation_seconds", 1.5)),
                    asr_early_flush_enabled=bool(profile.get("asr_early_flush_enabled", True)),
                    asr_early_flush_min_seconds=float(profile.get("asr_early_flush_min_seconds", 0.8)),
                    silence_commit_min_confidence=float(profile.get("silence_commit_min_confidence", 0.75)),
                    silence_commit_min_duration_seconds=float(profile.get("silence_commit_min_duration_seconds", 0.6)),
                    pre_vad_enabled=bool(profile.get("pre_vad_enabled", True)),
                    pre_vad_trailing_silence_seconds=float(profile.get("pre_vad_trailing_silence_seconds", 0.45)),
                    pre_vad_adaptive_trailing_silence_enabled=bool(profile.get("pre_vad_adaptive_trailing_silence_enabled", True)),
                    pre_vad_adaptive_fast_trailing_silence_seconds=float(profile.get("pre_vad_adaptive_fast_trailing_silence_seconds", 0.30)),
                    language_lock_enabled=bool(profile.get("language_lock_enabled", False)),
                    weak_speech_rescue_enabled=bool(profile.get("weak_speech_rescue_enabled", True)),
                    weak_speech_rescue_fast_window_seconds=float(profile.get("weak_speech_rescue_fast_window_seconds", 2.5)),
                    silence_rms_threshold=_env_float("MEETINGBRO_SILENCE_RMS_THRESHOLD", 0.002),
                    asr_overlap_seconds=_env_float("MEETINGBRO_ASR_OVERLAP_SECONDS", 0.0),
                    asr_retry_enabled=_env_bool("MEETINGBRO_ASR_RETRY_ENABLED", True),
                    asr_safeguard_enabled=_env_bool("MEETINGBRO_ASR_SAFEGUARD_ENABLED", True),
                    asr_safeguard_rtf_threshold=_env_float("MEETINGBRO_ASR_SAFEGUARD_RTF_THRESHOLD", 0.9),
                    asr_safeguard_cooldown_windows=_env_int("MEETINGBRO_ASR_SAFEGUARD_COOLDOWN_WINDOWS", 5),
                    denoise_enabled=_env_bool("MEETINGBRO_DENOISE_ENABLED", False),
                    denoise_strength=_env_float("MEETINGBRO_DENOISE_STRENGTH", 1.1),
                    denoise_noise_update_rms_threshold=_env_float("MEETINGBRO_DENOISE_NOISE_UPDATE_RMS_THRESHOLD", 0.02),
                    audio_conditioning_enabled=_env_bool("MEETINGBRO_AUDIO_CONDITIONING_ENABLED", True),
                    audio_conditioning_target_rms=_env_float("MEETINGBRO_AUDIO_CONDITIONING_TARGET_RMS", 0.035),
                    audio_conditioning_min_rms=_env_float("MEETINGBRO_AUDIO_CONDITIONING_MIN_RMS", 0.003),
                    audio_conditioning_max_gain=_env_float("MEETINGBRO_AUDIO_CONDITIONING_MAX_GAIN", 2.5),
                    audio_conditioning_peak_limit=_env_float("MEETINGBRO_AUDIO_CONDITIONING_PEAK_LIMIT", 0.98),
                    pre_vad_conditioning_enabled=_env_bool("MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED", True),
                    pre_vad_conditioning_target_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_TARGET_RMS", 0.03),
                    pre_vad_conditioning_min_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MIN_RMS", 0.001),
                    pre_vad_conditioning_max_gain=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MAX_GAIN", 4.0),
                    pre_vad_threshold=_env_float("MEETINGBRO_PRE_VAD_THRESHOLD", 0.38),
                    pre_vad_energy_rms_threshold=_env_float("MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD", 0.005),
                    pre_vad_adaptive_max_realtime_factor=_env_float("MEETINGBRO_PRE_VAD_ADAPTIVE_MAX_REALTIME_FACTOR", 0.5),
                    pre_vad_max_segment_seconds=_env_float("MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS", 8.0),
                    weak_speech_rescue_rms_min=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN", 0.0008),
                    weak_speech_rescue_rms_max=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX", 0.02),
                    weak_speech_rescue_fast_rms_max=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_FAST_RMS_MAX", 0.01),
                    weak_speech_rescue_window_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_WINDOW_SECONDS", 6.0),
                    weak_speech_rescue_cooldown_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_COOLDOWN_SECONDS", 8.0),
                    live_translation_backfill_limit=_env_int("MEETINGBRO_LIVE_TRANSLATION_BACKFILL_LIMIT", 20),
                    live_translation_max_pending=_env_int("MEETINGBRO_LIVE_TRANSLATION_MAX_PENDING", 12),
                    live_translation_safeguard_max_pending=_env_int("MEETINGBRO_LIVE_TRANSLATION_SAFEGUARD_MAX_PENDING", 4),
                    audio_input_queue_max_seconds=_env_float("MEETINGBRO_AUDIO_INPUT_QUEUE_MAX_SECONDS", 8.0),
                    audio_input_queue_warning_seconds=_env_float("MEETINGBRO_AUDIO_INPUT_QUEUE_WARNING_SECONDS", 3.0),
                    fast_preview_enabled=scenario.fast_preview_enabled,
                    fast_preview_interval_seconds=_env_float("MEETINGBRO_FAST_PREVIEW_INTERVAL_SECONDS", 0.8),
                    fast_preview_window_seconds=_env_float("MEETINGBRO_FAST_PREVIEW_WINDOW_SECONDS", 3.0),
                    fast_preview_max_backlog_seconds=_env_float("MEETINGBRO_FAST_PREVIEW_MAX_BACKLOG_SECONDS", 0.5),
                    fast_preview_max_asr_realtime_factor=_env_float("MEETINGBRO_FAST_PREVIEW_MAX_ASR_RTF", 0.65),
                    fast_preview_min_rms=_env_float("MEETINGBRO_FAST_PREVIEW_MIN_RMS", 0.002),
                    asr_executor_workers=_env_int("MEETINGBRO_ASR_EXECUTOR_WORKERS", 1),
                    preview_asr_executor_workers=_env_int("MEETINGBRO_PREVIEW_ASR_EXECUTOR_WORKERS", 1),
                    summary_executor_workers=_env_int("MEETINGBRO_SUMMARY_EXECUTOR_WORKERS", 1),
                    translation_executor_workers=_env_int("MEETINGBRO_TRANSLATION_EXECUTOR_WORKERS", 2),
                )
            )

            preview_texts: list[str] = []
            first_preview_at: float | None = None
            first_commit_at: float | None = None
            backlog_samples: list[float] = []
            formal_rtf_samples: list[float] = []
            preview_rtf_samples: list[float] = []
            stop_poll = asyncio.Event()

            async def collect_events(start_wall: float) -> None:
                nonlocal first_preview_at, first_commit_at
                async for event in manager.events():
                    now = time.monotonic()
                    if scenario.fast_preview_enabled and event.type == "transcript_preview":
                        segment = event.payload.get("segment")
                        if segment is not None:
                            preview_texts.append(str(segment.get("text") or ""))
                            if first_preview_at is None:
                                first_preview_at = now - start_wall
                    elif event.type == "transcript_segment" and first_commit_at is None:
                        first_commit_at = now - start_wall

            async def poll_state() -> None:
                while not stop_poll.is_set():
                    backlog_samples.append(manager._state.audio_input_backlog_seconds)
                    if manager._state.asr_realtime_factor is not None:
                        formal_rtf_samples.append(manager._state.asr_realtime_factor)
                    if manager._state.fast_preview_realtime_factor is not None:
                        preview_rtf_samples.append(manager._state.fast_preview_realtime_factor)
                    try:
                        await asyncio.wait_for(stop_poll.wait(), timeout=args.poll_interval)
                    except asyncio.TimeoutError:
                        continue

            start_wall = time.monotonic()
            collector = asyncio.create_task(collect_events(start_wall))
            poller = asyncio.create_task(poll_state())
            await manager.start()
            try:
                if manager._task is not None:
                    await manager._task
            finally:
                await manager.stop()
                stop_poll.set()
                try:
                    await asyncio.wait_for(poller, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    poller.cancel()
                try:
                    await asyncio.wait_for(collector, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    collector.cancel()
            wall_seconds = time.monotonic() - start_wall

            segments = storage.list_segments(manager.meeting_id)
            mismatch = _preview_mismatch_stats(preview_texts, [segment.text for segment in segments], threshold=args.preview_match_threshold)
            formal_rtf_avg = statistics.fmean(formal_rtf_samples) if formal_rtf_samples else None
            formal_rtf_max = max(formal_rtf_samples) if formal_rtf_samples else None
            backlog_avg = statistics.fmean(backlog_samples) if backlog_samples else 0.0
            backlog_max = max(backlog_samples) if backlog_samples else 0.0

            return {
                "row_type": run_type,
                "scenario": scenario.name,
                "file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "run_index": run_index,
                "included_in_aggregate": False,
                "env_equivalent": scenario.env_equivalent,
                "runtime_profile": args.runtime_profile,
                "audio_seconds": audio_duration,
                "wall_seconds": wall_seconds,
                "pipeline_realtime_factor": wall_seconds / max(audio_duration, 1e-6),
                "preview_backend": "off" if not scenario.fast_preview_enabled else ("shared" if scenario.preview_model_size is None else "dedicated"),
                "preview_model_size": "off" if not scenario.fast_preview_enabled else (scenario.preview_model_size or "shared"),
                "preview_multilingual": scenario.preview_multilingual if scenario.preview_model_size else None,
                "preview_beam_size": scenario.preview_beam_size if scenario.preview_model_size else None,
                "first_preview_latency_seconds": first_preview_at,
                "first_formal_latency_seconds": first_commit_at,
                "formal_asr_realtime_factor_avg": formal_rtf_avg,
                "formal_asr_realtime_factor_max": formal_rtf_max,
                "formal_asr_realtime_factor_last": manager._state.asr_realtime_factor,
                "audio_input_backlog_seconds_avg": backlog_avg,
                "audio_input_backlog_seconds_max": backlog_max,
                "audio_input_queue_drop_total": manager._state.audio_input_queue_drop_total,
                "fast_preview_attempts": manager._state.fast_preview_attempts,
                "fast_preview_emitted": manager._state.fast_preview_emitted,
                "fast_preview_skipped": manager._state.fast_preview_skipped,
                "committed_transcript_segment_count": len(segments),
                "preview_events": mismatch["preview_events"],
                "matched_preview_events": mismatch["matched_preview_events"],
                "mismatched_preview_events": mismatch["mismatched_preview_events"],
                "preview_mismatch_rate": mismatch["preview_mismatch_rate"],
                "fast_preview_realtime_factor_avg": statistics.fmean(preview_rtf_samples) if preview_rtf_samples else None,
            }
        finally:
            storage.close()


def _aggregate_rows(rows: list[dict[str, Any]], *, discard_first: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["row_type"] != "measurement":
            continue
        grouped.setdefault((str(row["scenario"]), str(row["file"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (scenario, file_name), group in sorted(grouped.items()):
        sorted_group = sorted(group, key=lambda item: int(item["run_index"] or 0))
        used_group = sorted_group[1:] if discard_first and len(sorted_group) > 1 else sorted_group
        base = dict(used_group[0] if used_group else sorted_group[0])
        base["row_type"] = "aggregate"
        base["scenario"] = scenario
        base["file"] = file_name
        base["run_index"] = None
        base["included_in_aggregate"] = None
        base["aggregate_source_run_indices"] = [row["run_index"] for row in used_group]
        base["aggregate_measurement_count"] = len(used_group)
        for field in MEDIAN_FIELDS:
            base[field] = _median([row.get(field) for row in used_group]) if used_group else None
        aggregate_rows.append(base)
    return aggregate_rows


def _print_table(rows: list[dict[str, Any]], *, title: str) -> None:
    headers = [
        "row_type", "scenario", "file", "run", "audio_s", "preview_model", "preview_ms", "formal_ms",
        "formal_rtf_avg", "formal_rtf_max", "backlog_avg_s", "backlog_max_s", "queue_drops",
        "preview_attempts", "preview_emitted", "preview_skipped", "commits", "preview_mismatch_rate", "pipeline_rtf",
    ]
    print(title)
    print("\t".join(headers))
    for row in rows:
        print("\t".join([
            str(row["row_type"]),
            str(row["scenario"]),
            str(row["file"]),
            _format_value(row["run_index"]),
            _format_value(row["audio_seconds"], 1),
            str(row["preview_model_size"]),
            _format_value(row["first_preview_latency_seconds"], 3),
            _format_value(row["first_formal_latency_seconds"], 3),
            _format_value(row["formal_asr_realtime_factor_avg"], 3),
            _format_value(row["formal_asr_realtime_factor_max"], 3),
            _format_value(row["audio_input_backlog_seconds_avg"], 3),
            _format_value(row["audio_input_backlog_seconds_max"], 3),
            _format_value(row["audio_input_queue_drop_total"]),
            _format_value(row["fast_preview_attempts"]),
            _format_value(row["fast_preview_emitted"]),
            _format_value(row["fast_preview_skipped"]),
            _format_value(row["committed_transcript_segment_count"]),
            _format_value(row["preview_mismatch_rate"], 3),
            _format_value(row["pipeline_realtime_factor"], 3),
        ]))


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2
    wavs = _resolve_inputs(args.wav)
    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        return 2

    scenarios = list(SCENARIOS.values()) if args.scenario == "all" else [SCENARIOS[args.scenario]]
    rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        for wav in wavs:
            if args.warmup:
                rows.append(await _run_one(wav, args, scenario, run_index=0, run_type="warmup"))
            for run_index in range(1, args.repeat + 1):
                row = await _run_one(wav, args, scenario, run_index=run_index, run_type="measurement")
                row["included_in_aggregate"] = not (args.discard_first and args.repeat > 1 and run_index == 1)
                rows.append(row)
                measurement_rows.append(row)

    aggregate_rows = _aggregate_rows(measurement_rows, discard_first=args.discard_first)
    rows.extend(aggregate_rows)

    _print_table(measurement_rows, title="Per-run measurement rows")
    print()
    _print_table(aggregate_rows, title="Aggregate median rows")

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))