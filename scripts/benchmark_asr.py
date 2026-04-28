"""Benchmark MeetingBro's local ASR path on WAV files.

This is intentionally LLM-free: it runs WavFileSource -> SessionManager ->
FasterWhisperAdapter with no-op summarization/translation so the metrics mostly
reflect audio preprocessing, VAD, ASR speed, weak-speech rescue, and safeguards.

Examples:
  python scripts/benchmark_asr.py data/sample_en.wav --language en
  python scripts/benchmark_asr.py data/*.wav --language auto --model-size small
  python scripts/benchmark_asr.py data/sample_en.wav --keywords Alice Bob budget
  python scripts/benchmark_asr.py data/sample_en.wav --json-out benchmark.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from meetingbro.asr.faster_whisper_adapter import FasterWhisperAdapter  # noqa: E402
from meetingbro.audio.capture import WavFileSource  # noqa: E402
from meetingbro.llm.openai_compatible import _load_dotenv_if_present  # noqa: E402
from meetingbro.session.manager import SessionConfig, SessionManager  # noqa: E402
from meetingbro.storage.db import Storage  # noqa: E402
from meetingbro.summarization.base import Summarizer  # noqa: E402
from meetingbro.translation.base import Translator  # noqa: E402


class NoopSummarizer(Summarizer):
    def summarize(self, segments, *, kind, language, previous_summary=None):
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

    with sf.SoundFile(str(path), mode="r") as f:
        return len(f) / float(f.samplerate)


def _keyword_recall(text: str, keywords: list[str]) -> dict[str, Any]:
    if not keywords:
        return {"expected": 0, "matched": 0, "recall": None, "missing": []}
    folded = text.casefold()
    missing = [kw for kw in keywords if kw.casefold() not in folded]
    matched = len(keywords) - len(missing)
    return {
        "expected": len(keywords),
        "matched": matched,
        "recall": matched / max(1, len(keywords)),
        "missing": missing,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MeetingBro ASR on WAV files.")
    parser.add_argument("wav", nargs="+", help="WAV file(s) or repo-relative glob(s).")
    parser.add_argument("--language", choices=("auto", "zh", "en", "de"), default="auto")
    parser.add_argument("--model-size", default=os.environ.get("MEETINGBRO_WHISPER_SIZE", "tiny"))
    parser.add_argument("--device", default=os.environ.get("MEETINGBRO_WHISPER_DEVICE", "cpu"))
    parser.add_argument("--compute-type", default=os.environ.get("MEETINGBRO_WHISPER_COMPUTE_TYPE", "int8"))
    parser.add_argument("--beam-size", type=int, default=_env_int("MEETINGBRO_WHISPER_BEAM_SIZE", 3))
    parser.add_argument("--chunk-seconds", type=float, default=_env_float("MEETINGBRO_FILE_CHUNK_SECONDS", 5.0))
    parser.add_argument("--keywords", nargs="*", default=[], help="Expected keywords for rough recall.")
    parser.add_argument("--json-out", default=None, help="Optional JSON report path.")
    parser.add_argument("--text-out-dir", default=None, help="Optional directory for transcript .txt files.")
    return parser.parse_args(argv[1:])


async def _run_one(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    forced_language = None if args.language == "auto" else args.language
    audio_duration = _wav_duration_seconds(path)
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp) / "benchmark.db")
        try:
            asr = FasterWhisperAdapter(
                model_size=args.model_size,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                cpu_threads=_env_int("MEETINGBRO_WHISPER_CPU_THREADS", 0),
                num_workers=_env_int("MEETINGBRO_WHISPER_NUM_WORKERS", 1),
                vad_threshold=_env_float("MEETINGBRO_WHISPER_VAD_THRESHOLD", 0.3),
                vad_min_speech_ms=_env_int("MEETINGBRO_WHISPER_VAD_MIN_SPEECH_MS", 100),
                vad_min_silence_ms=_env_int("MEETINGBRO_WHISPER_VAD_MIN_SILENCE_MS", 300),
                vad_speech_pad_ms=_env_int("MEETINGBRO_WHISPER_VAD_SPEECH_PAD_MS", 400),
                multilingual=_env_bool("MEETINGBRO_WHISPER_MULTILINGUAL", False),
            )
            cfg = SessionConfig(
                audio_source=WavFileSource(path, sample_rate=16_000, chunk_seconds=args.chunk_seconds, realtime=False),
                asr=asr,
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
                asr_accumulation_seconds=_env_float("MEETINGBRO_ASR_ACCUM_SECONDS", 2.5),
                silence_rms_threshold=_env_float("MEETINGBRO_SILENCE_RMS_THRESHOLD", 0.001),
                asr_overlap_seconds=_env_float("MEETINGBRO_ASR_OVERLAP_SECONDS", 0.0),
                vocabulary_hint=os.environ.get("MEETINGBRO_VOCABULARY_HINT") or None,
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
                pre_vad_enabled=_env_bool("MEETINGBRO_PRE_VAD_ENABLED", True),
                pre_vad_conditioning_enabled=_env_bool("MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED", True),
                pre_vad_conditioning_target_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_TARGET_RMS", 0.03),
                pre_vad_conditioning_min_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MIN_RMS", 0.001),
                pre_vad_conditioning_max_gain=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MAX_GAIN", 4.0),
                pre_vad_threshold=_env_float("MEETINGBRO_PRE_VAD_THRESHOLD", 0.38),
                pre_vad_energy_rms_threshold=_env_float("MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD", 0.005),
                pre_vad_trailing_silence_seconds=_env_float("MEETINGBRO_PRE_VAD_TRAILING_SILENCE_SECONDS", 0.45),
                pre_vad_max_segment_seconds=_env_float("MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS", 8.0),
                weak_speech_rescue_enabled=_env_bool("MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED", True),
                weak_speech_rescue_rms_min=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN", 0.0008),
                weak_speech_rescue_rms_max=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX", 0.02),
                weak_speech_rescue_window_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_WINDOW_SECONDS", 6.0),
                weak_speech_rescue_cooldown_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_COOLDOWN_SECONDS", 8.0),
                language_lock_enabled=_env_bool("MEETINGBRO_LANGUAGE_LOCK_ENABLED", False),
            )
            manager = SessionManager(cfg)
            t0 = time.perf_counter()
            await manager.start()
            if manager._task is not None:
                await manager._task
            await manager.stop()
            wall_seconds = time.perf_counter() - t0

            segments = storage.list_segments(manager.meeting_id)
            transcript = "\n".join(seg.text for seg in segments)
            language_counts = Counter(seg.original_language for seg in segments)
            keyword = _keyword_recall(transcript, args.keywords)
            if args.text_out_dir:
                out_dir = Path(args.text_out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{path.stem}.txt").write_text(transcript, encoding="utf-8")

            return {
                "file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "audio_seconds": audio_duration,
                "wall_seconds": wall_seconds,
                "pipeline_realtime_factor": wall_seconds / max(audio_duration, 1e-6),
                "segments": len(segments),
                "transcript_chars": len(transcript),
                "languages": dict(language_counts),
                "asr_last_realtime_factor": manager._state.asr_realtime_factor,
                "asr_safeguard_events": manager._state.asr_safeguard_events,
                "weak_rescue_attempts": manager._state.weak_rescue_attempts,
                "weak_rescue_emitted": manager._state.weak_rescue_emitted,
                "retry_windows_total": manager._state.retry_windows_total,
                "retry_windows_improved": manager._state.retry_windows_improved,
                "retry_windows_diverged": manager._state.retry_windows_diverged,
                "audio_drop_total": manager._audio_drop_total,
                "keyword": keyword,
                "first_segments": [
                    {
                        "start": round(seg.start_time, 2),
                        "end": round(seg.end_time, 2),
                        "language": seg.original_language,
                        "text": seg.text,
                    }
                    for seg in segments[:5]
                ],
            }
        finally:
            storage.close()


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "file",
        "audio",
        "wall",
        "rtf",
        "seg",
        "chars",
        "asr_rtf",
        "safe",
        "rescue",
        "kw",
    ]
    print("\t".join(headers))
    for row in rows:
        kw = row["keyword"]
        kw_text = "—" if kw["recall"] is None else f"{kw['matched']}/{kw['expected']}"
        print(
            "\t".join(
                [
                    row["file"],
                    f"{row['audio_seconds']:.1f}",
                    f"{row['wall_seconds']:.1f}",
                    f"{row['pipeline_realtime_factor']:.2f}",
                    str(row["segments"]),
                    str(row["transcript_chars"]),
                    "—" if row["asr_last_realtime_factor"] is None else f"{row['asr_last_realtime_factor']:.2f}",
                    str(row["asr_safeguard_events"]),
                    f"{row['weak_rescue_emitted']}/{row['weak_rescue_attempts']}",
                    kw_text,
                ]
            )
        )


async def main(argv: list[str]) -> int:
    _load_dotenv_if_present()
    args = _parse_args(argv)
    wavs = _resolve_inputs(args.wav)
    if not wavs:
        print("No WAV files found.", file=sys.stderr)
        return 2

    rows = []
    for wav in wavs:
        rows.append(await _run_one(wav, args))

    _print_table(rows)
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
