"""Benchmark sherpa-onnx Qwen3-ASR as an isolated near-streaming preview candidate.

This script does not touch MeetingBro session logic. It evaluates a local
Qwen3-ASR ONNX bundle by chunking one or more WAV files, using a lightweight
RMS-based endpointing loop, and periodically re-decoding the current utterance.

Important: this is a near-streaming PoC, not true streaming ASR. The Qwen3-ASR
path exposed by sherpa-onnx today is an offline recognizer, so partial updates
here are simulated by re-decoding the active utterance buffer.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
DEFAULT_INPUT_GLOB = "data/*.wav"
MODE_LABEL = "near-streaming (chunked replay + RMS endpointing + offline re-decode)"
MODE_WARNING_LINES = [
    "This benchmark is near-streaming, not true token-level streaming.",
    "Partial updates are simulated via chunked replay plus offline re-decode of the active utterance.",
    "language_hint is currently recorded for bookkeeping but not applied to sherpa-onnx Qwen3-ASR decoding.",
]
SUPPORTED_LANGUAGE_HINTS = {"auto", "en", "de", "zh"}


@dataclass
class UpdateEvent:
    kind: str
    audio_time_seconds: float
    wall_time_seconds: float
    utterance_index: int
    text: str


@dataclass
class FileBenchmarkResult:
    path: str
    mode: str
    audio_duration_seconds: float
    recognizer_create_seconds: float
    wall_time_seconds: float
    realtime_factor: float | None
    first_partial_audio_latency_seconds: float | None
    first_partial_wall_latency_seconds: float | None
    first_final_audio_latency_seconds: float | None
    first_final_wall_latency_seconds: float | None
    update_count: int
    final_segment_count: int
    language_hint: str
    language_hint_applied: bool
    hotwords: str
    baseline_rss_mb: float | None
    peak_rss_mb: float | None
    rss_delta_mb: float | None
    final_text: str
    updates: list[dict[str, Any]]


@dataclass(frozen=True)
class ModelPaths:
    model_dir: Path
    conv_frontend: Path
    encoder: Path
    decoder: Path
    tokenizer: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Benchmark sherpa-onnx Qwen3-ASR 0.6B int8 as an isolated "
            "near-streaming preview candidate."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "WAV files or glob patterns. If omitted, defaults to data/*.wav and "
            "uses any matches that exist."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing conv_frontend.onnx, encoder.int8.onnx, decoder.int8.onnx, and tokenizer/.",
    )
    parser.add_argument(
        "--language",
        choices=sorted(SUPPORTED_LANGUAGE_HINTS),
        default="auto",
        help=(
            "Requested language hint. The current sherpa-onnx offline Qwen3-ASR "
            "factory does not expose a language argument, so this is recorded for "
            "benchmark bookkeeping but not applied to decoding."
        ),
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=200.0,
        help="Chunk size used for simulated streaming replay.",
    )
    parser.add_argument(
        "--partial-interval-ms",
        type=float,
        default=800.0,
        help="Minimum interval between tentative re-decodes while inside speech.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=float,
        default=350.0,
        help="Minimum detected speech before tentative updates are allowed.",
    )
    parser.add_argument(
        "--endpoint-silence-ms",
        type=float,
        default=550.0,
        help="Trailing silence required before finalizing an utterance.",
    )
    parser.add_argument(
        "--max-active-audio-seconds",
        type=float,
        default=6.0,
        help="Force a final decode and reset if the active utterance exceeds this duration.",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=0.003,
        help="RMS threshold used by the simple speech detector.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of ONNX threads used by sherpa-onnx.",
    )
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--max-total-len",
        type=int,
        default=512,
        help="Qwen3-ASR KV-cache maximum total sequence length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per decode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1e-6,
        help="Sampling temperature passed to sherpa-onnx.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p threshold passed to sherpa-onnx.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to sherpa-onnx.",
    )
    parser.add_argument(
        "--hotwords",
        type=str,
        default="",
        help="Optional comma-separated hotwords forwarded to sherpa-onnx.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write structured benchmark output as JSON.",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Validate model files under --model-dir and exit without running transcription.",
    )
    return parser.parse_args()


def _resolve_inputs(patterns: list[str]) -> list[Path]:
    items = patterns or [DEFAULT_INPUT_GLOB]
    resolved: list[Path] = []
    for item in items:
        has_glob = any(ch in item for ch in "*?[")
        matches: list[Path] = []
        if has_glob:
            pattern = item if Path(item).is_absolute() else str(ROOT / item)
            matches = [Path(path) for path in glob.glob(pattern)]
        if not matches:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            matches = [path]
        for path in matches:
            path = path.resolve()
            if path.is_file() and path.suffix.lower() == ".wav":
                resolved.append(path)
    return sorted(dict.fromkeys(resolved))


def _current_rss_mb() -> float | None:
    if os.name == "nt":
        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.WorkingSetSize / (1024.0 * 1024.0)
        return None

    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _load_audio(path: Path) -> tuple[np.ndarray, int, float]:
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        channels = samples.shape[1]
        if channels != 1:
            raise ValueError(
                f"{path} must be mono for this benchmark, but has {channels} channels. "
                "Convert it to 16 kHz mono WAV before running the Qwen3-ASR benchmark."
            )
        samples = samples[:, 0]
    if samples.ndim != 1:
        raise ValueError(f"Unsupported audio shape for {path}: {samples.shape}")
    if int(sample_rate) != 16000:
        raise ValueError(
            f"{path} has sample rate {sample_rate} Hz. This benchmark requires explicit 16 kHz mono WAV input. "
            "Convert the file to 16 kHz mono before running so we do not silently rely on implicit resampling."
        )
    duration_seconds = 0.0 if sample_rate <= 0 else samples.shape[0] / float(sample_rate)
    return samples, int(sample_rate), duration_seconds


def _speech_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return math.sqrt(float(np.mean(np.square(samples, dtype=np.float32))))


def _extract_text_and_payload(result: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(result, dict):
        text = str(result.get("text", "")).strip()
        return text, result

    if hasattr(result, "text"):
        payload = {
            "text": str(getattr(result, "text", "")).strip(),
            "lang": str(getattr(result, "lang", "")),
            "emotion": str(getattr(result, "emotion", "")),
            "event": str(getattr(result, "event", "")),
        }
        for key in ("timestamps", "durations", "tokens", "words"):
            if hasattr(result, key):
                payload[key] = getattr(result, key)
        return payload["text"], payload

    raw = result if isinstance(result, str) else str(result)
    raw = raw.strip()
    if not raw:
        return "", {"text": ""}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {"text": raw}
    text = str(payload.get("text", "")).strip()
    return text, payload


def _import_sherpa_onnx() -> Any:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "Missing sherpa-onnx. Install it in the MeetingBro backend environment with: "
            "C:/Users/RL_Carla/anaconda3/envs/MeetingBro/python.exe -m pip install sherpa-onnx"
        ) from exc
    return sherpa_onnx


def _resolve_model_paths(model_dir: Path) -> ModelPaths:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}. "
            "Download or point --model-dir at a sherpa-onnx Qwen3-ASR export first."
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {model_dir}")

    encoder_candidates = [model_dir / "encoder.int8.onnx", model_dir / "encoder.onnx"]
    decoder_candidates = [model_dir / "decoder.int8.onnx", model_dir / "decoder.onnx"]

    def pick(candidates: list[Path], label: str) -> Path:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find {label} under {model_dir}. Tried: "
            + ", ".join(str(path.name) for path in candidates)
        )

    conv_frontend = model_dir / "conv_frontend.onnx"
    if not conv_frontend.is_file():
        raise FileNotFoundError(f"Missing conv_frontend.onnx under {model_dir}")

    tokenizer_dir = model_dir / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory under {model_dir}")

    return ModelPaths(
        model_dir=model_dir,
        conv_frontend=conv_frontend,
        encoder=pick(encoder_candidates, "encoder model"),
        decoder=pick(decoder_candidates, "decoder model"),
        tokenizer=tokenizer_dir,
    )


def _print_model_check(model_paths: ModelPaths) -> None:
    print("Model check passed:")
    print(f"  model_dir: {model_paths.model_dir}")
    print(f"  conv_frontend: {model_paths.conv_frontend}")
    print(f"  encoder: {model_paths.encoder}")
    print(f"  decoder: {model_paths.decoder}")
    print(f"  tokenizer: {model_paths.tokenizer}")


def _create_recognizer(args: argparse.Namespace) -> tuple[Any, float]:
    sherpa_onnx = _import_sherpa_onnx()
    model_paths = _resolve_model_paths(args.model_dir)

    started = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=str(model_paths.conv_frontend),
        encoder=str(model_paths.encoder),
        decoder=str(model_paths.decoder),
        tokenizer=str(model_paths.tokenizer),
        num_threads=args.num_threads,
        sample_rate=16000,
        feature_dim=128,
        decoding_method="greedy_search",
        debug=False,
        provider=args.provider,
        max_total_len=args.max_total_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        hotwords=args.hotwords,
    )
    return recognizer, time.perf_counter() - started


def _decode_samples(recognizer: Any, samples: np.ndarray, sample_rate: int) -> tuple[str, dict[str, Any]]:
    if samples.size == 0:
        return "", {"text": ""}
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return _extract_text_and_payload(stream.result)


def _maybe_record_update(
    updates: list[UpdateEvent],
    *,
    kind: str,
    utterance_index: int,
    text: str,
    audio_time_seconds: float,
    benchmark_started: float,
    last_emitted_text: str | None,
) -> str | None:
    text = text.strip()
    if not text or text == (last_emitted_text or ""):
        return last_emitted_text
    updates.append(
        UpdateEvent(
            kind=kind,
            utterance_index=utterance_index,
            text=text,
            audio_time_seconds=audio_time_seconds,
            wall_time_seconds=time.perf_counter() - benchmark_started,
        )
    )
    return text


def _benchmark_file(
    recognizer: Any,
    recognizer_create_seconds: float,
    path: Path,
    args: argparse.Namespace,
) -> FileBenchmarkResult:
    samples, sample_rate, duration_seconds = _load_audio(path)
    baseline_rss_mb = _current_rss_mb()
    peak_rss_mb = baseline_rss_mb

    chunk_size = max(1, int(round(sample_rate * (args.chunk_ms / 1000.0))))
    partial_interval_seconds = max(0.0, args.partial_interval_ms / 1000.0)
    min_speech_seconds = max(0.0, args.min_speech_ms / 1000.0)
    endpoint_silence_seconds = max(0.0, args.endpoint_silence_ms / 1000.0)
    max_active_audio_seconds = max(0.0, args.max_active_audio_seconds)

    benchmark_started = time.perf_counter()
    updates: list[UpdateEvent] = []
    final_texts: list[str] = []
    utterance_parts: list[np.ndarray] = []
    utterance_index = 0
    in_speech = False
    speech_seconds = 0.0
    silence_seconds = 0.0
    last_partial_decode_audio_time = -1e9
    last_emitted_text: str | None = None

    def finalize_utterance(audio_time_seconds: float) -> None:
        nonlocal utterance_parts
        nonlocal utterance_index
        nonlocal in_speech
        nonlocal speech_seconds
        nonlocal silence_seconds
        nonlocal last_partial_decode_audio_time
        nonlocal last_emitted_text

        if not utterance_parts:
            in_speech = False
            speech_seconds = 0.0
            silence_seconds = 0.0
            last_partial_decode_audio_time = -1e9
            last_emitted_text = None
            return

        utterance_audio = np.concatenate(utterance_parts)
        text, _payload = _decode_samples(recognizer, utterance_audio, sample_rate)
        last_emitted_text = _maybe_record_update(
            updates,
            kind="final",
            utterance_index=utterance_index,
            text=text,
            audio_time_seconds=audio_time_seconds,
            benchmark_started=benchmark_started,
            last_emitted_text=last_emitted_text,
        )
        if text.strip():
            final_texts.append(text.strip())
            utterance_index += 1

        current_rss_mb = _current_rss_mb()
        if current_rss_mb is not None:
            nonlocal peak_rss_mb
            peak_rss_mb = max(peak_rss_mb or current_rss_mb, current_rss_mb)

        utterance_parts = []
        in_speech = False
        speech_seconds = 0.0
        silence_seconds = 0.0
        last_partial_decode_audio_time = -1e9
        last_emitted_text = None

    total_chunks = 0
    for start in range(0, samples.shape[0], chunk_size):
        end = min(samples.shape[0], start + chunk_size)
        chunk = samples[start:end]
        chunk_duration_seconds = chunk.shape[0] / float(sample_rate)
        audio_time_seconds = end / float(sample_rate)
        chunk_rms = _speech_rms(chunk)
        total_chunks += 1

        if chunk_rms >= args.rms_threshold:
            if not in_speech:
                utterance_parts = []
                speech_seconds = 0.0
                silence_seconds = 0.0
                last_partial_decode_audio_time = -1e9
                last_emitted_text = None
                in_speech = True
            utterance_parts.append(chunk)
            speech_seconds += chunk_duration_seconds
            silence_seconds = 0.0
        elif in_speech:
            utterance_parts.append(chunk)
            silence_seconds += chunk_duration_seconds

        active_audio_seconds = 0.0
        if utterance_parts:
            active_audio_seconds = sum(part.shape[0] for part in utterance_parts) / float(sample_rate)

        if in_speech and speech_seconds >= min_speech_seconds:
            enough_time_elapsed = (audio_time_seconds - last_partial_decode_audio_time) >= partial_interval_seconds
            if enough_time_elapsed:
                utterance_audio = np.concatenate(utterance_parts)
                text, _payload = _decode_samples(recognizer, utterance_audio, sample_rate)
                last_emitted_text = _maybe_record_update(
                    updates,
                    kind="partial",
                    utterance_index=utterance_index,
                    text=text,
                    audio_time_seconds=audio_time_seconds,
                    benchmark_started=benchmark_started,
                    last_emitted_text=last_emitted_text,
                )
                last_partial_decode_audio_time = audio_time_seconds
                current_rss_mb = _current_rss_mb()
                if current_rss_mb is not None:
                    peak_rss_mb = max(peak_rss_mb or current_rss_mb, current_rss_mb)

        if in_speech and max_active_audio_seconds > 0.0 and active_audio_seconds >= max_active_audio_seconds:
            finalize_utterance(audio_time_seconds)
            continue

        if in_speech and silence_seconds >= endpoint_silence_seconds:
            finalize_utterance(audio_time_seconds)

    if in_speech and utterance_parts:
        finalize_utterance(duration_seconds)

    if not final_texts:
        text, _payload = _decode_samples(recognizer, samples, sample_rate)
        last_emitted_text = _maybe_record_update(
            updates,
            kind="final",
            utterance_index=0,
            text=text,
            audio_time_seconds=duration_seconds,
            benchmark_started=benchmark_started,
            last_emitted_text=last_emitted_text,
        )
        if text.strip():
            final_texts.append(text.strip())
        current_rss_mb = _current_rss_mb()
        if current_rss_mb is not None:
            peak_rss_mb = max(peak_rss_mb or current_rss_mb, current_rss_mb)

    wall_time_seconds = time.perf_counter() - benchmark_started
    realtime_factor = None if duration_seconds <= 0 else wall_time_seconds / duration_seconds

    first_partial = next((event for event in updates if event.kind == "partial"), None)
    first_final = next((event for event in updates if event.kind == "final"), None)

    rss_delta_mb = None
    if baseline_rss_mb is not None and peak_rss_mb is not None:
        rss_delta_mb = peak_rss_mb - baseline_rss_mb

    return FileBenchmarkResult(
        path=str(path),
        mode=MODE_LABEL,
        audio_duration_seconds=duration_seconds,
        recognizer_create_seconds=recognizer_create_seconds,
        wall_time_seconds=wall_time_seconds,
        realtime_factor=realtime_factor,
        first_partial_audio_latency_seconds=None if first_partial is None else first_partial.audio_time_seconds,
        first_partial_wall_latency_seconds=None if first_partial is None else first_partial.wall_time_seconds,
        first_final_audio_latency_seconds=None if first_final is None else first_final.audio_time_seconds,
        first_final_wall_latency_seconds=None if first_final is None else first_final.wall_time_seconds,
        update_count=len(updates),
        final_segment_count=sum(1 for event in updates if event.kind == "final"),
        language_hint=args.language,
        language_hint_applied=False,
        hotwords=args.hotwords,
        baseline_rss_mb=baseline_rss_mb,
        peak_rss_mb=peak_rss_mb,
        rss_delta_mb=rss_delta_mb,
        final_text="\n".join(final_texts).strip(),
        updates=[asdict(event) for event in updates],
    )


def _format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _print_result(result: FileBenchmarkResult) -> None:
    print(f"FILE: {result.path}")
    print(f"  mode: {result.mode}")
    print(f"  language_hint: {result.language_hint} (applied={result.language_hint_applied})")
    print(f"  audio_duration_seconds: {_format_number(result.audio_duration_seconds)}")
    print(f"  recognizer_create_seconds: {_format_number(result.recognizer_create_seconds)}")
    print(f"  wall_time_seconds: {_format_number(result.wall_time_seconds)}")
    print(f"  realtime_factor: {_format_number(result.realtime_factor)}")
    print(
        "  first_partial_latency_seconds: "
        f"audio={_format_number(result.first_partial_audio_latency_seconds)} "
        f"wall={_format_number(result.first_partial_wall_latency_seconds)}"
    )
    print(
        "  first_final_latency_seconds: "
        f"audio={_format_number(result.first_final_audio_latency_seconds)} "
        f"wall={_format_number(result.first_final_wall_latency_seconds)}"
    )
    print(f"  update_count: {result.update_count}")
    print(f"  final_segment_count: {result.final_segment_count}")
    print(
        "  memory_estimate_mb: "
        f"baseline={_format_number(result.baseline_rss_mb, digits=1)} "
        f"peak={_format_number(result.peak_rss_mb, digits=1)} "
        f"delta={_format_number(result.rss_delta_mb, digits=1)}"
    )
    print("  final_text:")
    if result.final_text:
        for line in result.final_text.splitlines():
            print(f"    {line}")
    else:
        print("    <empty>")
    print()


def _print_mode_warning() -> None:
    print("Mode warnings:")
    for line in MODE_WARNING_LINES:
        print(f"  - {line}")
    print()


def main() -> int:
    args = _parse_args()

    if args.language not in SUPPORTED_LANGUAGE_HINTS:
        print(f"Unsupported language hint: {args.language}", file=sys.stderr)
        return 2

    if args.check_model:
        try:
            model_paths = _resolve_model_paths(args.model_dir)
        except FileNotFoundError as exc:
            print(f"Model check failed: {exc}", file=sys.stderr)
            return 2
        _print_mode_warning()
        _print_model_check(model_paths)
        return 0

    inputs = _resolve_inputs(args.inputs)
    if not inputs:
        print(
            "No WAV files found. Provide one or more inputs or a glob such as "
            f"{DEFAULT_INPUT_GLOB}",
            file=sys.stderr,
        )
        return 2

    recognizer, recognizer_create_seconds = _create_recognizer(args)
    results = [
        _benchmark_file(recognizer, recognizer_create_seconds, path, args)
        for path in inputs
    ]

    print(f"Benchmark mode: {MODE_LABEL}")
    print(f"Inputs: {len(results)}")
    print()
    _print_mode_warning()
    for result in results:
        _print_result(result)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": MODE_LABEL,
            "provider": args.provider,
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "language_hint": args.language,
            "language_hint_applied": False,
            "results": [asdict(result) for result in results],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON results to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())