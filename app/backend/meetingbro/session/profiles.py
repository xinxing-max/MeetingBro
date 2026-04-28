from __future__ import annotations

from typing import Any


DEFAULT_RUNTIME_PROFILE = "balanced"
RUNTIME_PROFILE_NAMES = {"balanced", "low_latency", "robust", "multilingual", "single_language"}


RUNTIME_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    # Good default for real meetings: pre-VAD keeps long runs stable while early
    # flush and silence-boundary commits keep the first result responsive.
    "balanced": {
        "chunk_seconds": 0.5,
        "asr_accumulation_seconds": 1.5,
        "asr_early_flush_enabled": True,
        "asr_early_flush_min_seconds": 0.8,
        "silence_commit_min_confidence": 0.75,
        "silence_commit_min_duration_seconds": 0.6,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.45,
        "pre_vad_adaptive_trailing_silence_enabled": True,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.30,
        "language_lock_enabled": False,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 2.5,
    },
    # Prefer faster visible feedback. This may emit slightly smaller chunks and
    # can be less stable on long/noisy audio than balanced/robust.
    "low_latency": {
        "chunk_seconds": 0.5,
        "asr_accumulation_seconds": 1.0,
        "asr_early_flush_enabled": True,
        "asr_early_flush_min_seconds": 0.6,
        "silence_commit_min_confidence": 0.70,
        "silence_commit_min_duration_seconds": 0.4,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.35,
        "pre_vad_adaptive_trailing_silence_enabled": True,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.25,
        "language_lock_enabled": False,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 1.8,
    },
    # Prefer stability over first-word latency: longer context and less
    # aggressive silence commits reduce fragmenting and hallucination risk.
    "robust": {
        "chunk_seconds": 0.75,
        "asr_accumulation_seconds": 2.0,
        "asr_early_flush_enabled": True,
        "asr_early_flush_min_seconds": 1.0,
        "silence_commit_min_confidence": 0.80,
        "silence_commit_min_duration_seconds": 0.8,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.60,
        "pre_vad_adaptive_trailing_silence_enabled": False,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.30,
        "language_lock_enabled": False,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 3.0,
    },
    # Explicit mixed-language profile; currently close to balanced but documents
    # and enforces language-lock-off behavior for Chinese/English/German meetings.
    "multilingual": {
        "chunk_seconds": 0.5,
        "asr_accumulation_seconds": 1.5,
        "asr_early_flush_enabled": True,
        "asr_early_flush_min_seconds": 0.8,
        "silence_commit_min_confidence": 0.75,
        "silence_commit_min_duration_seconds": 0.6,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.45,
        "pre_vad_adaptive_trailing_silence_enabled": True,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.30,
        "language_lock_enabled": False,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 2.5,
    },
    # For mostly one-language meetings in auto-detect mode. Forced-language mode
    # still bypasses the lock inside SessionManager.
    "single_language": {
        "chunk_seconds": 0.5,
        "asr_accumulation_seconds": 1.5,
        "asr_early_flush_enabled": True,
        "asr_early_flush_min_seconds": 0.8,
        "silence_commit_min_confidence": 0.75,
        "silence_commit_min_duration_seconds": 0.6,
        "pre_vad_enabled": True,
        "pre_vad_trailing_silence_seconds": 0.45,
        "pre_vad_adaptive_trailing_silence_enabled": True,
        "pre_vad_adaptive_fast_trailing_silence_seconds": 0.30,
        "language_lock_enabled": True,
        "weak_speech_rescue_enabled": True,
        "weak_speech_rescue_fast_window_seconds": 2.5,
    },
}


def normalize_runtime_profile(name: str | None) -> str:
    normalized = (name or DEFAULT_RUNTIME_PROFILE).strip().lower().replace("-", "_")
    if normalized in RUNTIME_PROFILE_NAMES:
        return normalized
    return DEFAULT_RUNTIME_PROFILE


def runtime_profile_defaults(name: str | None) -> dict[str, Any]:
    return dict(RUNTIME_PROFILE_PRESETS[normalize_runtime_profile(name)])
