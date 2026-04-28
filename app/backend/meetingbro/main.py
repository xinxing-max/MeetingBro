from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .asr.faster_whisper_adapter import FasterWhisperAdapter
from .audio import AudioSource, MicrophoneSource, MixedAudioSource, SystemAudioLoopbackSource, WavFileSource
from .llm.openai_compatible import _load_dotenv_if_present
from .schemas import (
    CreateNoteRequest,
    ErrorPayload,
    LanguageCode,
    Note,
    SessionStatePayload,
    SummarySnapshot,
    TranscriptSegment,
)
from .session.manager import SessionConfig, SessionManager
from .session.profiles import DEFAULT_RUNTIME_PROFILE, normalize_runtime_profile, runtime_profile_defaults
from .storage.db import Storage
from .summarization.llm import LLMSummarizer
from .translation.llm import LLMTranslator

logger = logging.getLogger("meetingbro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "meetingbro.db"


def _recommended_asr_executor_workers() -> int:
    # SessionManager drives at most one ASR call at a time per session, so more
    # than one ASR worker per session adds thread overhead without throughput gain.
    return 1


def _recommended_summary_executor_workers() -> int:
    cores = os.cpu_count() or 4
    if cores >= 12:
        return 2
    return 1


def _recommended_translation_executor_workers() -> int:
    return 2


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("invalid %s=%r, using default %.3f", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("invalid %s=%r, using default %d", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid %s=%r, using default %s", name, value, default)
    return default


def _profile_value(profile: dict[str, object], key: str, env_name: str, default: object) -> object:
    if env_name in os.environ:
        return os.environ[env_name]
    return profile.get(key, default)


def _profile_float(profile: dict[str, object], key: str, env_name: str, default: float) -> float:
    value = _profile_value(profile, key, env_name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("invalid %s/%s=%r, using default %.3f", key, env_name, value, default)
        return default


def _profile_bool(profile: dict[str, object], key: str, env_name: str, default: bool) -> bool:
    value = _profile_value(profile, key, env_name, default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid %s/%s=%r, using default %s", key, env_name, value, default)
    return default


def _runtime_settings_from_profile(profile_name: str) -> dict[str, object]:
    profile = runtime_profile_defaults(profile_name)
    return {
        "asr_accumulation_seconds": _profile_float(profile, "asr_accumulation_seconds", "MEETINGBRO_ASR_ACCUM_SECONDS", 1.5),
        "asr_early_flush_enabled": _profile_bool(profile, "asr_early_flush_enabled", "MEETINGBRO_ASR_EARLY_FLUSH_ENABLED", True),
        "asr_early_flush_min_seconds": _profile_float(profile, "asr_early_flush_min_seconds", "MEETINGBRO_ASR_EARLY_FLUSH_MIN_SECONDS", 0.8),
        "silence_commit_min_confidence": _profile_float(profile, "silence_commit_min_confidence", "MEETINGBRO_SILENCE_COMMIT_MIN_CONFIDENCE", 0.75),
        "silence_commit_min_duration_seconds": _profile_float(profile, "silence_commit_min_duration_seconds", "MEETINGBRO_SILENCE_COMMIT_MIN_DURATION_SECONDS", 0.6),
        "pre_vad_enabled": _profile_bool(profile, "pre_vad_enabled", "MEETINGBRO_PRE_VAD_ENABLED", True),
        "pre_vad_trailing_silence_seconds": _profile_float(profile, "pre_vad_trailing_silence_seconds", "MEETINGBRO_PRE_VAD_TRAILING_SILENCE_SECONDS", 0.45),
        "pre_vad_adaptive_trailing_silence_enabled": _profile_bool(profile, "pre_vad_adaptive_trailing_silence_enabled", "MEETINGBRO_PRE_VAD_ADAPTIVE_TRAILING_SILENCE_ENABLED", True),
        "pre_vad_adaptive_fast_trailing_silence_seconds": _profile_float(profile, "pre_vad_adaptive_fast_trailing_silence_seconds", "MEETINGBRO_PRE_VAD_ADAPTIVE_FAST_TRAILING_SILENCE_SECONDS", 0.30),
        "language_lock_enabled": _profile_bool(profile, "language_lock_enabled", "MEETINGBRO_LANGUAGE_LOCK_ENABLED", False),
        "weak_speech_rescue_enabled": _profile_bool(profile, "weak_speech_rescue_enabled", "MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED", True),
        "weak_speech_rescue_fast_window_seconds": _profile_float(profile, "weak_speech_rescue_fast_window_seconds", "MEETINGBRO_WEAK_SPEECH_RESCUE_FAST_WINDOW_SECONDS", 2.5),
    }


def _chunk_seconds_for_profile(profile_name: str) -> float:
    profile = runtime_profile_defaults(profile_name)
    return _profile_float(profile, "chunk_seconds", "MEETINGBRO_CHUNK_SECONDS", 0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_dotenv_if_present()
    storage = Storage(Path(os.environ.get("MEETINGBRO_DB", str(DEFAULT_DB_PATH))))
    asr = FasterWhisperAdapter(
        model_size=os.environ.get("MEETINGBRO_WHISPER_SIZE", "medium"),
        device=os.environ.get("MEETINGBRO_WHISPER_DEVICE", "cpu"),
        compute_type=os.environ.get("MEETINGBRO_WHISPER_COMPUTE_TYPE", "int8"),
        beam_size=_env_int("MEETINGBRO_WHISPER_BEAM_SIZE", 3),
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
    app.state.storage = storage
    app.state.asr = asr
    logger.info(
        "MeetingBro backend starting db=%s whisper_size=%s whisper_device=%s compute=%s beam=%d cpu_threads=%d whisper_workers=%d chunk=%.2fs accum=%.2fs silence_rms=%.4f audio_conditioning=%s denoise=%s pre_vad=%s pre_vad_conditioning=%s pre_vad_threshold=%.2f pre_vad_energy_rms=%.4f pre_vad_max=%.1fs weak_rescue=%s weak_rescue_rms=%.4f..%.4f language_lock=%s asr_retry=%s asr_safeguard=%s safeguard_rtf=%.2f suspicious_no_speech=%.2f suspicious_avg_logprob=%.2f suspicious_compression=%.2f mixed_mic_gain=%.2f mixed_system_gain=%.2f mixed_auto_balance=%s mixed_max_mic_boost=%.2f asr_workers=%d summary_workers=%d translation_workers=%d translation_backfill=%d translation_max_pending=%d",
        storage._db_path,
        os.environ.get("MEETINGBRO_WHISPER_SIZE", "medium"),
        os.environ.get("MEETINGBRO_WHISPER_DEVICE", "cpu"),
        os.environ.get("MEETINGBRO_WHISPER_COMPUTE_TYPE", "int8"),
        _env_int("MEETINGBRO_WHISPER_BEAM_SIZE", 3),
        _env_int("MEETINGBRO_WHISPER_CPU_THREADS", 0),
        _env_int("MEETINGBRO_WHISPER_NUM_WORKERS", 1),
        _env_float("MEETINGBRO_CHUNK_SECONDS", 0.5),
        _env_float("MEETINGBRO_ASR_ACCUM_SECONDS", 1.5),
        _env_float("MEETINGBRO_SILENCE_RMS_THRESHOLD", 0.002),
        _env_bool("MEETINGBRO_AUDIO_CONDITIONING_ENABLED", True),
        _env_bool("MEETINGBRO_DENOISE_ENABLED", False),
        _env_bool("MEETINGBRO_PRE_VAD_ENABLED", True),
        _env_bool("MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED", True),
        _env_float("MEETINGBRO_PRE_VAD_THRESHOLD", 0.38),
        _env_float("MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD", 0.005),
        _env_float("MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS", 8.0),
        _env_bool("MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED", True),
        _env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN", 0.0008),
        _env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX", 0.02),
        _env_bool("MEETINGBRO_LANGUAGE_LOCK_ENABLED", False),
        _env_bool("MEETINGBRO_ASR_RETRY_ENABLED", True),
        _env_bool("MEETINGBRO_ASR_SAFEGUARD_ENABLED", True),
        _env_float("MEETINGBRO_ASR_SAFEGUARD_RTF_THRESHOLD", 0.9),
        _env_float("MEETINGBRO_SUSPICIOUS_SEGMENT_NO_SPEECH_PROB", 0.6),
        _env_float("MEETINGBRO_SUSPICIOUS_SEGMENT_AVG_LOGPROB", -0.9),
        _env_float("MEETINGBRO_SUSPICIOUS_SEGMENT_COMPRESSION_RATIO", 2.1),
        _env_float("MEETINGBRO_MIXED_MIC_GAIN", 1.2),
        _env_float("MEETINGBRO_MIXED_SYSTEM_GAIN", 1.0),
        _env_bool("MEETINGBRO_MIXED_AUTO_BALANCE_ENABLED", True),
        _env_float("MEETINGBRO_MIXED_MAX_MIC_BOOST", 1.8),
        _env_int("MEETINGBRO_ASR_EXECUTOR_WORKERS", _recommended_asr_executor_workers()),
        _env_int("MEETINGBRO_SUMMARY_EXECUTOR_WORKERS", _recommended_summary_executor_workers()),
        _env_int("MEETINGBRO_TRANSLATION_EXECUTOR_WORKERS", _recommended_translation_executor_workers()),
        _env_int("MEETINGBRO_LIVE_TRANSLATION_BACKFILL_LIMIT", 20),
        _env_int("MEETINGBRO_LIVE_TRANSLATION_MAX_PENDING", 12),
    )
    try:
        yield
    finally:
        storage.close()
        logger.info("MeetingBro backend stopping")


app = FastAPI(title="MeetingBro backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/notes", response_model=Note)
async def create_note(req: CreateNoteRequest) -> Note:
    storage: Storage = app.state.storage
    note = Note(
        id=str(uuid.uuid4()),
        meeting_id=req.meeting_id,
        content=req.content,
        source_type=req.source_type,
        source_id=req.source_id,
        created_at=datetime.now(tz=timezone.utc),
    )
    storage.insert_note(note)
    return note


@app.get("/meetings/{meeting_id}/notes", response_model=list[Note])
async def list_notes(meeting_id: str) -> list[Note]:
    storage: Storage = app.state.storage
    return storage.list_notes(meeting_id)


@app.get("/meetings/{meeting_id}/transcript", response_model=list[TranscriptSegment])
async def list_transcript(meeting_id: str) -> list[TranscriptSegment]:
    storage: Storage = app.state.storage
    return storage.list_segments(meeting_id)


@app.get("/meetings/{meeting_id}/summaries", response_model=list[SummarySnapshot])
async def list_summaries(
    meeting_id: str,
    summary_type: Optional[str] = Query(default=None),
) -> list[SummarySnapshot]:
    storage: Storage = app.state.storage
    return storage.list_snapshots(meeting_id, summary_type=summary_type)  # type: ignore[arg-type]


def _build_audio_source(source: str, *, chunk_seconds: Optional[float] = None) -> AudioSource:
    """Build an AudioSource from a URI-like spec.

    - "mic" or empty: microphone capture (offline / in-person mode).
    - "loopback" / "system": WASAPI system-audio loopback (Windows-only,
      online meeting mode — Teams/Zoom/BBB audio).
    - "mixed": microphone + system loopback mixed together.
    - "file:<path>": replay a WAV file for the E2E vertical-slice test.
    """
    chunk_seconds = chunk_seconds if chunk_seconds is not None else _env_float("MEETINGBRO_CHUNK_SECONDS", 0.5)
    if not source or source == "mic":
        return MicrophoneSource(sample_rate=16_000, chunk_seconds=chunk_seconds)
    if source in ("loopback", "system"):
        return SystemAudioLoopbackSource(sample_rate=16_000, chunk_seconds=chunk_seconds)
    if source == "mixed":
        return MixedAudioSource(
            sample_rate=16_000,
            chunk_seconds=chunk_seconds,
            microphone_gain=_env_float("MEETINGBRO_MIXED_MIC_GAIN", 1.2),
            system_gain=_env_float("MEETINGBRO_MIXED_SYSTEM_GAIN", 1.0),
            auto_balance_enabled=_env_bool("MEETINGBRO_MIXED_AUTO_BALANCE_ENABLED", True),
            max_microphone_boost=_env_float("MEETINGBRO_MIXED_MAX_MIC_BOOST", 1.8),
            microphone_activity_floor=_env_float("MEETINGBRO_MIXED_MIC_ACTIVITY_FLOOR", 0.008),
            balance_smoothing=_env_float("MEETINGBRO_MIXED_BALANCE_SMOOTHING", 0.35),
        )
    if source.startswith("file:"):
        path = Path(source[len("file:") :]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return WavFileSource(
            path,
            sample_rate=16_000,
            chunk_seconds=_env_float("MEETINGBRO_FILE_CHUNK_SECONDS", 5.0),
            realtime=False,
        )
    raise ValueError(f"unsupported audio source: {source}")


async def _forward_events(manager: SessionManager, ws: WebSocket) -> None:
    async for ev in manager.events():
        await ws.send_text(json.dumps({"type": ev.type, "payload": ev.payload}))


@app.websocket("/ws/session")
async def session_ws(
    ws: WebSocket,
    source: str = Query(default="mic"),
    summary_language: str = Query(default="en"),
    forced_language: Optional[str] = Query(default=None),
    runtime_profile: Optional[str] = Query(default=None),
) -> None:
    await ws.accept()

    summary_lang: LanguageCode = summary_language if summary_language in ("zh", "en", "de") else "en"  # type: ignore[assignment]
    profile_name = normalize_runtime_profile(runtime_profile or os.environ.get("MEETINGBRO_RUNTIME_PROFILE"))
    profile_settings = _runtime_settings_from_profile(profile_name)
    chunk_seconds = _chunk_seconds_for_profile(profile_name)

    try:
        audio_source = _build_audio_source(source, chunk_seconds=chunk_seconds)
    except Exception as exc:
        await ws.send_text(
            json.dumps(
                {
                    "type": "error",
                    "payload": ErrorPayload(code="bad_source", message=str(exc)).model_dump(),
                }
            )
        )
        await ws.close()
        return

    config = SessionConfig(
        audio_source=audio_source,
        audio_source_name=source,
        audio_chunk_seconds=chunk_seconds,
        runtime_profile=profile_name,
        asr=app.state.asr,
        summarizer=LLMSummarizer(),
        translator=LLMTranslator(),
        storage=app.state.storage,
        forced_language=forced_language,
        summary_language=summary_lang,
        live_translation_language=None,
        asr_accumulation_seconds=float(profile_settings["asr_accumulation_seconds"]),
        silence_rms_threshold=_env_float("MEETINGBRO_SILENCE_RMS_THRESHOLD", 0.002),
        asr_overlap_seconds=_env_float("MEETINGBRO_ASR_OVERLAP_SECONDS", 0.0),
        vocabulary_hint=os.environ.get("MEETINGBRO_VOCABULARY_HINT") or None,
        suspicious_segment_no_speech_prob=_env_float(
            "MEETINGBRO_SUSPICIOUS_SEGMENT_NO_SPEECH_PROB",
            0.6,
        ),
        suspicious_segment_avg_logprob=_env_float(
            "MEETINGBRO_SUSPICIOUS_SEGMENT_AVG_LOGPROB",
            -0.9,
        ),
        suspicious_segment_compression_ratio=_env_float(
            "MEETINGBRO_SUSPICIOUS_SEGMENT_COMPRESSION_RATIO",
            2.1,
        ),
        asr_retry_enabled=_env_bool("MEETINGBRO_ASR_RETRY_ENABLED", True),
        asr_safeguard_enabled=_env_bool("MEETINGBRO_ASR_SAFEGUARD_ENABLED", True),
        asr_safeguard_rtf_threshold=_env_float("MEETINGBRO_ASR_SAFEGUARD_RTF_THRESHOLD", 0.9),
        asr_safeguard_cooldown_windows=_env_int("MEETINGBRO_ASR_SAFEGUARD_COOLDOWN_WINDOWS", 5),
        asr_early_flush_enabled=bool(profile_settings["asr_early_flush_enabled"]),
        asr_early_flush_min_seconds=float(profile_settings["asr_early_flush_min_seconds"]),
        silence_commit_min_confidence=float(profile_settings["silence_commit_min_confidence"]),
        silence_commit_min_duration_seconds=float(profile_settings["silence_commit_min_duration_seconds"]),
        denoise_enabled=_env_bool("MEETINGBRO_DENOISE_ENABLED", False),
        denoise_strength=_env_float("MEETINGBRO_DENOISE_STRENGTH", 1.1),
        denoise_noise_update_rms_threshold=_env_float(
            "MEETINGBRO_DENOISE_NOISE_UPDATE_RMS_THRESHOLD",
            0.02,
        ),
        audio_conditioning_enabled=_env_bool("MEETINGBRO_AUDIO_CONDITIONING_ENABLED", True),
        audio_conditioning_target_rms=_env_float("MEETINGBRO_AUDIO_CONDITIONING_TARGET_RMS", 0.035),
        audio_conditioning_min_rms=_env_float("MEETINGBRO_AUDIO_CONDITIONING_MIN_RMS", 0.003),
        audio_conditioning_max_gain=_env_float("MEETINGBRO_AUDIO_CONDITIONING_MAX_GAIN", 2.5),
        audio_conditioning_peak_limit=_env_float("MEETINGBRO_AUDIO_CONDITIONING_PEAK_LIMIT", 0.98),
        pre_vad_enabled=bool(profile_settings["pre_vad_enabled"]),
        pre_vad_conditioning_enabled=_env_bool("MEETINGBRO_PRE_VAD_CONDITIONING_ENABLED", True),
        pre_vad_conditioning_target_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_TARGET_RMS", 0.03),
        pre_vad_conditioning_min_rms=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MIN_RMS", 0.001),
        pre_vad_conditioning_max_gain=_env_float("MEETINGBRO_PRE_VAD_CONDITIONING_MAX_GAIN", 4.0),
        pre_vad_threshold=_env_float("MEETINGBRO_PRE_VAD_THRESHOLD", 0.38),
        pre_vad_energy_rms_threshold=_env_float("MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD", 0.005),
        pre_vad_trailing_silence_seconds=float(profile_settings["pre_vad_trailing_silence_seconds"]),
        pre_vad_adaptive_trailing_silence_enabled=bool(profile_settings["pre_vad_adaptive_trailing_silence_enabled"]),
        pre_vad_adaptive_fast_trailing_silence_seconds=float(profile_settings["pre_vad_adaptive_fast_trailing_silence_seconds"]),
        pre_vad_max_segment_seconds=_env_float("MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS", 8.0),
        weak_speech_rescue_enabled=bool(profile_settings["weak_speech_rescue_enabled"]),
        weak_speech_rescue_rms_min=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MIN", 0.0008),
        weak_speech_rescue_rms_max=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_RMS_MAX", 0.02),
        weak_speech_rescue_fast_window_seconds=float(profile_settings["weak_speech_rescue_fast_window_seconds"]),
        weak_speech_rescue_window_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_WINDOW_SECONDS", 6.0),
        weak_speech_rescue_cooldown_seconds=_env_float("MEETINGBRO_WEAK_SPEECH_RESCUE_COOLDOWN_SECONDS", 8.0),
        language_lock_enabled=bool(profile_settings["language_lock_enabled"]),
        live_translation_backfill_limit=_env_int("MEETINGBRO_LIVE_TRANSLATION_BACKFILL_LIMIT", 20),
        live_translation_max_pending=_env_int("MEETINGBRO_LIVE_TRANSLATION_MAX_PENDING", 12),
        live_translation_safeguard_max_pending=_env_int("MEETINGBRO_LIVE_TRANSLATION_SAFEGUARD_MAX_PENDING", 4),
        asr_executor_workers=_env_int(
            "MEETINGBRO_ASR_EXECUTOR_WORKERS",
            _recommended_asr_executor_workers(),
        ),
        summary_executor_workers=_env_int(
            "MEETINGBRO_SUMMARY_EXECUTOR_WORKERS",
            _recommended_summary_executor_workers(),
        ),
        translation_executor_workers=_env_int(
            "MEETINGBRO_TRANSLATION_EXECUTOR_WORKERS",
            _recommended_translation_executor_workers(),
        ),
    )
    manager = SessionManager(config)
    await manager.start()
    logger.info(
        "session ws accepted meeting_id=%s source=%s profile=%s chunk=%.2fs accum=%.2fs",
        manager.meeting_id,
        source,
        profile_name,
        chunk_seconds,
        config.asr_accumulation_seconds,
    )

    forward_task = asyncio.create_task(_forward_events(manager, ws))

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "save_note":
                payload = data.get("payload") or {}
                content = (payload.get("content") or "").strip()
                if content:
                    note = manager.save_note(
                        content=content,
                        source_type=payload.get("source_type"),
                        source_id=payload.get("source_id"),
                    )
                    await ws.send_text(
                        json.dumps(
                            {"type": "note_saved", "payload": note.model_dump(mode="json")}
                        )
                    )
            elif data.get("type") == "update_settings":
                payload = data.get("payload") or {}
                next_summary_language = payload.get("summary_language")
                summary_lang = (
                    next_summary_language
                    if next_summary_language in ("zh", "en", "de")
                    else None
                )
                forced_language = payload.get("forced_language")
                if forced_language == "auto":
                    forced_language = None
                elif forced_language not in (None, "zh", "en", "de"):
                    forced_language = manager._cfg.forced_language
                live_translation_language = payload.get("subtitle_language")
                if live_translation_language == "off":
                    live_translation_language = None
                elif live_translation_language not in (None, "zh", "en", "de"):
                    live_translation_language = manager._cfg.live_translation_language
                next_source = payload.get("source")
                next_profile = normalize_runtime_profile(payload.get("runtime_profile") or manager._cfg.runtime_profile)
                next_profile_settings = _runtime_settings_from_profile(next_profile)
                next_chunk_seconds = _chunk_seconds_for_profile(next_profile)
                profile_changed = next_profile != manager._cfg.runtime_profile
                chunk_changed = abs(next_chunk_seconds - manager._cfg.audio_chunk_seconds) > 1e-6
                if next_source and (next_source != source or (profile_changed and chunk_changed)):
                    try:
                        next_audio_source = _build_audio_source(next_source, chunk_seconds=next_chunk_seconds)
                    except Exception as exc:
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "payload": ErrorPayload(code="bad_source", message=str(exc)).model_dump(),
                                }
                            )
                        )
                    else:
                        manager.update_audio_source(
                            next_audio_source,
                            source_name=next_source,
                            chunk_seconds=next_chunk_seconds,
                        )
                        source = next_source
                manager.update_runtime_settings(
                    forced_language=forced_language,
                    summary_language=summary_lang,
                    live_translation_language=live_translation_language,
                    runtime_profile=next_profile,
                    runtime_settings={
                        **next_profile_settings,
                        "audio_chunk_seconds": next_chunk_seconds,
                    },
                )
            elif data.get("type") == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await manager.stop()
        forward_task.cancel()
        try:
            await forward_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await ws.close()
        except Exception:
            pass


def run() -> None:
    uvicorn.run(
        "meetingbro.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
