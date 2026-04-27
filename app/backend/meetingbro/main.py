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
from .audio import AudioSource, MicrophoneSource, SystemAudioLoopbackSource, WavFileSource
from .diarization.energy import EnergyDiarizer
from .schemas import (
    CreateNoteRequest,
    ErrorPayload,
    LanguageCode,
    Note,
    SessionStatePayload,
    Speaker,
    SummarySnapshot,
    TranscriptSegment,
)
from .session.manager import SessionConfig, SessionManager
from .storage.db import Storage
from .summarization.llm import LLMSummarizer
from .translation.llm import LLMTranslator

logger = logging.getLogger("meetingbro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "meetingbro.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage = Storage(Path(os.environ.get("MEETINGBRO_DB", str(DEFAULT_DB_PATH))))
    asr = FasterWhisperAdapter(
        model_size=os.environ.get("MEETINGBRO_WHISPER_SIZE", "tiny"),
    )
    app.state.storage = storage
    app.state.asr = asr
    logger.info("MeetingBro backend starting db=%s", storage._db_path)
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


@app.get("/meetings/{meeting_id}/speakers", response_model=list[Speaker])
async def list_speakers(meeting_id: str) -> list[Speaker]:
    storage: Storage = app.state.storage
    return storage.list_speakers(meeting_id)


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


def _build_audio_source(source: str) -> AudioSource:
    """Build an AudioSource from a URI-like spec.

    - "mic" or empty: microphone capture (offline / in-person mode).
    - "loopback" / "system": WASAPI system-audio loopback (Windows-only,
      online meeting mode — Teams/Zoom/BBB audio).
    - "file:<path>": replay a WAV file for the E2E vertical-slice test.
    """
    if not source or source == "mic":
        return MicrophoneSource(sample_rate=16_000, chunk_seconds=1.5)
    if source in ("loopback", "system"):
        return SystemAudioLoopbackSource(sample_rate=16_000, chunk_seconds=1.5)
    if source.startswith("file:"):
        path = Path(source[len("file:") :]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return WavFileSource(path, sample_rate=16_000, chunk_seconds=5.0, realtime=False)
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
) -> None:
    await ws.accept()

    summary_lang: LanguageCode = summary_language if summary_language in ("zh", "en", "de") else "en"  # type: ignore[assignment]

    try:
        audio_source = _build_audio_source(source)
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
        asr=app.state.asr,
        summarizer=LLMSummarizer(),
        translator=LLMTranslator(),
        storage=app.state.storage,
        diarizer=EnergyDiarizer(),
        forced_language=forced_language,
        summary_language=summary_lang,
    )
    manager = SessionManager(config)
    await manager.start()
    logger.info("session ws accepted meeting_id=%s source=%s", manager.meeting_id, source)

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
