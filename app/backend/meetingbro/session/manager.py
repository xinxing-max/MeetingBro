from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import numpy as np

from ..asr.base import ASRAdapter
from ..audio.capture import AudioSource
from ..diarization.base import Diarizer
from ..schemas import (
    ErrorPayload,
    LanguageCode,
    Note,
    SessionStatePayload,
    Speaker,
    SummarySnapshot,
    SummaryType,
    TranscriptSegment,
)
from ..storage.db import Storage
from ..summarization.base import Summarizer
from ..translation.base import Translator

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    audio_source: AudioSource
    asr: ASRAdapter
    summarizer: Summarizer
    translator: Translator
    storage: Storage
    diarizer: Optional[Diarizer] = None
    forced_language: Optional[str] = None  # None => auto-detect
    summary_language: LanguageCode = "en"
    rolling_window_seconds: float = 180.0  # last ~3 minutes for rolling summary input
    rolling_interval_seconds: float = 60.0  # cadence target for rolling refresh
    cumulative_interval_seconds: float = 180.0  # cadence target for cumulative refresh
    min_segments_for_rolling: int = 1
    min_segments_for_cumulative: int = 3
    asr_accumulation_seconds: float = 2.5  # accumulate audio before running ASR


@dataclass
class SessionEvent:
    type: str
    payload: dict


@dataclass
class _State:
    meeting_id: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    last_rolling_at: float = 0.0
    last_cumulative_at: float = 0.0
    latest_cumulative_text: Optional[str] = None
    elapsed_seconds: float = 0.0
    speakers: dict[str, Speaker] = field(default_factory=dict)  # label -> Speaker


class SessionManager:
    """Orchestrates one live meeting session.

    Pipeline per audio chunk:
      audio_source -> ASR -> TranscriptSegment persisted & emitted ->
      (on cadence) rolling_summary built from last N seconds of segments ->
      (on cadence) cumulative_meeting_summary built from all segments so far.
    """

    def __init__(self, config: SessionConfig) -> None:
        self._cfg = config
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._state = _State(meeting_id=str(uuid.uuid4()))
        self._stopped = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None
        self._summary_tasks: set[asyncio.Task[None]] = set()
        self._pending_speaker_updates: list[Speaker] = []

    @property
    def meeting_id(self) -> str:
        return self._state.meeting_id

    async def events(self) -> AsyncIterator[SessionEvent]:
        while not (self._stopped.is_set() and self._event_queue.empty()):
            try:
                ev = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            yield ev

    async def _emit(self, event_type: str, payload_model) -> None:
        payload = (
            payload_model.model_dump(mode="json")
            if hasattr(payload_model, "model_dump")
            else dict(payload_model)
        )
        await self._event_queue.put(SessionEvent(type=event_type, payload=payload))

    def save_note(self, content: str, source_type: Optional[str] = None, source_id: Optional[str] = None) -> Note:
        note = Note(
            id=str(uuid.uuid4()),
            meeting_id=self._state.meeting_id,
            content=content,
            source_type=source_type,
            source_id=source_id,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._cfg.storage.insert_note(note)
        return note

    async def start(self) -> None:
        self._cfg.storage.create_meeting(
            self._state.meeting_id,
            preferred_summary_language=self._cfg.summary_language,
        )
        await self._emit(
            "session_state",
            SessionStatePayload(state="running", meeting_id=self._state.meeting_id),
        )
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        try:
            await self._cfg.audio_source.aclose()
        except Exception:
            logger.exception("audio_source.aclose failed")
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        # Wait for any pending summary background tasks.
        if self._summary_tasks:
            done, pending = await asyncio.wait(self._summary_tasks, timeout=10.0)
            for t in pending:
                t.cancel()
            self._summary_tasks.clear()
        self._cfg.storage.end_meeting(self._state.meeting_id)
        await self._emit(
            "session_state",
            SessionStatePayload(state="ended", meeting_id=self._state.meeting_id),
        )

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        # Accumulation buffer: collect small audio chunks until we have enough
        # for a quality ASR pass, reducing latency vs the old 3s monolithic chunks.
        accum_samples: list[np.ndarray] = []
        accum_duration: float = 0.0
        accum_start_time: float = 0.0
        threshold = self._cfg.asr_accumulation_seconds

        try:
            async for chunk in self._cfg.audio_source.stream():
                if self._stopped.is_set():
                    break

                if not accum_samples:
                    accum_start_time = chunk.start_time

                accum_samples.append(chunk.samples)
                accum_duration += len(chunk.samples) / chunk.sample_rate

                if accum_duration < threshold:
                    continue

                # Flush the accumulated buffer through ASR.
                merged = np.concatenate(accum_samples)
                buf_start = accum_start_time
                sample_rate = chunk.sample_rate
                accum_samples.clear()
                accum_duration = 0.0

                segments = await loop.run_in_executor(
                    None,
                    lambda m=merged, sr=sample_rate: self._cfg.asr.transcribe(
                        m,
                        sr,
                        forced_language=self._cfg.forced_language,
                        offset_seconds=0.0,
                    ),
                )

                # Run diarization in parallel with segment processing.
                dia_segments = None
                if self._cfg.diarizer is not None:
                    dia_segments = await loop.run_in_executor(
                        None,
                        lambda m=merged, sr=sample_rate, bs=buf_start: self._cfg.diarizer.diarize(
                            m, sr, offset_seconds=bs,
                        ),
                    )

                for asr_seg in segments:
                    seg = TranscriptSegment(
                        id=str(uuid.uuid4()),
                        meeting_id=self._state.meeting_id,
                        start_time=buf_start + asr_seg.start_time,
                        end_time=buf_start + asr_seg.end_time,
                        text=asr_seg.text,
                        original_language=asr_seg.language,
                        confidence=asr_seg.confidence,
                    )

                    # Assign speaker from diarization by best time overlap.
                    if dia_segments:
                        speaker_label = self._match_speaker(
                            seg.start_time, seg.end_time, dia_segments,
                        )
                        if speaker_label:
                            speaker = self._ensure_speaker(speaker_label)
                            seg.speaker_id = speaker.id
                    if (
                        seg.original_language != "unknown"
                        and self._cfg.summary_language != seg.original_language
                    ):
                        seg.translations[self._cfg.summary_language] = (
                            self._cfg.translator.translate(
                                seg.text,
                                source_language=seg.original_language,
                                target_language=self._cfg.summary_language,
                            )
                        )
                    self._state.segments.append(seg)
                    self._cfg.storage.insert_segment(seg)
                    await self._emit("transcript_segment", seg)

                # Emit any new speaker updates.
                for sp in self._pending_speaker_updates:
                    await self._emit("speaker_update", sp)
                self._pending_speaker_updates.clear()

                self._state.elapsed_seconds = max(
                    self._state.elapsed_seconds,
                    buf_start + len(merged) / sample_rate,
                )

                # Fire summary checks as background tasks so they don't block
                # the audio→ASR→emit pipeline.
                self._schedule_summary(self._maybe_emit_rolling())
                self._schedule_summary(self._maybe_emit_cumulative())

            # Flush remaining accumulated audio.
            if accum_samples:
                merged = np.concatenate(accum_samples)
                sample_rate = self._cfg.audio_source._sample_rate if hasattr(self._cfg.audio_source, '_sample_rate') else 16_000
                segments = await loop.run_in_executor(
                    None,
                    lambda m=merged, sr=sample_rate: self._cfg.asr.transcribe(
                        m,
                        sr,
                        forced_language=self._cfg.forced_language,
                        offset_seconds=0.0,
                    ),
                )
                for asr_seg in segments:
                    seg = TranscriptSegment(
                        id=str(uuid.uuid4()),
                        meeting_id=self._state.meeting_id,
                        start_time=accum_start_time + asr_seg.start_time,
                        end_time=accum_start_time + asr_seg.end_time,
                        text=asr_seg.text,
                        original_language=asr_seg.language,
                        confidence=asr_seg.confidence,
                    )
                    if (
                        seg.original_language != "unknown"
                        and self._cfg.summary_language != seg.original_language
                    ):
                        seg.translations[self._cfg.summary_language] = (
                            self._cfg.translator.translate(
                                seg.text,
                                source_language=seg.original_language,
                                target_language=self._cfg.summary_language,
                            )
                        )
                    self._state.segments.append(seg)
                    self._cfg.storage.insert_segment(seg)
                    await self._emit("transcript_segment", seg)
                self._state.elapsed_seconds = max(
                    self._state.elapsed_seconds,
                    accum_start_time + len(merged) / sample_rate,
                )

            # Wait for in-flight summary tasks before final summary.
            if self._summary_tasks:
                await asyncio.wait(self._summary_tasks, timeout=15.0)
                self._summary_tasks.clear()

            await self._emit_final()
        except Exception as exc:
            logger.exception("session loop crashed")
            await self._emit(
                "error",
                ErrorPayload(code="session_loop", message=str(exc)),
            )

    def _schedule_summary(self, coro) -> None:
        """Run a summary coroutine as a background task."""
        task = asyncio.create_task(coro)
        self._summary_tasks.add(task)
        task.add_done_callback(self._summary_tasks.discard)

    def _match_speaker(
        self, seg_start: float, seg_end: float, dia_segments
    ) -> Optional[str]:
        """Find the diarization segment with the best time overlap."""
        best_label = None
        best_overlap = 0.0
        for ds in dia_segments:
            overlap_start = max(seg_start, ds.start_time)
            overlap_end = min(seg_end, ds.end_time)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = ds.speaker_label
        return best_label

    def _ensure_speaker(self, label: str) -> Speaker:
        """Get or create a Speaker for the given label, persisting to DB."""
        if label in self._state.speakers:
            return self._state.speakers[label]
        speaker = Speaker(
            id=str(uuid.uuid4()),
            meeting_id=self._state.meeting_id,
            inferred_label=label,
            confidence=0.5,
        )
        self._state.speakers[label] = speaker
        self._cfg.storage.upsert_speaker(speaker)
        self._pending_speaker_updates.append(speaker)
        return speaker

    def _segments_in_window(self, window_seconds: float) -> list[TranscriptSegment]:
        cutoff = max(0.0, self._state.elapsed_seconds - window_seconds)
        return [s for s in self._state.segments if s.end_time >= cutoff]

    async def _build_and_emit_snapshot(
        self,
        *,
        summary_type: SummaryType,
        segments: list[TranscriptSegment],
        time_start: float,
        time_end: float,
        previous_summary: Optional[str] = None,
    ) -> Optional[SummarySnapshot]:
        if not segments:
            return None
        kind = summary_type if summary_type in {
            "rolling_summary",
            "cumulative_meeting_summary",
            "final_summary",
        } else "rolling_summary"
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(
            None,
            lambda: self._cfg.summarizer.summarize(
                segments,
                kind=kind,  # type: ignore[arg-type]
                language=self._cfg.summary_language,
                previous_summary=previous_summary,
            ),
        )
        if not content:
            return None
        snap = SummarySnapshot(
            id=str(uuid.uuid4()),
            meeting_id=self._state.meeting_id,
            summary_type=summary_type,
            time_start=time_start,
            time_end=time_end,
            language=self._cfg.summary_language,
            content=content,
            source_segment_ids=[s.id for s in segments],
            is_latest=True,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._cfg.storage.insert_snapshot(snap)
        await self._emit("summary_snapshot", snap)
        return snap

    async def _maybe_emit_rolling(self) -> None:
        now = self._state.elapsed_seconds
        due = (now - self._state.last_rolling_at) >= self._cfg.rolling_interval_seconds
        if not due:
            return
        window_segments = self._segments_in_window(self._cfg.rolling_window_seconds)
        if len(window_segments) < self._cfg.min_segments_for_rolling:
            return
        start = min(s.start_time for s in window_segments)
        end = max(s.end_time for s in window_segments)
        snap = await self._build_and_emit_snapshot(
            summary_type="rolling_summary",
            segments=window_segments,
            time_start=start,
            time_end=end,
        )
        if snap is not None:
            self._state.last_rolling_at = now

    async def _maybe_emit_cumulative(self) -> None:
        now = self._state.elapsed_seconds
        due = (now - self._state.last_cumulative_at) >= self._cfg.cumulative_interval_seconds
        if not due:
            return
        all_segments = list(self._state.segments)
        if len(all_segments) < self._cfg.min_segments_for_cumulative:
            return
        end = max(s.end_time for s in all_segments)
        snap = await self._build_and_emit_snapshot(
            summary_type="cumulative_meeting_summary",
            segments=all_segments,
            time_start=0.0,
            time_end=end,
            previous_summary=self._state.latest_cumulative_text,
        )
        if snap is not None:
            self._state.last_cumulative_at = now
            self._state.latest_cumulative_text = snap.content

    async def _emit_final(self) -> None:
        if not self._state.segments:
            return
        end = max(s.end_time for s in self._state.segments)
        await self._build_and_emit_snapshot(
            summary_type="final_summary",
            segments=list(self._state.segments),
            time_start=0.0,
            time_end=end,
            previous_summary=self._state.latest_cumulative_text,
        )
