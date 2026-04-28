from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .capture import AudioChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechSpan:
    start_frame: int
    end_frame: int


class SpeechDetector(Protocol):
    def detect(self, samples: np.ndarray, sample_rate: int) -> list[SpeechSpan]:
        ...


_SILERO_MODEL = None
_SILERO_LOCK = threading.Lock()


class SileroSpeechDetector:
    """Detect speech spans using silero-vad with an energy fallback.

    The fallback keeps the pipeline functional if the optional dependency is not
    installed yet, but the intended production path is silero-vad.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 120,
        min_silence_duration_ms: int = 250,
        energy_rms_threshold: float = 0.01,
    ) -> None:
        self._threshold = threshold
        self._min_speech_duration_ms = min_speech_duration_ms
        self._min_silence_duration_ms = min_silence_duration_ms
        self._energy_rms_threshold = energy_rms_threshold
        self._warned_fallback = False

    def _get_model(self):
        global _SILERO_MODEL
        if _SILERO_MODEL is not None:
            return _SILERO_MODEL
        with _SILERO_LOCK:
            if _SILERO_MODEL is not None:
                return _SILERO_MODEL
            from silero_vad import load_silero_vad

            _SILERO_MODEL = load_silero_vad()
            return _SILERO_MODEL

    def detect(self, samples: np.ndarray, sample_rate: int) -> list[SpeechSpan]:
        if samples.size == 0:
            return []
        try:
            from silero_vad import get_speech_timestamps

            timestamps = get_speech_timestamps(
                samples,
                self._get_model(),
                sampling_rate=sample_rate,
                threshold=self._threshold,
                min_speech_duration_ms=self._min_speech_duration_ms,
                min_silence_duration_ms=self._min_silence_duration_ms,
                return_seconds=False,
            )
            return [
                SpeechSpan(start_frame=int(item["start"]), end_frame=int(item["end"]))
                for item in timestamps
                if int(item["end"]) > int(item["start"])
            ]
        except Exception as exc:
            if not self._warned_fallback:
                logger.warning("silero-vad unavailable, falling back to RMS gate: %s", exc)
                self._warned_fallback = True
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if rms < self._energy_rms_threshold:
                return []
            return [SpeechSpan(start_frame=0, end_frame=len(samples))]


class PreVadSegmenter:
    """Stateful pre-ASR segmenter driven by speech activity.

    Input chunks may be fixed-size capture blocks. This class buffers only the
    regions that contain speech and emits a coarser utterance-sized chunk once a
    trailing silence boundary is observed.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        detector: Optional[SpeechDetector] = None,
        enabled: bool = True,
        detector_threshold: float = 0.38,
        detector_energy_rms_threshold: float = 0.005,
        pre_speech_pad_ms: int = 120,
        post_speech_pad_ms: int = 220,
        trailing_silence_seconds: float = 0.45,
        max_segment_seconds: float = 12.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._detector = detector or SileroSpeechDetector(
            threshold=detector_threshold,
            energy_rms_threshold=detector_energy_rms_threshold,
        )
        self._enabled = enabled
        self._pre_speech_pad_frames = int(sample_rate * pre_speech_pad_ms / 1000)
        self._post_speech_pad_frames = int(sample_rate * post_speech_pad_ms / 1000)
        self._trailing_silence_seconds = trailing_silence_seconds
        self._max_segment_seconds = max_segment_seconds
        self._pending_samples: list[np.ndarray] = []
        self._pending_start_time: Optional[float] = None
        self._silence_after_speech_seconds: float = 0.0

    def push(self, chunk: AudioChunk) -> list[AudioChunk]:
        if not self._enabled:
            return [chunk]

        out: list[AudioChunk] = []
        spans = self._detector.detect(chunk.samples, chunk.sample_rate)
        chunk_duration = len(chunk.samples) / chunk.sample_rate

        if spans:
            start_frame = max(0, spans[0].start_frame - self._pre_speech_pad_frames)
            end_frame = min(len(chunk.samples), spans[-1].end_frame + self._post_speech_pad_frames)
            trimmed = chunk.samples[start_frame:end_frame]
            if trimmed.size > 0:
                out.extend(
                    self._append_speech_samples(
                        trimmed.astype(np.float32, copy=False),
                        chunk.start_time + start_frame / chunk.sample_rate,
                    )
                )
                speech_end_seconds = end_frame / chunk.sample_rate
                self._silence_after_speech_seconds = max(0.0, chunk_duration - speech_end_seconds)
        elif self._pending_samples:
            self._silence_after_speech_seconds += chunk_duration

        if self._pending_samples and self._silence_after_speech_seconds >= self._trailing_silence_seconds:
            flushed = self._flush_pending()
            if flushed is not None:
                out.append(flushed)

        return out

    def finish(self) -> list[AudioChunk]:
        flushed = self._flush_pending()
        return [flushed] if flushed is not None else []

    def _pending_duration_seconds(self) -> float:
        if not self._pending_samples:
            return 0.0
        return sum(len(part) for part in self._pending_samples) / self._sample_rate

    def _pending_frame_count(self) -> int:
        return sum(len(part) for part in self._pending_samples)

    def _append_speech_samples(self, samples: np.ndarray, start_time: float) -> list[AudioChunk]:
        out: list[AudioChunk] = []
        if samples.size == 0:
            return out

        max_frames = max(1, int(self._max_segment_seconds * self._sample_rate))
        offset = 0
        while offset < len(samples):
            if self._pending_start_time is None:
                self._pending_start_time = start_time + offset / self._sample_rate
                self._pending_samples.clear()

            remaining = max_frames - self._pending_frame_count()
            if remaining <= 0:
                flushed = self._flush_pending()
                if flushed is not None:
                    out.append(flushed)
                continue

            take = min(remaining, len(samples) - offset)
            if take <= 0:
                break
            self._pending_samples.append(samples[offset : offset + take].copy())
            offset += take

            if self._pending_frame_count() >= max_frames:
                flushed = self._flush_pending()
                if flushed is not None:
                    out.append(flushed)

        return out

    def _flush_pending(self) -> Optional[AudioChunk]:
        if not self._pending_samples or self._pending_start_time is None:
            self._pending_samples.clear()
            self._pending_start_time = None
            self._silence_after_speech_seconds = 0.0
            return None

        merged = np.concatenate(self._pending_samples).astype(np.float32, copy=False)
        out = AudioChunk(
            samples=merged,
            sample_rate=self._sample_rate,
            start_time=self._pending_start_time,
        )
        self._pending_samples.clear()
        self._pending_start_time = None
        self._silence_after_speech_seconds = 0.0
        return out
