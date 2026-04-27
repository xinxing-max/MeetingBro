"""Audio capture abstractions.

Supported sources for the MVP vertical slice:

- ``MicrophoneSource``  — Windows / macOS / Linux microphone via ``sounddevice``.
  Offline / in-person meeting mode. This is the **first targeted OS path** for
  the MVP; system-audio loopback (online meeting mode) lands in a later phase.

- ``WavFileSource`` — replays a WAV file as an audio stream. Used for the
  deterministic end-to-end vertical-slice test and for dev without a real mic.

Both sources yield fixed-duration ``AudioChunk`` objects of mono ``float32`` PCM
at a configurable sample rate (default 16 kHz to match Whisper).
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A fixed-duration slice of mono float32 PCM."""

    samples: np.ndarray  # shape (N,), dtype float32, range ~[-1, 1]
    sample_rate: int
    start_time: float  # seconds since session start


class AudioSource(ABC):
    """Abstract audio source. Implementations stream ``AudioChunk``s."""

    @abstractmethod
    async def stream(self) -> AsyncIterator[AudioChunk]:
        ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


def _resample_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    ratio = dst_rate / src_rate
    out_len = int(round(len(samples) * ratio))
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float64)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


class MicrophoneSource(AudioSource):
    """Microphone capture via ``sounddevice``.

    ``chunk_seconds`` controls how often the source emits an ``AudioChunk``. The
    default (3.0 s) is a balance between ASR quality and UI latency.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.0,
        device: Optional[int | str] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._device = device
        self._stop = threading.Event()

    async def stream(self) -> AsyncIterator[AudioChunk]:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sounddevice is required for MicrophoneSource"
            ) from exc

        loop = asyncio.get_running_loop()
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        chunk_frames = int(self._sample_rate * self._chunk_seconds)

        def _callback(indata, frames, time_info, status):  # noqa: ARG001 - sd API
            if status:
                logger.debug("audio status: %s", status)
            # indata shape: (frames, channels). We force mono by averaging.
            mono = indata.mean(axis=1).astype(np.float32, copy=False).copy()
            try:
                q.put_nowait(mono)
            except queue.Full:
                logger.warning("audio queue full — dropping chunk")

        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            device=self._device,
            callback=_callback,
        )
        stream.start()
        logger.info(
            "mic capture started rate=%d chunk_frames=%d device=%s",
            self._sample_rate, chunk_frames, self._device,
        )
        t0 = 0.0
        try:
            while not self._stop.is_set():
                mono = await loop.run_in_executor(None, q.get)
                yield AudioChunk(samples=mono, sample_rate=self._sample_rate, start_time=t0)
                t0 += len(mono) / self._sample_rate
        finally:
            stream.stop()
            stream.close()
            logger.info("mic capture stopped")

    async def aclose(self) -> None:
        self._stop.set()


class WavFileSource(AudioSource):
    """Replay a WAV file as if it were a live stream."""

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.0,
        realtime: bool = False,
    ) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._realtime = realtime

    async def stream(self) -> AsyncIterator[AudioChunk]:
        import soundfile as sf

        with sf.SoundFile(str(self._path), mode="r") as f:
            src_rate = f.samplerate
            samples = f.read(dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = _resample_mono(samples, src_rate, self._sample_rate)

        chunk_frames = int(self._sample_rate * self._chunk_seconds)
        t0 = 0.0
        for start in range(0, len(samples), chunk_frames):
            chunk = samples[start : start + chunk_frames]
            if len(chunk) == 0:
                break
            yield AudioChunk(samples=chunk, sample_rate=self._sample_rate, start_time=t0)
            t0 += len(chunk) / self._sample_rate
            if self._realtime:
                await asyncio.sleep(len(chunk) / self._sample_rate)
