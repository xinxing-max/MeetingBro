from __future__ import annotations

import logging

import numpy as np

from .capture import AudioChunk

logger = logging.getLogger(__name__)


class AdaptiveNoiseReducer:
    """Lightweight streaming noise reducer for meeting audio.

    This is intentionally modest: it performs spectral subtraction with a
    slowly-adapting noise floor learned from low-energy chunks. The goal is not
    studio-quality denoising; it is to improve VAD stability and reduce obvious
    stationary noise before ASR.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        enabled: bool = False,
        strength: float = 1.1,
        noise_update_rms_threshold: float = 0.02,
        noise_ema: float = 0.92,
        floor_ratio: float = 0.18,
        nperseg: int = 400,
        noverlap: int = 240,
    ) -> None:
        self._sample_rate = sample_rate
        self._enabled = enabled
        self._strength = strength
        self._noise_update_rms_threshold = noise_update_rms_threshold
        self._noise_ema = noise_ema
        self._floor_ratio = floor_ratio
        self._nperseg = nperseg
        self._noverlap = noverlap
        self._noise_mag: np.ndarray | None = None

    def process(self, chunk: AudioChunk) -> AudioChunk:
        if not self._enabled or chunk.samples.size < self._nperseg:
            return chunk

        samples = chunk.samples.astype(np.float32, copy=False)
        chunk_rms = float(np.sqrt(np.mean(samples ** 2)))
        try:
            from scipy.signal import istft, stft

            _, _, spectrum = stft(
                samples,
                fs=self._sample_rate,
                window="hann",
                nperseg=self._nperseg,
                noverlap=self._noverlap,
                boundary="zeros",
                padded=True,
            )
            if spectrum.size == 0:
                return chunk

            magnitude = np.abs(spectrum)
            phase = np.exp(1j * np.angle(spectrum))
            current_noise = np.quantile(magnitude, 0.2, axis=1)

            if self._noise_mag is None:
                self._noise_mag = current_noise
            elif chunk_rms <= self._noise_update_rms_threshold:
                self._noise_mag = (
                    self._noise_ema * self._noise_mag
                    + (1.0 - self._noise_ema) * current_noise
                )

            noise_mag = self._noise_mag
            if noise_mag is None:
                return chunk

            noise_mag = np.maximum(noise_mag, 1e-6)
            gain = np.clip(
                (magnitude - self._strength * noise_mag[:, None]) / (magnitude + 1e-6),
                self._floor_ratio,
                1.0,
            )
            reduced_mag = magnitude * gain
            filtered = reduced_mag * phase
            _, enhanced = istft(
                filtered,
                fs=self._sample_rate,
                window="hann",
                nperseg=self._nperseg,
                noverlap=self._noverlap,
                input_onesided=True,
            )
            if enhanced.size == 0:
                return chunk
            enhanced = enhanced[: len(samples)]
            if enhanced.size < len(samples):
                enhanced = np.pad(enhanced, (0, len(samples) - enhanced.size))
            enhanced = np.clip(enhanced.astype(np.float32, copy=False), -1.0, 1.0)
            return AudioChunk(
                samples=enhanced,
                sample_rate=chunk.sample_rate,
                start_time=chunk.start_time,
            )
        except Exception as exc:
            logger.debug("noise reduction skipped for chunk: %s", exc)
            return chunk


class AudioConditioner:
    """Small ASR-facing level conditioner.

    The intent is to remove DC offset, gently lift quiet valid speech, and keep
    peaks below clipping before speech detection or Whisper sees the buffer.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        target_rms: float = 0.035,
        min_rms_for_gain: float = 0.003,
        max_gain: float = 2.5,
        peak_limit: float = 0.98,
        remove_dc: bool = True,
    ) -> None:
        self._enabled = enabled
        self._target_rms = max(0.0, target_rms)
        self._min_rms_for_gain = max(0.0, min_rms_for_gain)
        self._max_gain = max(1.0, max_gain)
        self._peak_limit = min(1.0, max(0.1, peak_limit))
        self._remove_dc = remove_dc

    def process(self, chunk: AudioChunk) -> AudioChunk:
        processed = self.process_samples(chunk.samples)
        if processed is chunk.samples:
            return chunk
        return AudioChunk(
            samples=processed,
            sample_rate=chunk.sample_rate,
            start_time=chunk.start_time,
        )

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        if not self._enabled or samples.size == 0:
            return samples

        out = samples.astype(np.float32, copy=False)
        if self._remove_dc:
            # Capture devices can have a small DC bias; Whisper and RMS gates do
            # better when the waveform is centered around zero.
            out = out - np.float32(np.mean(out))

        rms = float(np.sqrt(np.mean(out ** 2)))
        if self._target_rms > 0.0 and rms >= self._min_rms_for_gain and rms < self._target_rms:
            gain = min(self._max_gain, self._target_rms / max(rms, 1e-8))
            out = out * np.float32(gain)

        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > self._peak_limit:
            out = out * np.float32(self._peak_limit / max(peak, 1e-8))

        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)
