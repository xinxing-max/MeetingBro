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