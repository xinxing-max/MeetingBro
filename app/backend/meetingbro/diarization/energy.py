"""Energy-based heuristic speaker diarizer.

Uses short-term energy and zero-crossing rate to detect speaker changes.
This is an approximate heuristic that works reasonably for 2–3 speakers
in a meeting/classroom setting. Speaker labels are inferred, not verified.

No external model or API is needed — this runs entirely with numpy.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import Diarizer, DiarizationSegment

logger = logging.getLogger(__name__)

# Minimum silence gap (seconds) to consider a potential speaker change.
_SILENCE_GAP_SECONDS = 0.8
# RMS threshold below which audio is considered silence.
_SILENCE_RMS_THRESHOLD = 0.01
# Frame length for energy analysis (seconds).
_FRAME_SECONDS = 0.03
# Minimum segment duration to emit (seconds).
_MIN_SEGMENT_SECONDS = 0.3


class EnergyDiarizer(Diarizer):
    """Heuristic diarizer based on energy envelope and silence gaps.

    Detects speaker turns by finding silence gaps in the audio. Each
    contiguous speech region separated by a silence gap is assigned to a
    speaker. Speaker identity is tracked across chunks using a simple
    spectral centroid signature.

    Parameters
    ----------
    max_speakers
        Maximum number of distinct speakers to track.
    silence_gap_seconds
        Minimum silence duration to trigger a potential speaker change.
    silence_rms_threshold
        RMS amplitude below which a frame is considered silent.
    """

    def __init__(
        self,
        *,
        max_speakers: int = 6,
        silence_gap_seconds: float = _SILENCE_GAP_SECONDS,
        silence_rms_threshold: float = _SILENCE_RMS_THRESHOLD,
    ) -> None:
        self._max_speakers = max_speakers
        self._silence_gap = silence_gap_seconds
        self._silence_rms = silence_rms_threshold
        # Accumulated speaker signatures: centroid mean per speaker.
        self._speaker_centroids: dict[str, float] = {}
        self._next_speaker_idx = 1

    def reset(self) -> None:
        self._speaker_centroids.clear()
        self._next_speaker_idx = 1

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        offset_seconds: float = 0.0,
    ) -> list[DiarizationSegment]:
        if samples.size == 0:
            return []

        frame_len = max(1, int(sample_rate * _FRAME_SECONDS))
        n_frames = len(samples) // frame_len
        if n_frames == 0:
            return []

        # Compute per-frame RMS energy.
        frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Adaptive threshold: use P10 of frame RMS as noise floor estimate so
        # the diarizer works across both high-level loopback and low-level far-field
        # mic scenarios.  The constructor parameter acts as a lower bound.
        noise_floor = float(np.percentile(rms, 10))
        adaptive_threshold = max(self._silence_rms, noise_floor * 2.5)

        is_speech = rms > adaptive_threshold

        # Find contiguous speech regions separated by silence gaps.
        regions = self._find_speech_regions(is_speech, frame_len, sample_rate)

        if not regions:
            return []

        # Assign speaker labels based on spectral characteristics.
        result: list[DiarizationSegment] = []
        for start_sec, end_sec in regions:
            if (end_sec - start_sec) < _MIN_SEGMENT_SECONDS:
                continue
            start_sample = int(start_sec * sample_rate)
            end_sample = min(int(end_sec * sample_rate), len(samples))
            region_audio = samples[start_sample:end_sample]

            centroid = self._spectral_centroid(region_audio, sample_rate)
            speaker = self._match_or_create_speaker(centroid)

            result.append(DiarizationSegment(
                start_time=offset_seconds + start_sec,
                end_time=offset_seconds + end_sec,
                speaker_label=speaker,
                confidence=0.5,  # heuristic confidence — explicitly approximate
            ))

        return result

    def _find_speech_regions(
        self,
        is_speech: np.ndarray,
        frame_len: int,
        sample_rate: int,
    ) -> list[tuple[float, float]]:
        """Find contiguous speech regions, merging gaps shorter than the threshold."""
        regions: list[tuple[float, float]] = []
        in_speech = False
        start_frame = 0
        silence_frames = 0
        gap_frames = int(self._silence_gap * sample_rate / frame_len)

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    start_frame = i
                    in_speech = True
                silence_frames = 0
            else:
                if in_speech:
                    silence_frames += 1
                    if silence_frames >= gap_frames:
                        end_frame = i - silence_frames + 1
                        start_sec = start_frame * frame_len / sample_rate
                        end_sec = end_frame * frame_len / sample_rate
                        if end_sec > start_sec:
                            regions.append((start_sec, end_sec))
                        in_speech = False
                        silence_frames = 0

        # Close any open region.
        if in_speech:
            end_frame = len(is_speech)
            start_sec = start_frame * frame_len / sample_rate
            end_sec = end_frame * frame_len / sample_rate
            if end_sec > start_sec:
                regions.append((start_sec, end_sec))

        return regions

    def _spectral_centroid(self, samples: np.ndarray, sample_rate: int) -> float:
        """Compute the spectral centroid of an audio segment."""
        if len(samples) < 256:
            return 0.0
        windowed = samples * np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
        total = spectrum.sum()
        if total < 1e-10:
            return 0.0
        return float(np.sum(freqs * spectrum) / total)

    def _match_or_create_speaker(self, centroid: float) -> str:
        """Match a spectral centroid to an existing speaker or create a new one."""
        if not self._speaker_centroids:
            label = f"Speaker {self._next_speaker_idx}"
            self._speaker_centroids[label] = centroid
            self._next_speaker_idx += 1
            return label

        # Find closest existing speaker by centroid distance.
        best_label = ""
        best_dist = float("inf")
        for label, ref_centroid in self._speaker_centroids.items():
            dist = abs(centroid - ref_centroid)
            if dist < best_dist:
                best_dist = dist
                best_label = label

        # If close enough, match to existing speaker. Otherwise, create new.
        # Threshold: 15% of the centroid value — tuned for speech.
        threshold = max(200.0, abs(centroid) * 0.15)
        if best_dist < threshold and best_label:
            # Update running average of centroid for this speaker.
            self._speaker_centroids[best_label] = (
                0.8 * self._speaker_centroids[best_label] + 0.2 * centroid
            )
            return best_label

        if len(self._speaker_centroids) >= self._max_speakers:
            # At speaker limit — assign to closest.
            if best_label:
                self._speaker_centroids[best_label] = (
                    0.9 * self._speaker_centroids[best_label] + 0.1 * centroid
                )
                return best_label

        label = f"Speaker {self._next_speaker_idx}"
        self._speaker_centroids[label] = centroid
        self._next_speaker_idx += 1
        return label
