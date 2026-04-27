from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DiarizationSegment:
    """A speaker-attributed time span within an audio chunk."""

    start_time: float  # seconds relative to chunk start
    end_time: float
    speaker_label: str  # e.g. "Speaker 1", "Speaker 2"
    confidence: float  # [0, 1]


class Diarizer(ABC):
    """Abstract speaker diarizer.

    Implementations take raw audio and return speaker segments. Speaker labels
    are explicitly approximate — inferred, not verified identities.
    """

    @abstractmethod
    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        offset_seconds: float = 0.0,
    ) -> list[DiarizationSegment]:
        """Assign speaker labels to time spans within the audio.

        Parameters
        ----------
        samples
            Mono float32 audio, shape ``(N,)``.
        sample_rate
            Sample rate in Hz.
        offset_seconds
            Session-time offset for the start of this audio buffer.

        Returns
        -------
        list[DiarizationSegment]
            Speaker-labeled time spans. May overlap with ASR segment boundaries.
        """
        ...

    def reset(self) -> None:
        """Reset any accumulated state (e.g., between sessions)."""
        pass
