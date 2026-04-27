from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

from ..schemas import LanguageCode, TranscriptSegment

SummaryKind = Literal[
    "rolling_summary",
    "cumulative_meeting_summary",
    "final_summary",
]


class Summarizer(ABC):
    @abstractmethod
    def summarize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        kind: SummaryKind,
        language: LanguageCode,
        previous_summary: str | None = None,
    ) -> str:
        """Produce a summary string for the given transcript segments.

        ``kind`` lets the implementation specialize wording/length for rolling vs
        cumulative vs final summaries. ``previous_summary`` is optional and, if
        supplied (typically for cumulative summaries), implementations may use
        it to keep prompt size bounded.
        """
