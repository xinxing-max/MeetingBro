from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence

from ..schemas import LanguageCode, TranscriptSegment

SummaryKind = Literal[
    "meeting_memory",
    "rolling_summary",
    "cumulative_meeting_summary",
    "final_summary",
    "refined_transcript",
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
        vocabulary: Optional[str] = None,
    ) -> str:
        """Produce a summary string for the given transcript segments.

        ``kind`` lets the implementation specialize wording/length for rolling vs
        cumulative vs final summaries. ``previous_summary`` is optional and, if
        supplied (typically for cumulative summaries), implementations may use
        it to keep prompt size bounded.
        """

    def finalize_meeting(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        language: LanguageCode,
        vocabulary: Optional[str] = None,
        meeting_memory: Optional[str] = None,
    ) -> dict:
        """Returns {'chapters': [...], 'action_items': [...], 'final_summary': str}."""
        del meeting_memory
        return {
            "chapters": [],
            "action_items": [],
            "final_summary": self.summarize(segments, kind="final_summary", language=language, vocabulary=vocabulary),
        }
