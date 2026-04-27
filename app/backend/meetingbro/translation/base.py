from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import LanguageCode


class Translator(ABC):
    """Translates text among Chinese, English, and German.

    The data model and UI treat translations as derived outputs; the original
    transcript and its original-language summaries remain the source of truth.
    """

    @abstractmethod
    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: LanguageCode,
    ) -> str:
        ...
