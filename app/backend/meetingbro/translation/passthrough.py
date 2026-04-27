from __future__ import annotations

from ..schemas import LanguageCode
from .base import Translator


class PassthroughTranslator(Translator):
    """No-op translator for Phase 2.

    Keeps the interface, data model, and UI language selector in place so that a
    real translation provider can drop in without pipeline changes. When the
    source language already equals the target, the text is returned unchanged.
    Otherwise, the text is returned unchanged with a ``[untranslated:<src>→<tgt>]``
    marker so that the UI and exporter can show that translation is pending.
    """

    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: LanguageCode,
    ) -> str:
        if not text:
            return text
        if source_language == target_language:
            return text
        return f"[untranslated:{source_language}→{target_language}] {text}"
