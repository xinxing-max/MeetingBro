from __future__ import annotations

import logging
import os
from typing import Sequence

from ..llm import OpenAICompatibleClient
from ..schemas import LanguageCode, TranscriptSegment
from .base import Summarizer, SummaryKind
from .heuristic import HeuristicSummarizer

logger = logging.getLogger(__name__)


_PROMPTS = {
    "meeting_memory": (
        "You maintain a compressed, structured memory of an ongoing meeting in {language}. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and do not ignore non-English text. "
        "Update the previous memory using only the new transcript. Keep it concise and "
        "bounded. Preserve durable facts; remove resolved or redundant details. Output "
        "Markdown with these exact headings: ## Topics, ## Decisions, ## Action Items, "
        "## Open Questions, ## Important Facts. Do not invent content."
    ),
    "rolling_summary": (
        "You are summarizing the last few minutes of a live meeting transcript. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and summarize in {language}. "
        "Write a concise, factual recap in 2-4 sentences in {language}. "
        "Cover what was just discussed; do not invent content."
    ),
    "cumulative_meeting_summary": (
        "You are maintaining a live cumulative summary of an ongoing meeting. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and summarize in {language}. "
        "Use the compressed meeting memory as the durable context and the recent "
        "transcript as fresh evidence. Write a compact update in {language} covering "
        "topics, decisions, action items, and open questions so far. 3-6 sentences. "
        "Do not invent content."
    ),
    "final_summary": (
        "You are writing the final recap of a meeting. Use the compressed meeting "
        "memory as the durable context and the recent transcript as fresh evidence. "
        "The transcript may mix Chinese, English, and German; account for content in all "
        "languages and summarize in {language}. "
        "In {language}, produce: key points, decisions, action items, open questions. "
        "Be faithful to the provided context; do not invent content."
    ),
}

_LANG_NAMES = {"en": "English", "zh": "Chinese", "de": "German"}


class LLMSummarizer(Summarizer):
    """LLM-backed summarizer.

    Activated when a supported API key is present in the environment. For
    LongCat, set ``LONGCAT_API_KEY``; the default endpoint/model are:
    ``https://api.longcat.chat/openai`` and ``LongCat-Flash-Chat``.

    Falls back to the heuristic summarizer otherwise, so the pipeline always
    produces transcript-driven output.
    """

    def __init__(self, *, fallback: Summarizer | None = None) -> None:
        self._fallback = fallback or HeuristicSummarizer()
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self._openai_key = os.environ.get("OPENAI_API_KEY")
        self._anthropic_client = None
        self._openai_client = None
        self._compatible_client = OpenAICompatibleClient.from_env()

    def _ensure_client(self):
        if self._compatible_client is not None:
            return "openai_compatible"
        if self._anthropic_key and self._anthropic_client is None:
            try:
                import anthropic  # type: ignore

                self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_key)
                return "anthropic"
            except Exception as exc:
                logger.warning("anthropic client unavailable: %s", exc)
        if self._openai_key and self._openai_client is None:
            try:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=self._openai_key)
                return "openai"
            except Exception as exc:
                logger.warning("openai client unavailable: %s", exc)
        if self._anthropic_client is not None:
            return "anthropic"
        if self._openai_client is not None:
            return "openai"
        return None

    def summarize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        kind: SummaryKind,
        language: LanguageCode,
        previous_summary: str | None = None,
    ) -> str:
        if not segments:
            return ""
        provider = self._ensure_client()
        if provider is None:
            return self._fallback.summarize(
                segments,
                kind=kind,
                language=language,
                previous_summary=previous_summary,
            )

        transcript_text = "\n".join(
            f"[{s.start_time:.1f}-{s.end_time:.1f}] {s.text}" for s in segments
        )
        system_prompt = _PROMPTS[kind].format(language=_LANG_NAMES.get(language, language))
        user_content = transcript_text
        if previous_summary:
            previous_label = (
                "Previous meeting memory"
                if kind == "meeting_memory"
                else "Compressed meeting memory / prior summary"
            )
            new_label = (
                "New transcript to fold into memory"
                if kind == "meeting_memory"
                else "Recent transcript tail"
            )
            user_content = f"{previous_label}:\n{previous_summary}\n\n{new_label}:\n{transcript_text}"

        try:
            if provider == "openai_compatible":
                assert self._compatible_client is not None
                return self._compatible_client.chat(
                    system=system_prompt,
                    user=user_content,
                    max_tokens=900 if kind == "meeting_memory" else 600,
                    temperature=0.2,
                )
            elif provider == "anthropic":
                assert self._anthropic_client is not None
                msg = self._anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=900 if kind == "meeting_memory" else 600,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
                return "".join(
                    b.text for b in msg.content if getattr(b, "type", "") == "text"
                ).strip()
            else:
                assert self._openai_client is not None
                resp = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("LLM summarization failed, falling back: %s", exc)
            return self._fallback.summarize(
                segments,
                kind=kind,
                language=language,
                previous_summary=previous_summary,
            )
