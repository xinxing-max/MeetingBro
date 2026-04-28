from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from ..schemas import LanguageCode, TranscriptSegment
from .base import Summarizer, SummaryKind


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u4E00-\u9FFF]+")
_STOPWORDS = {
    "en": {
        "the","a","an","and","or","but","is","are","was","were","be","been","being",
        "to","of","in","on","for","with","at","by","it","this","that","these","those",
        "i","you","he","she","we","they","me","him","her","us","them","my","your","our","their",
        "so","as","if","then","than","not","no","yes","do","does","did","have","has","had",
        "can","could","would","should","may","might","will","shall","just","also",
    },
    "de": {
        "der","die","das","und","oder","aber","ist","sind","war","waren","sei","sein",
        "zu","von","in","auf","mit","bei","für","im","am","ein","eine","einen","einer","eines",
        "nicht","auch","nur","schon","noch","mehr","als","wie","wenn","dann",
        "ich","du","er","sie","es","wir","ihr","mein","dein","unser","euer",
    },
    "zh": set(),  # CJK stopword filtering is out of scope for MVP heuristic
}


def _content_words(text: str, language: LanguageCode) -> list[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    stop = _STOPWORDS.get(language, set())
    return [t for t in tokens if t not in stop and len(t) > 1]


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts


def _score_sentence(sentence: str, keyword_weights: dict[str, float], language: LanguageCode) -> float:
    words = _content_words(sentence, language)
    if not words:
        return 0.0
    return sum(keyword_weights.get(w, 0.0) for w in words) / (1.0 + 0.02 * len(words))


def _extractive(text: str, language: LanguageCode, max_sentences: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    word_counts = Counter()
    for s in sentences:
        word_counts.update(_content_words(s, language))
    if not word_counts:
        return " ".join(sentences[:max_sentences])

    top_n = max(5, max_sentences * 4)
    top_keywords = dict(word_counts.most_common(top_n))

    scored = [
        (i, s, _score_sentence(s, top_keywords, language))
        for i, s in enumerate(sentences)
    ]
    chosen = sorted(scored, key=lambda t: t[2], reverse=True)[:max_sentences]
    chosen.sort(key=lambda t: t[0])
    return " ".join(s for _, s, _ in chosen)


def _keywords(text: str, language: LanguageCode, k: int = 6) -> list[str]:
    counts = Counter(_content_words(text, language))
    return [w for w, _ in counts.most_common(k)]


def _bullet_lines(items: list[str], *, fallback: str = "None yet") -> list[str]:
    if not items:
        return [f"- {fallback}"]
    return [f"- {item}" for item in items if item]


class HeuristicSummarizer(Summarizer):
    """Transcript-driven extractive summarizer.

    Quality is bounded by its extractive nature. It is NOT demo content — every
    word in the output is derived from the transcript segments passed in. The
    ``LLMSummarizer`` drop-in replaces this with a provider-backed
    summarizer when API keys are available.
    """

    def __init__(
        self,
        *,
        max_rolling_sentences: int = 3,
        max_cumulative_sentences: int = 5,
        max_final_sentences: int = 10,
    ) -> None:
        self._rolling = max_rolling_sentences
        self._cumulative = max_cumulative_sentences
        self._final = max_final_sentences

    def summarize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        kind: SummaryKind,
        language: LanguageCode,
        previous_summary: str | None = None,
        vocabulary: str | None = None,
    ) -> str:
        if not segments:
            return ""

        transcript_text = " ".join(s.text.strip() for s in segments if s.text.strip())
        if not transcript_text:
            return ""

        if kind == "rolling_summary":
            max_s = self._rolling
            label = "Recent discussion"
        elif kind == "meeting_memory":
            max_s = self._cumulative
            label = "## Topics"
        elif kind == "cumulative_meeting_summary":
            max_s = self._cumulative
            label = "Meeting so far"
        else:
            max_s = self._final
            label = "Meeting recap"

        # Guess a language for stopword handling. We use the majority-vote from
        # segments rather than the requested output language, because the
        # heuristic operates on the original transcript text.
        lang_counts = Counter(
            s.original_language for s in segments if s.original_language in _STOPWORDS
        )
        heur_lang: LanguageCode = (
            lang_counts.most_common(1)[0][0] if lang_counts else language
        )  # type: ignore[assignment]

        body = _extractive(transcript_text, heur_lang, max_s)
        keywords = _keywords(transcript_text, heur_lang, k=6)
        keyword_line = ", ".join(keywords) if keywords else ""

        if kind == "meeting_memory":
            parts = [
                f"## Topics\n- {body}" if body else "## Topics",
                "## Decisions",
                "## Action Items",
                "## Open Questions",
                "## Important Facts",
            ]
        elif kind == "cumulative_meeting_summary":
            state_lines = _bullet_lines([body] if body else [])
            decision_lines = _bullet_lines([])
            action_lines = _bullet_lines([f"Key terms: {keyword_line}"] if keyword_line else [])
            open_question_lines = _bullet_lines([])
            parts = [
                "## Meeting State",
                *state_lines,
                "",
                "## Decisions",
                *decision_lines,
                "",
                "## Action Items",
                *action_lines,
                "",
                "## Open Questions",
                *open_question_lines,
            ]
        else:
            parts = [f"{label}: {body}" if body else label]
        if keyword_line:
            if kind == "meeting_memory":
                parts.append(f"- Key terms: {keyword_line}")
            elif kind not in {"cumulative_meeting_summary"}:
                parts.append(f"Key terms: {keyword_line}")
        if kind in {"cumulative_meeting_summary", "final_summary"} and previous_summary:
            parts.append(f"(Compressed meeting memory: {previous_summary[:240]})")
        return "\n".join(parts)

    def finalize_meeting(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        language: LanguageCode,
        vocabulary: str | None = None,
        meeting_memory: str | None = None,
    ) -> dict:
        del meeting_memory
        return {
            "chapters": [],
            "action_items": [],
            "final_summary": self.summarize(segments, kind="final_summary", language=language, vocabulary=vocabulary),
        }
