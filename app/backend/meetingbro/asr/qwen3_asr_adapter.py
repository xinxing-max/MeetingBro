"""Optional Qwen3-ASR preview backend adapter.

IMPORTANT: This is a batch / near-streaming preview path, NOT true token-level
streaming ASR.  Each ``transcribe()`` call decodes a complete audio window
offline through ``sherpa_onnx.OfflineRecognizer.from_qwen3_asr`` and returns at
most one ``ASRSegment``, exactly as ``FasterWhisperAdapter`` does.

sherpa-onnx is an optional dependency.  A clear ``RuntimeError`` is raised only
when this adapter is *constructed* (i.e. when the qwen3 backend is selected via
env), so the normal faster-whisper startup path is completely unaffected.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from ..schemas import OriginalLanguage
from .base import ASRAdapter, ASRSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language support
# ---------------------------------------------------------------------------

_SUPPORTED_LANGUAGES: tuple[OriginalLanguage, ...] = ("zh", "en", "de")


def _normalize_language(code: Optional[str]) -> OriginalLanguage:
    if not code:
        return "unknown"
    c = code.lower()
    if c in _SUPPORTED_LANGUAGES:
        return c  # type: ignore[return-value]
    return "unknown"


# ---------------------------------------------------------------------------
# Filler suppression
# ---------------------------------------------------------------------------

# Full-text exact match after strip + lower (lowercasing CJK is a no-op).
_FILLER_SET: frozenset[str] = frozenset(
    {
        # Mandarin vocalisations
        "嗯",
        "嗯。",
        "啊",
        "啊。",
        # English
        "um",
        "uh",
        "ah",
    }
)

# ---------------------------------------------------------------------------
# Script filter
# ---------------------------------------------------------------------------

# Script character-class patterns.  We count printable (non-whitespace) chars.
_CJK_RE = re.compile(
    # CJK Unified Ideographs main block + Extension-A + Compatibility ideographs
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]"
)
_ARABIC_RE = re.compile(
    # Arabic and Arabic Supplement / Presentation Forms
    r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]"
)
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff\u0500-\u052f]")
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")

_DOMINANT_THRESHOLD = 0.5  # fraction required to declare a script "dominant"


def _script_ratios(text: str) -> dict[str, float]:
    """Return per-script fractions of non-whitespace characters in *text*."""
    chars = [c for c in text if not c.isspace()]
    n = len(chars)
    if n == 0:
        return {"cjk": 0.0, "arabic": 0.0, "cyrillic": 0.0, "latin": 0.0}
    return {
        "cjk": sum(1 for c in chars if _CJK_RE.match(c)) / n,
        "arabic": sum(1 for c in chars if _ARABIC_RE.match(c)) / n,
        "cyrillic": sum(1 for c in chars if _CYRILLIC_RE.match(c)) / n,
        "latin": sum(1 for c in chars if _LATIN_RE.match(c)) / n,
    }


def _wrong_script(text: str, forced_language: Optional[str]) -> bool:
    """Return ``True`` when *text*'s dominant script conflicts with *forced_language*.

    Rules:
    - ``en`` / ``de``:
        Reject if CJK, Arabic, or Cyrillic is each individually dominant
        (> 50 % of printable chars).
    - ``zh``:
        Reject if Arabic or Cyrillic is dominant.
        Reject if there are *no* CJK characters at all (pure-Latin output is
        almost certainly a hallucination for a zh session).
    - ``None`` / other:
        No strict filtering applied.
    """
    if forced_language not in ("en", "de", "zh"):
        return False

    ratios = _script_ratios(text)

    if forced_language in ("en", "de"):
        return (
            ratios["cjk"] > _DOMINANT_THRESHOLD
            or ratios["arabic"] > _DOMINANT_THRESHOLD
            or ratios["cyrillic"] > _DOMINANT_THRESHOLD
        )

    # forced_language == "zh"
    if ratios["arabic"] > _DOMINANT_THRESHOLD or ratios["cyrillic"] > _DOMINANT_THRESHOLD:
        return True
    # No CJK at all → pure-Latin (or other) hallucination.
    return ratios["cjk"] == 0.0


# ---------------------------------------------------------------------------
# Incomplete-preview filter
# ---------------------------------------------------------------------------

# Patterns that indicate the model produced a trailing-ellipsis fragment that
# looks like mid-thought output rather than a complete utterance.
_TRAILING_ELLIPSIS_RE = re.compile(
    r"(?:\.{2,}|\u2026|\u3002{2,}|\u2014|\u2013|-)\s*$",
    re.UNICODE,
)

# Mostly-punctuation guard: if > 60 % of all characters are punctuation or
# whitespace the whole output is treated as noise.
_PUNCT_OR_SPACE_RE = re.compile(r"[\s\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e\u3000-\u303f\uff00-\uffef]", re.UNICODE)


def _looks_incomplete_preview(text: str) -> bool:
    """Return ``True`` when *text* looks like a trailing-ellipsis or otherwise
    incomplete fragment that should not be shown in the preview UI.

    Checks:
    1. Text ends with ASCII ellipsis ``...``, Unicode ellipsis ``\u2026``,
       CJK triple-period ``\u3002\u3002\u3002``, or a dangling dash
       (``-``, ``\u2014``, ``\u2013``).
    2. More than 60 % of all characters are punctuation/whitespace (pure noise).
    """
    if not text:
        return False
    if _TRAILING_ELLIPSIS_RE.search(text):
        return True
    total = len(text)
    punct_count = sum(1 for c in text if _PUNCT_OR_SPACE_RE.match(c))
    if total > 0 and punct_count / total > 0.60:
        return True
    return False


# ---------------------------------------------------------------------------
# Output quality heuristics
# ---------------------------------------------------------------------------

# Repeated word/phrase pattern: the same token repeated ≥ 3 times in a row.
# Covers "hello hello hello" and CJK repetition like "你好你好你好".
_REPEATED_WORD_RE = re.compile(
    r"(?:^|(?<=\s))(\S{2,})(?:\s+\1){2,}",
    re.IGNORECASE | re.UNICODE,
)


def _is_garbage(text: str) -> bool:
    """Return ``True`` when *text* looks like ASR noise rather than real speech.

    Checks:
    1. Single-character non-CJK output (likely a stray token, not speech).
    2. A single character repeated throughout (e.g. "aaaaaaa" or "......").
    3. A word or short phrase repeated 3+ consecutive times in a row.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True

    # Single non-CJK character → almost certainly noise.
    if len(chars) == 1 and not _CJK_RE.match(chars[0]):
        return True

    # One character dominates ≥ 80 % of all printable chars.
    if len(chars) >= 4:
        most_common_count = Counter(chars).most_common(1)[0][1]
        if most_common_count / len(chars) >= 0.80:
            return True

    # Repeated word / phrase.
    if _REPEATED_WORD_RE.search(text):
        return True

    return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class Qwen3ASRAdapter(ASRAdapter):
    """Preview ASR backed by sherpa-onnx Qwen3-ASR 0.6B.

    Implements the standard ``ASRAdapter`` interface so it can be dropped in as
    MeetingBro's dedicated preview ASR backend.  The ``OfflineRecognizer`` is
    created lazily on the first ``transcribe()`` call so that construction is
    fast and the startup log message appears close to the first real decode.
    """

    def __init__(
        self,
        *,
        model_dir: str | Path,
        num_threads: int = 2,
        provider: str = "cpu",
        max_total_len: int = 512,
        max_new_tokens: int = 256,
        filter_language_script: bool = True,
        suppress_fillers: bool = True,
    ) -> None:
        # Fail fast at construction time so the error surfaces before the
        # server is fully up, not on the first preview decode.
        try:
            import sherpa_onnx as _so  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MEETINGBRO_PREVIEW_ASR_BACKEND=qwen3 requires sherpa-onnx. "
                "Install it with: pip install sherpa-onnx"
            ) from exc

        model_dir = Path(model_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise RuntimeError(
                f"Qwen3-ASR model directory not found: {model_dir}. "
                "Set MEETINGBRO_PREVIEW_QWEN3_MODEL_DIR to a valid "
                "sherpa-onnx Qwen3-ASR export directory."
            )

        conv_frontend = model_dir / "conv_frontend.onnx"
        if not conv_frontend.is_file():
            raise RuntimeError(f"Missing conv_frontend.onnx under {model_dir}")

        tokenizer_dir = model_dir / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise RuntimeError(f"Missing tokenizer/ directory under {model_dir}")

        def _pick(candidates: list[Path], label: str) -> Path:
            for cand in candidates:
                if cand.is_file():
                    return cand
            raise RuntimeError(
                f"Could not find {label} under {model_dir}. Tried: "
                + ", ".join(p.name for p in candidates)
            )

        encoder = _pick(
            [model_dir / "encoder.int8.onnx", model_dir / "encoder.onnx"],
            "encoder",
        )
        decoder = _pick(
            [model_dir / "decoder.int8.onnx", model_dir / "decoder.onnx"],
            "decoder",
        )

        self._conv_frontend = str(conv_frontend)
        self._encoder = str(encoder)
        self._decoder = str(decoder)
        self._tokenizer = str(tokenizer_dir)
        self._num_threads = max(1, num_threads)
        self._provider = provider
        self._max_total_len = max_total_len
        self._max_new_tokens = max_new_tokens
        self._filter_language_script = filter_language_script
        self._suppress_fillers = suppress_fillers
        self._recognizer = None  # lazy – created on first transcribe call

        logger.info(
            "Qwen3ASRAdapter configured model_dir=%s num_threads=%d provider=%s "
            "max_total_len=%d max_new_tokens=%d filter_script=%s suppress_fillers=%s",
            model_dir,
            self._num_threads,
            self._provider,
            self._max_total_len,
            self._max_new_tokens,
            self._filter_language_script,
            self._suppress_fillers,
        )

    # ------------------------------------------------------------------

    def _ensure_recognizer(self):
        if self._recognizer is None:
            import sherpa_onnx

            logger.info(
                "loading Qwen3-ASR recognizer encoder=%s decoder=%s "
                "provider=%s threads=%d",
                self._encoder,
                self._decoder,
                self._provider,
                self._num_threads,
            )
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=self._conv_frontend,
                encoder=self._encoder,
                decoder=self._decoder,
                tokenizer=self._tokenizer,
                num_threads=self._num_threads,
                sample_rate=16_000,
                feature_dim=128,
                decoding_method="greedy_search",
                debug=False,
                provider=self._provider,
                max_total_len=self._max_total_len,
                max_new_tokens=self._max_new_tokens,
            )
        return self._recognizer

    # ------------------------------------------------------------------

    def prewarm(self) -> None:
        """Load the Qwen3 recognizer without running a decode.

        This is intended for a background startup task so the first visible
        preview window does not pay the model construction cost.
        """
        self._ensure_recognizer()

    # ------------------------------------------------------------------

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        forced_language: Optional[str] = None,
        offset_seconds: float = 0.0,
        initial_prompt: Optional[str] = None,
        quality_preset: str = "realtime",
    ) -> list[ASRSegment]:
        if samples.size == 0:
            return []

        if sample_rate != 16_000:
            raise ValueError(
                f"Qwen3ASRAdapter expects 16 kHz mono input, got {sample_rate} Hz. "
                "MeetingBro's standard audio pipeline always outputs 16 kHz."
            )

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        duration = float(len(samples)) / sample_rate
        recognizer = self._ensure_recognizer()

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)

        result = stream.result
        if hasattr(result, "text"):
            text = (str(getattr(result, "text", "")) or "").strip()
        elif isinstance(result, dict):
            text = str(result.get("text", "")).strip()
        else:
            text = str(result).strip()

        if not text:
            return []

        # Filler suppression: exact full-text match, case-insensitive for Latin.
        if self._suppress_fillers and text.lower() in _FILLER_SET:
            logger.debug("Qwen3ASRAdapter: suppressed filler %r", text)
            return []

        # Garbage / noise detection.
        if _is_garbage(text):
            logger.debug("Qwen3ASRAdapter: garbage-filter dropped %r", text)
            return []

        # Wrong-script filter.
        if self._filter_language_script and _wrong_script(text, forced_language):
            logger.debug(
                "Qwen3ASRAdapter: wrong-script filter dropped %r (forced_language=%s)",
                text,
                forced_language,
            )
            return []

        lang = _normalize_language(forced_language)

        # Conservative confidence placeholder.
        # Qwen3-ASR does not expose per-segment log-probabilities through the
        # sherpa-onnx offline API, so we use a fixed conservative value.
        # Using 0.60 (below the silence_commit_min_confidence default of 0.75)
        # keeps preview as preview and prevents accidental silence-commit
        # promotion to a formal transcript segment.
        confidence = 0.60

        return [
            ASRSegment(
                start_time=offset_seconds,
                end_time=offset_seconds + duration,
                text=text,
                language=lang,
                confidence=confidence,
            )
        ]

