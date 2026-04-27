from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ..schemas import OriginalLanguage
from .base import ASRAdapter, ASRSegment

logger = logging.getLogger(__name__)

_SUPPORTED: tuple[OriginalLanguage, ...] = ("zh", "en", "de")


def _normalize_language(code: Optional[str]) -> OriginalLanguage:
    if not code:
        return "unknown"
    code = code.lower()
    if code in _SUPPORTED:
        return code  # type: ignore[return-value]
    return "unknown"


class FasterWhisperAdapter(ASRAdapter):
    """ASR via faster-whisper.

    Auto-detect is used when ``forced_language`` is ``None``. The first
    targeted first-class languages are Chinese, English, and German, matching
    the product requirements.
    """

    def __init__(
        self,
        *,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "loading faster-whisper model size=%s device=%s compute=%s",
                self._model_size, self._device, self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        forced_language: Optional[str] = None,
        offset_seconds: float = 0.0,
    ) -> list[ASRSegment]:
        if samples.size == 0:
            return []
        if sample_rate != 16_000:
            raise ValueError(
                f"FasterWhisperAdapter expects 16 kHz input, got {sample_rate}"
            )
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        model = self._ensure_model()
        segments_iter, info = model.transcribe(
            samples,
            language=forced_language,
            vad_filter=True,
            beam_size=1,
            word_timestamps=False,
        )
        detected_language = forced_language or info.language
        lang = _normalize_language(detected_language)

        out: list[ASRSegment] = []
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            # avg_logprob is log-probability; convert to a rough [0,1] confidence.
            conf = 1.0 / (1.0 + math.exp(-getattr(s, "avg_logprob", 0.0)))
            out.append(
                ASRSegment(
                    start_time=offset_seconds + float(s.start),
                    end_time=offset_seconds + float(s.end),
                    text=text,
                    language=lang,
                    confidence=max(0.0, min(1.0, conf)),
                )
            )
        return out
