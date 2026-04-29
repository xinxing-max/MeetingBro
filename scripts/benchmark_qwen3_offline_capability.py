"""Single-shot offline capability benchmark for sherpa-onnx Qwen3-ASR 0.6B int8.

Each audio file is decoded ONCE, in full, with a single shared recognizer.
No chunking, no near-streaming simulation, no RMS endpointing, no force-finalize.
This is the *upper-bound quality reference* that all production/near-streaming
paths should be compared against.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/benchmark_qwen3_offline_capability.py

The script sets sys.stdout.reconfigure(encoding="utf-8") automatically; the env
vars above are a belt-and-suspenders fallback for subprocesses / redirected pipes.
"""
from __future__ import annotations

import sys

# Windows cp1252 stdout chokes on CJK/Arabic — force UTF-8 before any print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
TEST_WAVS_DIR = MODEL_DIR / "test_wavs"
TRANSCRIPT_FILE = TEST_WAVS_DIR / "transcript.txt"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = Path("C:/tmp/qwen3_capability")
TARGET_SR = 16_000

MEETINGBRO_WAVS = [
    DATA_DIR / "sample_en.wav",
    DATA_DIR / "voice_test_short_wikimedia.wav",
    DATA_DIR / "voice_test_statement_wikimedia.wav",
]

# Section C
_HOTWORDS_CANDIDATES = ["noise1-en.wav", "f1_noise.wav", "rap1.wav"]
HOTWORDS_STRING = "Mariachi|driveway|registration|riot|babe"

# Factory kwargs — shared recognizer uses hotwords=""
_BASE_KWARGS: dict[str, Any] = dict(
    num_threads=2,
    sample_rate=16_000,
    feature_dim=128,
    decoding_method="greedy_search",
    debug=False,
    provider="cpu",
    max_total_len=512,
    max_new_tokens=512,
    temperature=1e-6,
    top_p=0.8,
    seed=42,
    hotwords="",
)

# ---------------------------------------------------------------------------
# Memory (psutil only — no ctypes)
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)

# ---------------------------------------------------------------------------
# Audio loading (in-memory resample + downmix, no disk write)
# ---------------------------------------------------------------------------

def _load_audio(path: Path) -> np.ndarray:
    """Load WAV file → mono float32 at TARGET_SR. No disk writes."""
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Downmix to mono
    mono = samples.mean(axis=1)
    # Resample if needed
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, sr // g).astype(np.float32)
    return mono.astype(np.float32)

# ---------------------------------------------------------------------------
# Decode (times only decode_stream — not create_stream / accept_waveform)
# ---------------------------------------------------------------------------

def _decode(recognizer: Any, samples: np.ndarray) -> tuple[str, float]:
    """Returns (text, decode_wall_seconds). Timing covers only decode_stream()."""
    stream = recognizer.create_stream()
    stream.accept_waveform(TARGET_SR, samples)
    t0 = time.perf_counter()
    recognizer.decode_stream(stream)
    wall = time.perf_counter() - t0
    result = stream.result
    if hasattr(result, "text"):
        text = str(result.text).strip()
    elif isinstance(result, dict):
        text = str(result.get("text", "")).strip()
    else:
        text = str(result).strip()
    return text, wall

# ---------------------------------------------------------------------------
# WER / CER — tiny inline implementation, no jiwer
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    """O(m·n) Levenshtein, O(n) space."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

def _wer(ref: str, hyp: str) -> float:
    ref_w = _PUNCT.sub(" ", ref.lower()).split()
    hyp_w = _PUNCT.sub(" ", hyp.lower()).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _edit_distance(ref_w, hyp_w) / len(ref_w)

def _cer(ref: str, hyp: str) -> float:
    ref_c = [c for c in ref if not c.isspace()]
    hyp_c = [c for c in hyp if not c.isspace()]
    if not ref_c:
        return 0.0 if not hyp_c else 1.0
    return _edit_distance(ref_c, hyp_c) / len(ref_c)

# ---------------------------------------------------------------------------
# Script detection helpers
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\u3040-\u309f\u30a0-\u30ff]"
)
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")

def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))

_STEM_TO_LANG: dict[str, str] = {
    "ar1": "ar",
    "de": "de",
    "es1": "es",
    "fr1": "fr",
    "ru1": "ru",
    "ja1": "ja",
    "cantonese": "yue",
    "codeswitch": "mixed",
    "fast1": "zh",
    "raokouling": "zh",
    "noise2": "zh",
    "qiqiu1": "zh",
    "rap1": "en",
    "noise1-en": "en",
    "f1_noise": "en",
}

def _guess_lang(filename: str, gt: str = "") -> str:
    stem = Path(filename).stem.lower()
    if stem in _STEM_TO_LANG:
        return _STEM_TO_LANG[stem]
    # Fallback: infer from ground-truth script.
    if gt:
        if _CJK_RE.search(gt):
            return "zh"
        if _ARABIC_RE.search(gt):
            return "ar"
        if _CYRILLIC_RE.search(gt):
            return "ru"
    return "unk"

# ---------------------------------------------------------------------------
# ASCII table printer (handles Unicode in cells gracefully)
# ---------------------------------------------------------------------------

def _cell(v: Any, width: int) -> str:
    s = str(v)
    # pad to width based on character count (not byte count)
    return s + " " * max(0, width - len(s))

def _table(headers: list[str], rows: list[list[Any]], title: str = "") -> str:
    # Compute column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(min(len(row), len(widths))):
            widths[i] = max(widths[i], len(str(row[i])))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr_line = "| " + " | ".join(_cell(h, widths[i]) for i, h in enumerate(headers)) + " |"
    lines: list[str] = []
    if title:
        bar = "=" * len(sep)
        lines += [bar, title, bar]
    lines += [sep, hdr_line, sep]
    for row in rows:
        cells = [_cell(row[i] if i < len(row) else "", widths[i]) for i in range(len(widths))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(sep)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Import sherpa-onnx ──────────────────────────────────────────────────
    try:
        import sherpa_onnx  # type: ignore[import]
    except ImportError as exc:
        print(f"ERROR: sherpa-onnx not available — {exc}", file=sys.stderr)
        return 1

    # ── Verify model files ──────────────────────────────────────────────────
    conv_frontend = MODEL_DIR / "conv_frontend.onnx"
    encoder       = MODEL_DIR / "encoder.int8.onnx"
    decoder       = MODEL_DIR / "decoder.int8.onnx"
    tokenizer     = MODEL_DIR / "tokenizer"
    for p in [conv_frontend, encoder, decoder, tokenizer, TEST_WAVS_DIR]:
        if not p.exists():
            print(f"ERROR: required path not found: {p}", file=sys.stderr)
            return 1

    # ── Env / model summary ─────────────────────────────────────────────────
    import platform
    sherpa_ver = getattr(sherpa_onnx, "__version__", "unknown")
    bar = "=" * 72
    print(bar)
    print("  sherpa-onnx Qwen3-ASR 0.6B int8  —  OFFLINE CAPABILITY BENCHMARK")
    print(bar)
    print(f"  model      : {MODEL_DIR.name}")
    print(f"  python     : {sys.version.split()[0]}")
    print(f"  platform   : {platform.system()} {platform.release()}")
    print(f"  sherpa-onnx: {sherpa_ver}")
    print(f"  threads    : {_BASE_KWARGS['num_threads']}  provider: {_BASE_KWARGS['provider']}")
    print(f"  max_total  : {_BASE_KWARGS['max_total_len']}  max_new_tokens: {_BASE_KWARGS['max_new_tokens']}")
    print(f"  temperature: {_BASE_KWARGS['temperature']}  top_p: {_BASE_KWARGS['top_p']}  seed: {_BASE_KWARGS['seed']}")
    print()

    # ── Parse ground truth ──────────────────────────────────────────────────
    ground_truth: dict[str, str] = {}
    if TRANSCRIPT_FILE.is_file():
        with open(TRANSCRIPT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    ground_truth[parts[0]] = parts[1]

    # ── Build shared recognizer (cold, timed) ──────────────────────────────
    rss_pre_build = _rss_mb()
    print("Building shared recognizer (cold, timed)…")
    t_build = time.perf_counter()
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(conv_frontend),
            encoder=str(encoder),
            decoder=str(decoder),
            tokenizer=str(tokenizer),
            **_BASE_KWARGS,
        )
    except Exception as exc:
        print(f"ERROR: recognizer build failed — {exc}", file=sys.stderr)
        return 1
    cold_build_s = time.perf_counter() - t_build
    rss_post_build = _rss_mb()
    peak_rss = rss_post_build

    print(f"  Build wall : {cold_build_s:.2f}s")
    print(f"  RSS delta  : {rss_post_build - rss_pre_build:.1f} MB  (now {rss_post_build:.1f} MB)")
    print()

    # Accumulate results payload
    all_results: dict[str, Any] = {
        "model": MODEL_DIR.name,
        "cold_build_wall_s": round(cold_build_s, 3),
        "factory_kwargs": dict(_BASE_KWARGS),
        "sections": {},
    }
    rtfs_ab: list[float] = []   # RTFs for VERDICT mean/max across A + B

    # ───────────────────────────────────────────────────────────────────────
    # SECTION A — Bundled multilingual reference quality
    # ───────────────────────────────────────────────────────────────────────
    print(bar)
    print("SECTION A — Bundled multilingual reference quality")
    print(bar)

    A_HEADERS = ["file", "lang", "dur_s", "wall_s", "RTF", "rss_mb", "error_rate", "decoded (80ch)"]
    sec_a_rows: list[list[Any]] = []
    sec_a_items: list[dict[str, Any]] = []
    cer_vals: list[float] = []
    wer_vals: list[float] = []

    # Cache codeswitch result for Section E reuse
    _cs_cache: dict[str, Any] = {}

    wav_files_a = sorted(
        p for p in TEST_WAVS_DIR.iterdir()
        if p.suffix.lower() == ".wav"
    )

    for wav_path in wav_files_a:
        fname = wav_path.name
        gt = ground_truth.get(fname, "")
        lang = _guess_lang(fname, gt)
        use_cer = _has_cjk(gt)
        item: dict[str, Any] = {"file": fname, "lang": lang, "ground_truth": gt}

        try:
            samples = _load_audio(wav_path)
            dur_s = len(samples) / TARGET_SR
            rss_pre = _rss_mb()
            text, wall = _decode(recognizer, samples)
            rss_post = _rss_mb()
            peak_rss = max(peak_rss, rss_post)
            rtf = wall / dur_s if dur_s > 0 else 0.0
            rtfs_ab.append(rtf)

            if gt:
                if use_cer:
                    er = _cer(gt, text)
                    er_label = "CER"
                    cer_vals.append(er)
                else:
                    er = _wer(gt, text)
                    er_label = "WER"
                    wer_vals.append(er)
                er_str = f"{er:.3f} {er_label}"
            else:
                er = None
                er_str = "—"

            item.update({
                "dur_s": round(dur_s, 2),
                "decode_wall_s": round(wall, 3),
                "rtf": round(rtf, 4),
                "rss_peak_mb": round(rss_post, 1),
                "error_rate": round(er, 4) if er is not None else None,
                "error_type": ("CER" if use_cer else "WER") if gt else None,
                "decoded": text,
                "error": None,
            })
            text_sample = (text[:77] + "…") if len(text) > 80 else text
            sec_a_rows.append([fname[:24], lang, f"{dur_s:.1f}", f"{wall:.3f}",
                                f"{rtf:.3f}", f"{rss_post:.0f}", er_str, text_sample])

            # Cache for section E
            if fname == "codeswitch.wav":
                _cs_cache = {"dur_s": dur_s, "wall": wall, "rtf": rtf,
                             "text": text, "rss_post": rss_post}

        except Exception as exc:
            item["error"] = str(exc)
            sec_a_rows.append([fname[:24], lang, "ERR", "ERR", "ERR", "ERR", "ERR", str(exc)[:80]])
            print(f"  [WARN] {fname}: {exc}")

        sec_a_items.append(item)

    print(_table(A_HEADERS, sec_a_rows))
    mean_cer = sum(cer_vals) / len(cer_vals) if cer_vals else None
    mean_wer = sum(wer_vals) / len(wer_vals) if wer_vals else None
    if mean_cer is not None:
        print(f"  Mean CER (CJK  items, n={len(cer_vals)}): {mean_cer:.4f}")
    if mean_wer is not None:
        print(f"  Mean WER (Latin items, n={len(wer_vals)}): {mean_wer:.4f}")
    print()

    all_results["sections"]["A"] = {
        "items": sec_a_items,
        "mean_cer_cjk": round(mean_cer, 4) if mean_cer is not None else None,
        "mean_wer_other": round(mean_wer, 4) if mean_wer is not None else None,
    }

    # ───────────────────────────────────────────────────────────────────────
    # SECTION B — MeetingBro real audio
    # ───────────────────────────────────────────────────────────────────────
    print(bar)
    print("SECTION B — MeetingBro real audio (in-memory resample + downmix)")
    print(bar)

    sec_b_rows: list[list[Any]] = []
    sec_b_items: list[dict[str, Any]] = []

    for wav_path in MEETINGBRO_WAVS:
        fname = wav_path.name
        item2: dict[str, Any] = {"file": str(wav_path)}

        try:
            samples = _load_audio(wav_path)
            dur_s = len(samples) / TARGET_SR
            rss_pre = _rss_mb()
            text, wall = _decode(recognizer, samples)
            rss_post = _rss_mb()
            peak_rss = max(peak_rss, rss_post)
            rtf = wall / dur_s if dur_s > 0 else 0.0
            rtfs_ab.append(rtf)
            text_sample = (text[:77] + "…") if len(text) > 80 else text

            item2.update({
                "dur_s": round(dur_s, 2),
                "decode_wall_s": round(wall, 3),
                "rtf": round(rtf, 4),
                "rss_peak_mb": round(rss_post, 1),
                "decoded": text,
                "error": None,
            })
            sec_b_rows.append([fname[:30], "en", f"{dur_s:.1f}", f"{wall:.3f}",
                                f"{rtf:.3f}", f"{rss_post:.0f}", "—", text_sample])

        except Exception as exc:
            item2["error"] = str(exc)
            sec_b_rows.append([fname[:30], "en", "ERR", "ERR", "ERR", "ERR", "—", str(exc)[:80]])
            print(f"  [WARN] {fname}: {exc}")

        sec_b_items.append(item2)

    print(_table(A_HEADERS, sec_b_rows))
    print()
    all_results["sections"]["B"] = {"items": sec_b_items}

    # ───────────────────────────────────────────────────────────────────────
    # SECTION C — Hotwords ablation
    # ───────────────────────────────────────────────────────────────────────
    print(bar)
    print("SECTION C — Hotwords ablation")
    print(bar)

    # Pick the first available candidate file.
    hw_file: Path | None = None
    for candidate in _HOTWORDS_CANDIDATES:
        p = TEST_WAVS_DIR / candidate
        if p.is_file():
            hw_file = p
            break

    sec_c: dict[str, Any] = {
        "file": str(hw_file) if hw_file else None,
        "hotwords": HOTWORDS_STRING,
    }

    C_HEADERS = ["mode", "dur_s", "wall_s", "RTF", "rss_mb", "hw_found", "decoded (80ch)"]

    if hw_file is None:
        print("  [WARN] No hotwords ablation file found; skipping section C.")
        sec_c["error"] = "no candidate file found"
        sec_c_rows: list[list[Any]] = []
    else:
        print(f"  Audio file : {hw_file.name}")
        print(f"  Hotwords   : {HOTWORDS_STRING}\n")
        try:
            samples_hw = _load_audio(hw_file)
            dur_hw = len(samples_hw) / TARGET_SR

            # Without hotwords (shared recognizer)
            text_nohw, wall_nohw = _decode(recognizer, samples_hw)
            rss_nohw = _rss_mb()
            peak_rss = max(peak_rss, rss_nohw)
            rtf_nohw = wall_nohw / dur_hw

            # Build second recognizer WITH hotwords
            print("  Building hotwords recognizer (second instance)…")
            kw_hw = dict(_BASE_KWARGS, hotwords=HOTWORDS_STRING)
            t_hw_build = time.perf_counter()
            recognizer_hw = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=str(conv_frontend),
                encoder=str(encoder),
                decoder=str(decoder),
                tokenizer=str(tokenizer),
                **kw_hw,
            )
            hw_build_s = time.perf_counter() - t_hw_build
            print(f"  Hotwords recognizer built in {hw_build_s:.2f}s\n")

            text_hw, wall_hw = _decode(recognizer_hw, samples_hw)
            rss_hw_post = _rss_mb()
            peak_rss = max(peak_rss, rss_hw_post)
            rtf_hw = wall_hw / dur_hw

            hw_words = [w.lower() for w in HOTWORDS_STRING.split("|")]
            hw_in_nohw = [w for w in hw_words if w in text_nohw.lower()]
            hw_in_hw   = [w for w in hw_words if w in text_hw.lower()]
            measurable  = (text_nohw.strip() != text_hw.strip())

            print(f"  WITHOUT hotwords:")
            print(f"    {text_nohw[:300]}")
            print(f"  WITH hotwords:")
            print(f"    {text_hw[:300]}")
            print(f"\n  Hotwords in no-hw output : {hw_in_nohw}")
            print(f"  Hotwords in hw output    : {hw_in_hw}")
            print(f"  Text changed             : {measurable}")

            sec_c_rows = [
                ["no hotwords",   f"{dur_hw:.1f}", f"{wall_nohw:.3f}", f"{rtf_nohw:.3f}",
                 f"{rss_nohw:.0f}", str(hw_in_nohw),
                 (text_nohw[:77] + "…") if len(text_nohw) > 80 else text_nohw],
                ["with hotwords", f"{dur_hw:.1f}", f"{wall_hw:.3f}",  f"{rtf_hw:.3f}",
                 f"{rss_hw_post:.0f}", str(hw_in_hw),
                 (text_hw[:77] + "…") if len(text_hw) > 80 else text_hw],
            ]

            sec_c.update({
                "dur_s": round(dur_hw, 2),
                "text_no_hotwords": text_nohw,
                "text_with_hotwords": text_hw,
                "wall_no_hotwords_s": round(wall_nohw, 3),
                "wall_with_hotwords_s": round(wall_hw, 3),
                "rtf_no_hotwords": round(rtf_nohw, 4),
                "rtf_with_hotwords": round(rtf_hw, 4),
                "hotwords_in_no_hw_output": hw_in_nohw,
                "hotwords_in_hw_output": hw_in_hw,
                "measurable_effect": measurable,
                "error": None,
            })

        except Exception as exc:
            sec_c["error"] = str(exc)
            sec_c_rows = []
            print(f"  [ERROR] Section C failed: {exc}")
            measurable = None
            hw_in_nohw = []
            hw_in_hw = []

        print()
        if sec_c_rows:
            print(_table(C_HEADERS, sec_c_rows))
        print()

    all_results["sections"]["C"] = sec_c

    # ───────────────────────────────────────────────────────────────────────
    # SECTION D — Silence robustness
    # ───────────────────────────────────────────────────────────────────────
    print(bar)
    print("SECTION D — Silence robustness")
    print(bar)

    rng = np.random.default_rng(42)
    silence_cases: list[tuple[str, np.ndarray]] = [
        ("pure_silence_5s",          np.zeros(TARGET_SR * 5, dtype=np.float32)),
        ("low_noise_5s_σ=0.001",     rng.normal(0.0, 0.001, TARGET_SR * 5).astype(np.float32)),
    ]

    sec_d: dict[str, Any] = {}
    d_rows: list[list[Any]] = []

    for case_name, samples_d in silence_cases:
        dur_d = len(samples_d) / TARGET_SR
        try:
            rss_pre_d = _rss_mb()
            text_d, wall_d = _decode(recognizer, samples_d)
            rss_post_d = _rss_mb()
            peak_rss = max(peak_rss, rss_post_d)
            rtf_d = wall_d / dur_d
            hallucinated = bool(text_d.strip())
            sec_d[case_name] = {
                "dur_s": dur_d,
                "decode_wall_s": round(wall_d, 3),
                "rtf": round(rtf_d, 4),
                "rss_peak_mb": round(rss_post_d, 1),
                "decoded": text_d,
                "hallucinated": hallucinated,
            }
            d_rows.append([
                case_name,
                f"{dur_d:.1f}",
                f"{wall_d:.3f}",
                f"{rtf_d:.3f}",
                f"{rss_post_d:.0f}",
                "YES — hallucination!" if hallucinated else "clean (empty)",
                (text_d[:77] + "…") if len(text_d) > 80 else (text_d or "(empty)"),
            ])
        except Exception as exc:
            sec_d[case_name] = {"error": str(exc)}
            d_rows.append([case_name, "ERR", "ERR", "ERR", "ERR", "ERR", str(exc)[:80]])

    D_HEADERS = ["case", "dur_s", "wall_s", "RTF", "rss_mb", "hallucinated?", "decoded"]
    print(_table(D_HEADERS, d_rows))
    print()
    all_results["sections"]["D"] = sec_d

    # ───────────────────────────────────────────────────────────────────────
    # SECTION E — Code-switch (reuse cached A result for codeswitch.wav)
    # ───────────────────────────────────────────────────────────────────────
    print(bar)
    print("SECTION E — Code-switch (codeswitch.wav)")
    print(bar)

    cs_gt = ground_truth.get("codeswitch.wav", "")
    sec_e: dict[str, Any] = {
        "file": str(TEST_WAVS_DIR / "codeswitch.wav"),
        "ground_truth": cs_gt,
    }

    if _cs_cache:
        dur_cs  = _cs_cache["dur_s"]
        wall_cs = _cs_cache["wall"]
        rtf_cs  = _cs_cache["rtf"]
        text_cs = _cs_cache["text"]
        rss_cs  = _cs_cache["rss_post"]
        wer_cs  = _wer(cs_gt, text_cs) if cs_gt else None

        # Per-language coverage of the expected segments
        lang_coverage = {
            "English (alone|myself)": any(w in text_cs.lower() for w in ["alone", "myself"]),
            "French (tout seul|suis)": any(w in text_cs.lower() for w in ["tout", "seul", "suis"]),
            "Italian (tutto|sono)":   any(w in text_cs.lower() for w in ["tutto", "sono"]),
            "Spanish (estoy|solo)":   any(w in text_cs.lower() for w in ["estoy", "solo"]),
        }

        print(f"  Ground truth : {cs_gt}")
        print(f"  Decoded      : {text_cs}")
        print(f"  WER          : {wer_cs:.3f}" if wer_cs is not None else "  WER          : N/A")
        print(f"  Duration     : {dur_cs:.2f}s  wall: {wall_cs:.3f}s  RTF: {rtf_cs:.3f}")
        print("  Language coverage:")
        for lang_label, present in lang_coverage.items():
            print(f"    {lang_label}: {'PRESENT' if present else 'MISSING / language-drifted'}")

        sec_e.update({
            "dur_s": round(dur_cs, 2),
            "decode_wall_s": round(wall_cs, 3),
            "rtf": round(rtf_cs, 4),
            "decoded": text_cs,
            "wer": round(wer_cs, 4) if wer_cs is not None else None,
            "lang_coverage": lang_coverage,
            "error": None,
        })

        e_rows = [[
            "codeswitch.wav", "mixed",
            f"{dur_cs:.1f}", f"{wall_cs:.3f}", f"{rtf_cs:.3f}", f"{rss_cs:.0f}",
            f"{wer_cs:.3f} WER" if wer_cs is not None else "—",
            (text_cs[:77] + "…") if len(text_cs) > 80 else text_cs,
        ]]
        print()
        print(_table(A_HEADERS, e_rows))
    else:
        sec_e["error"] = "codeswitch.wav was not decoded in section A"
        print("  [WARN] codeswitch.wav result not found (decode failed in section A).")

    print()
    all_results["sections"]["E"] = sec_e

    # ───────────────────────────────────────────────────────────────────────
    # Write JSON (first pass, without verdict)
    # ───────────────────────────────────────────────────────────────────────
    out_json = OUTPUT_DIR / "results.json"
    with open(out_json, "w", encoding="utf-8") as fj:
        json.dump(all_results, fj, ensure_ascii=False, indent=2)

    # ───────────────────────────────────────────────────────────────────────
    # VERDICT
    # ───────────────────────────────────────────────────────────────────────
    mean_rtf = sum(rtfs_ab) / len(rtfs_ab) if rtfs_ab else float("nan")
    max_rtf  = max(rtfs_ab) if rtfs_ab else float("nan")

    sil_pure  = sec_d.get("pure_silence_5s",       {})
    sil_noise = sec_d.get("low_noise_5s_σ=0.001",  {})
    sil_pure_hall  = sil_pure.get("hallucinated",  "error")
    sil_noise_hall = sil_noise.get("hallucinated", "error")
    sil_pure_text  = sil_pure.get("decoded",  "")
    sil_noise_text = sil_noise.get("decoded", "")

    hw_eff    = sec_c.get("measurable_effect",         "error")
    hw_in_no  = sec_c.get("hotwords_in_no_hw_output",  [])
    hw_in_yes = sec_c.get("hotwords_in_hw_output",     [])

    print(bar)
    print("VERDICT")
    print(bar)
    print(f"  Mean RTF (sections A+B)    : {mean_rtf:.4f}  ({1/mean_rtf:.1f}× real-time)" if mean_rtf == mean_rtf else "  Mean RTF: N/A")
    print(f"  Max  RTF (sections A+B)    : {max_rtf:.4f}")
    print(f"  Cold construct wall time   : {cold_build_s:.2f}s")
    print(f"  Peak RSS observed          : {peak_rss:.1f} MB")
    print(f"  Mean CER (CJK  items, A)   : {f'{mean_cer:.4f}' if mean_cer is not None else 'N/A'}"
          + (f"  (n={len(cer_vals)})" if cer_vals else ""))
    print(f"  Mean WER (Latin items, A)  : {f'{mean_wer:.4f}' if mean_wer is not None else 'N/A'}"
          + (f"  (n={len(wer_vals)})" if wer_vals else ""))
    if sil_pure_hall == "error":
        print("  Silence pure 5s (D)        : test error")
    elif sil_pure_hall:
        print(f"  Silence pure 5s (D)        : HALLUCINATED → {repr(sil_pure_text)[:80]}")
    else:
        print("  Silence pure 5s (D)        : clean — no text emitted")
    if sil_noise_hall == "error":
        print("  Silence low-noise 5s (D)   : test error")
    elif sil_noise_hall:
        print(f"  Silence low-noise 5s (D)   : HALLUCINATED → {repr(sil_noise_text)[:80]}")
    else:
        print("  Silence low-noise 5s (D)   : clean — no text emitted")
    if hw_eff == "error":
        print("  Hotwords effect (C)        : test error")
    elif hw_eff:
        print(f"  Hotwords effect (C)        : YES — no-hw found {hw_in_no}, with-hw found {hw_in_yes}")
    else:
        print(f"  Hotwords effect (C)        : no measurable difference  (both found {hw_in_no})")
    print(bar)
    print(f"\nFull results JSON: {out_json}")

    # Final JSON write with verdict included
    all_results["verdict"] = {
        "mean_rtf_ab": round(mean_rtf, 4) if mean_rtf == mean_rtf else None,
        "max_rtf_ab": round(max_rtf, 4) if max_rtf == max_rtf else None,
        "cold_build_wall_s": round(cold_build_s, 3),
        "peak_rss_mb": round(peak_rss, 1),
        "mean_cer_cjk": round(mean_cer, 4) if mean_cer is not None else None,
        "mean_wer_other": round(mean_wer, 4) if mean_wer is not None else None,
        "silence_pure_5s_hallucinated": bool(sil_pure_hall) if sil_pure_hall != "error" else None,
        "silence_noise_5s_hallucinated": bool(sil_noise_hall) if sil_noise_hall != "error" else None,
        "hotwords_measurable_effect": hw_eff if hw_eff != "error" else None,
        "hotwords_in_no_hw_output": hw_in_no,
        "hotwords_in_hw_output": hw_in_yes,
    }
    with open(out_json, "w", encoding="utf-8") as fj:
        json.dump(all_results, fj, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
