import { useEffect, useMemo, useRef, useState } from "react";
import type { SummarySnapshot } from "./types";
import { useSessionSocket } from "./session/useSessionSocket";

function formatRange(start: number, end: number): string {
  const fmt = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, "0");
    const sec = Math.floor(s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  };
  return `${fmt(start)} – ${fmt(end)}`;
}

function formatApproxClock(iso: string | null, seconds: number): string {
  if (!iso) return formatRange(seconds, seconds).slice(0, 5);
  const dt = new Date(new Date(iso).getTime() + seconds * 1000);
  return dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatApproxClockRange(iso: string | null, start: number, end: number): string {
  return `${formatApproxClock(iso, start)} – ${formatApproxClock(iso, end)}`;
}

function formatCreatedAt(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatLagSeconds(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  return `${Math.round(seconds)}s`;
}

function getDelayTone(seconds: number | null): "ok" | "warn" | "danger" | "unknown" {
  if (seconds == null) return "unknown";
  if (seconds < 1.2) return "ok";
  if (seconds < 2.8) return "warn";
  return "danger";
}

function getConfidenceTone(confidence: number): "stable" | "soft" | "uncertain" {
  if (confidence < 0.48) return "uncertain";
  if (confidence < 0.67) return "soft";
  return "stable";
}

function getConfidenceLabel(confidence: number): string | null {
  const tone = getConfidenceTone(confidence);
  if (tone === "uncertain") return "uncertain";
  if (tone === "soft") return "low conf";
  return null;
}

function formatElapsedSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60).toString().padStart(2, "0");
  const s = Math.floor(total % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function formatGain(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${value.toFixed(2)}x`;
}

function formatRtf(value: number | null | undefined): string {
  if (value == null) return "???";
  return `${value.toFixed(2)}x`;
}

function getGainTone(
  effectiveMicrophoneGain: number | null | undefined,
  baseMicrophoneGain: number | null | undefined,
  activeSource: string | null | undefined,
): "ok" | "warn" | "danger" | "unknown" {
  if (activeSource !== "mixed") return "unknown";
  if (effectiveMicrophoneGain == null || baseMicrophoneGain == null || baseMicrophoneGain <= 0) {
    return "unknown";
  }
  const boostRatio = effectiveMicrophoneGain / baseMicrophoneGain;
  if (boostRatio < 1.15) return "ok";
  if (boostRatio < 1.45) return "warn";
  return "danger";
}

function formatSourceLabel(source: string | null | undefined): string {
  switch (source) {
    case "mic":
      return "Microphone";
    case "loopback":
    case "system":
      return "System Audio";
    case "mixed":
      return "System + Mic";
    default:
      return source ?? "—";
  }
}

function formatLanguageLabel(language: "zh" | "en" | "de" | null | undefined): string {
  switch (language) {
    case "zh":
      return "Chinese";
    case "en":
      return "English";
    case "de":
      return "German";
    default:
      return "Off";
  }
}

function getMixedGainHint(
  tone: "ok" | "warn" | "danger" | "unknown",
  activeSource: string | null | undefined,
): string {
  if (activeSource !== "mixed") {
    return "mixed diagnostics appear when source is System + Mic";
  }
  if (tone === "danger") {
    return "mic seems weak · consider raising MEETINGBRO_MIXED_MIC_GAIN";
  }
  if (tone === "warn") {
    return "moderate mic compensation active · monitor voice clarity";
  }
  if (tone === "ok") {
    return "mic and system levels look balanced";
  }
  return "waiting for mixed-level diagnostics";
}

function isNearBottom(element: HTMLDivElement): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 48;
}

interface SummaryPanelProps {
  title: string;
  subtitle: string;
  snapshot: SummarySnapshot | undefined;
  history: SummarySnapshot[];
  sessionStartedAt: string | null;
  accent: string;
  className?: string;
  onSaveToNotes: (snapshot: SummarySnapshot) => Promise<void>;
}

function SummaryPanel({ title, subtitle, snapshot, history, sessionStartedAt, accent, className, onSaveToNotes }: SummaryPanelProps) {
  const displayedHistory = history.length > 0 ? history : snapshot ? [snapshot] : [];
  const latestSnapshot = displayedHistory.at(-1) ?? snapshot;
  const range = latestSnapshot
    ? formatApproxClockRange(sessionStartedAt, latestSnapshot.time_start, latestSnapshot.time_end)
    : "—";
  const relativeRange = latestSnapshot
    ? formatRange(latestSnapshot.time_start, latestSnapshot.time_end)
    : "—";
  const createdAt = latestSnapshot
    ? new Date(latestSnapshot.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";

  const bodyRef = useRef<HTMLDivElement>(null);
  const autoFollowRef = useRef(true);
  useEffect(() => {
    if (autoFollowRef.current && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [displayedHistory.length, latestSnapshot?.id]);

  const onCopy = async () => {
    if (!latestSnapshot) return;
    await navigator.clipboard.writeText(latestSnapshot.content);
  };
  const onExpand = () => {
    if (!latestSnapshot) return;
    const historyText = history
      .map((item, idx) => `#${idx + 1}  ${formatApproxClockRange(sessionStartedAt, item.time_start, item.time_end)}\n${item.content}`)
      .join("\n\n");
    alert(`${title} — ${range}\n${relativeRange}\n\n${historyText}`);
  };
  const onSave = async () => {
    if (!latestSnapshot) return;
    await onSaveToNotes(latestSnapshot);
  };

  return (
    <section className={`panel ${className ?? ""}`.trim()} style={{ borderTop: `3px solid ${accent}` }}>
      <header>
        <div>
          <h2>{title}</h2>
          <p className="subtitle">{subtitle}</p>
          <p className="summary-meta">covered {relativeRange} · updated {createdAt}</p>
        </div>
        <div className="range">{range}</div>
      </header>
      <div
        ref={bodyRef}
        className="panel-body summary-history"
        onScroll={(e) => {
          autoFollowRef.current = isNearBottom(e.currentTarget);
        }}
      >
        {displayedHistory.length === 0 ? (
          "waiting for the first snapshot…"
        ) : (
          displayedHistory.map((item, idx) => {
            const itemRange = formatApproxClockRange(sessionStartedAt, item.time_start, item.time_end);
            const itemRelativeRange = formatRange(item.time_start, item.time_end);
            const itemCreatedAt = new Date(item.created_at).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            });
            const isLatest = idx === displayedHistory.length - 1;
            return (
              <article key={item.id} className={`summary-history-item ${isLatest ? "latest" : ""}`.trim()}>
                <div className="summary-history-meta">
                  <span>{itemRange}</span>
                  <span>{itemRelativeRange}</span>
                  <span>{itemCreatedAt}</span>
                </div>
                <div>{item.content}</div>
              </article>
            );
          })
        )}
      </div>
      <footer>
        <button onClick={onCopy} disabled={!latestSnapshot}>Copy Latest</button>
        <button onClick={onExpand} disabled={!latestSnapshot}>Expand</button>
        <button onClick={onSave} disabled={!latestSnapshot}>Save Latest</button>
      </footer>
    </section>
  );
}

export default function App() {
  const [source, setSource] = useState("mixed");
  const [speechLanguage, setSpeechLanguage] = useState("auto");
  const [summaryLanguage, setSummaryLanguage] = useState("en");
  const [subtitleLanguage, setSubtitleLanguage] = useState("off");
  const [sessionEnabled, setSessionEnabled] = useState(false);
  const { connected, state, meetingId, sessionStartedAt, elapsedSeconds, sessionStats, segments, previewSegment, latestByType, historyByType, notes, lastError, saveNote, stopSession } =
    useSessionSocket({ enabled: sessionEnabled, source, speechLanguage, summaryLanguage, subtitleLanguage });

  const rolling = latestByType.rolling_summary;
  const cumulative = latestByType.cumulative_meeting_summary;

  const visibleSegments = useMemo(() => segments.slice(-200), [segments]);
  const latestSegment = segments.at(-1);
  const latestVisualSegment = previewSegment ?? latestSegment;
  const latestVisibleSegment = visibleSegments.at(-1);
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!sessionEnabled) return;
    const timer = window.setInterval(() => setNowMs(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [sessionEnabled]);

  const audioClockSeconds = useMemo(() => {
    if (elapsedSeconds > 0) {
      return elapsedSeconds;
    }
    if (!sessionStartedAt) {
      return null;
    }
    const sessionStartMs = new Date(sessionStartedAt).getTime();
    if (Number.isNaN(sessionStartMs)) {
      return null;
    }
    return Math.max(0, (nowMs - sessionStartMs) / 1000);
  }, [elapsedSeconds, nowMs, sessionStartedAt]);

  const committedLagSeconds = useMemo(() => {
    if (!latestSegment || state !== "running" || audioClockSeconds == null) {
      return null;
    }
    const committedAt = latestSegment.emitted_at_elapsed_seconds;
    if (committedAt != null) {
      return Math.max(0, committedAt - latestSegment.end_time);
    }
    return Math.max(0, audioClockSeconds - latestSegment.end_time);
  }, [audioClockSeconds, latestSegment, state]);

  const liveGapSeconds = useMemo(() => {
    if (!latestVisualSegment || state !== "running" || audioClockSeconds == null) {
      return null;
    }
    return Math.max(0, audioClockSeconds - latestVisualSegment.end_time);
  }, [audioClockSeconds, latestVisualSegment, state]);

  const transcriptLagSeconds = liveGapSeconds ?? committedLagSeconds;

  const latestSegmentRange = latestVisualSegment
    ? formatRange(latestVisualSegment.start_time, latestVisualSegment.end_time)
    : "—";
  const visibleTranscriptSeconds = latestSegment?.end_time ?? null;
  const backpressureAgeSeconds = useMemo(() => {
    const lastBackpressure = sessionStats?.last_backpressure_elapsed_seconds;
    if (lastBackpressure == null || elapsedSeconds <= 0) {
      return null;
    }
    return Math.max(0, elapsedSeconds - lastBackpressure);
  }, [elapsedSeconds, sessionStats?.last_backpressure_elapsed_seconds]);
  const retryTotal = sessionStats?.retry_windows_total ?? 0;
  const retryImproved = sessionStats?.retry_windows_improved ?? 0;
  const retryUnchanged = sessionStats?.retry_windows_unchanged ?? 0;
  const retryDiverged = sessionStats?.retry_windows_diverged ?? 0;
  const asrRealtimeFactor = sessionStats?.asr_realtime_factor ?? null;
  const asrWallSeconds = sessionStats?.asr_last_wall_seconds ?? null;
  const asrAudioSeconds = sessionStats?.asr_last_audio_seconds ?? null;
  const asrSafeguardActive = sessionStats?.asr_safeguard_active ?? false;
  const asrSafeguardEvents = sessionStats?.asr_safeguard_events ?? 0;
  const weakRescueAttempts = sessionStats?.weak_rescue_attempts ?? 0;
  const weakRescueEmitted = sessionStats?.weak_rescue_emitted ?? 0;
  const weakRescueBufferSeconds = sessionStats?.weak_rescue_buffer_seconds ?? 0;
  const translationPendingCount = sessionStats?.translation_pending_count ?? 0;
  const summaryPendingCount = sessionStats?.summary_pending_count ?? 0;
  const translationTrimTotal = sessionStats?.translation_backlog_trim_total ?? 0;
  const audioDropTotal = sessionStats?.audio_drop_total ?? 0;
  const delayTone = getDelayTone(transcriptLagSeconds);
  const activeSource = sessionStats?.source ?? source;
  const mixedMicGain = sessionStats?.mixed_microphone_gain ?? null;
  const mixedSystemGain = sessionStats?.mixed_system_gain ?? null;
  const mixedEffectiveMicGain = sessionStats?.mixed_effective_microphone_gain ?? null;
  const mixedAutoBalanceEnabled = sessionStats?.mixed_auto_balance_enabled ?? null;
  const mixedGainTone = getGainTone(mixedEffectiveMicGain, mixedMicGain, activeSource);
  const mixedGainRatio =
    activeSource === "mixed"
      ? `mic ${formatGain(mixedEffectiveMicGain)} vs system ${formatGain(mixedSystemGain)}`
      : "—";
  const mixedGainHint = getMixedGainHint(mixedGainTone, activeSource);
  const activeSubtitleLanguage: "zh" | "en" | "de" | null = subtitleLanguage === "off"
    ? null
    : ((sessionStats?.live_translation_language ?? subtitleLanguage) as "zh" | "en" | "de");
  const latestSubtitleText = activeSubtitleLanguage && latestSegment
    ? latestSegment.translations[activeSubtitleLanguage]
    : null;
  const subtitleStatus = activeSubtitleLanguage == null
    ? "off"
    : !latestSegment
      ? "waiting"
      : latestSegment.original_language === activeSubtitleLanguage
        ? "same language"
        : latestSubtitleText && latestSubtitleText.trim()
          ? "translated"
          : "translating";
  const subtitleStatusDetail = activeSubtitleLanguage == null
    ? "subtitles disabled"
    : !latestSegment
      ? `target ${formatLanguageLabel(activeSubtitleLanguage)} · waiting for transcript`
      : latestSegment.original_language === activeSubtitleLanguage
        ? `latest segment already in ${formatLanguageLabel(activeSubtitleLanguage)}`
        : latestSubtitleText && latestSubtitleText.trim()
          ? `latest subtitle ready in ${formatLanguageLabel(activeSubtitleLanguage)}`
          : `target ${formatLanguageLabel(activeSubtitleLanguage)} · translation pending`;

  const transcriptBodyRef = useRef<HTMLDivElement>(null);
  const transcriptBottomRef = useRef<HTMLDivElement>(null);
  const transcriptAutoFollowRef = useRef(true);
  const transcriptProgrammaticScrollRef = useRef(false);
  const [transcriptAutoFollow, setTranscriptAutoFollow] = useState(true);

  const scrollTranscriptToBottom = (behavior: ScrollBehavior = "auto") => {
    window.requestAnimationFrame(() => {
      const body = transcriptBodyRef.current;
      if (!body) return;
      transcriptProgrammaticScrollRef.current = true;
      body.scrollTo({ top: body.scrollHeight, behavior });
      transcriptBottomRef.current?.scrollIntoView({ block: "end", behavior });
      window.setTimeout(() => {
        const latestBody = transcriptBodyRef.current;
        if (latestBody && transcriptAutoFollowRef.current) {
          latestBody.scrollTop = latestBody.scrollHeight;
        }
        transcriptProgrammaticScrollRef.current = false;
      }, behavior === "smooth" ? 350 : 80);
    });
  };

  useEffect(() => {
    if (transcriptAutoFollowRef.current) {
      scrollTranscriptToBottom("auto");
    }
  }, [
    visibleSegments.length,
    previewSegment?.id,
    previewSegment?.text,
    latestSegment?.id,
    latestSubtitleText,
    activeSubtitleLanguage,
    latestVisibleSegment?.translations[activeSubtitleLanguage ?? "en"],
  ]);

  const notesBodyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (notesBodyRef.current) {
      notesBodyRef.current.scrollTop = notesBodyRef.current.scrollHeight;
    }
  }, [notes]);

  const handleSaveToNotes = async (snapshot: SummarySnapshot) => {
    await saveNote({
      content: snapshot.content,
      source_type: snapshot.summary_type,
      source_id: snapshot.id,
    });
  };

  const transcriptEmptyMessage = !sessionEnabled
    ? "session stopped — press Start to begin capture"
    : state === "starting"
      ? "connecting to the live session…"
      : activeSource === "loopback" || activeSource === "system"
        ? "listening to system audio only — play meeting audio or switch to System + Microphone"
        : activeSource === "mic"
          ? "listening to microphone — speak to start the transcript"
          : "listening to system audio + microphone — speak or play meeting audio to start the transcript";

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <strong>MeetingBro</strong>
          <span className="sep">·</span>
          <span>session: {state}</span>
          {meetingId && (
            <>
              <span className="sep">·</span>
              <span className="muted">{meetingId.slice(0, 8)}</span>
            </>
          )}
        </div>
        <div className="header-controls">
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="mic">Microphone</option>
            <option value="loopback">System Audio (Loopback)</option>
            <option value="mixed">System Audio + Microphone</option>
          </select>
          <select value={speechLanguage} onChange={(e) => setSpeechLanguage(e.target.value)}>
            <option value="auto">Speech: Auto / Multilingual</option>
            <option value="en">Speech: English</option>
            <option value="zh">Speech: 中文</option>
            <option value="de">Speech: Deutsch</option>
          </select>
          <select value={summaryLanguage} onChange={(e) => setSummaryLanguage(e.target.value)}>
            <option value="en">Summary: English</option>
            <option value="zh">Summary: 中文</option>
            <option value="de">Summary: Deutsch</option>
          </select>
          <select value={subtitleLanguage} onChange={(e) => setSubtitleLanguage(e.target.value)}>
            <option value="off">Subtitles: Off</option>
            <option value="en">Subtitles: English</option>
            <option value="zh">Subtitles: 中文</option>
            <option value="de">Subtitles: Deutsch</option>
          </select>
          {!sessionEnabled ? (
            <button className="start-btn" onClick={() => setSessionEnabled(true)}>Start</button>
          ) : (
            <button
              className="stop-btn"
              onClick={() => {
                stopSession();
                setSessionEnabled(false);
              }}
            >
              Stop
            </button>
          )}
          <span className={connected ? "dot ok" : "dot bad"} />
          {connected ? "connected" : "disconnected"}
        </div>
      </header>

      {lastError && <div className="error">{lastError}</div>}

      <main className="grid">
        <section className="panel transcript">
          <header>
            <div>
              <h2>Live Transcript</h2>
              <p className="subtitle">source of truth, timestamped</p>
              <p className="transcript-lag-meta">
                <span className={`delay-pill delay-${delayTone}`}>realtime gap {formatLagSeconds(transcriptLagSeconds)}</span>
                {committedLagSeconds != null ? ` · commit delay ${formatLagSeconds(committedLagSeconds)}` : ""}
                {elapsedSeconds > 0 ? " · backend clock" : " · frontend estimate"} · latest span {latestSegmentRange}{previewSegment ? " · previewing" : ""}
              </p>
            </div>
            <div className="range">{segments.length} segments</div>
          </header>
          <div
            ref={transcriptBodyRef}
            className="panel-body transcript-body"
            onScroll={(e) => {
              if (transcriptProgrammaticScrollRef.current) {
                return;
              }
              const follow = isNearBottom(e.currentTarget);
              transcriptAutoFollowRef.current = follow;
              setTranscriptAutoFollow(follow);
            }}
          >
            {visibleSegments.length === 0 && (
              <div className="muted">{transcriptEmptyMessage}</div>
            )}
            {visibleSegments.map((s) => {
              const confidenceTone = getConfidenceTone(s.confidence);
              const confidenceLabel = getConfidenceLabel(s.confidence);
              const subtitleText = activeSubtitleLanguage ? s.translations[activeSubtitleLanguage] : null;
              const showSubtitlePlaceholder =
                activeSubtitleLanguage != null &&
                s.original_language !== activeSubtitleLanguage &&
                !(subtitleText && subtitleText.trim());
              return (
                <div key={s.id} className={`segment ${confidenceTone !== "stable" ? `segment-${confidenceTone}` : ""}`.trim()}>
                  <span className="ts">
                    {formatCreatedAt(s.created_at)}
                    {confidenceLabel && <em className={`confidence-flag confidence-${confidenceTone}`}>{confidenceLabel}</em>}
                  </span>
                  <div className="segment-text">
                    <span>{s.text}</span>
                    {subtitleText && subtitleText.trim() && subtitleText.trim() !== s.text.trim() && (
                      <span className="segment-subtitle">{subtitleText}</span>
                    )}
                    {showSubtitlePlaceholder && (
                      <span className="segment-subtitle segment-subtitle-pending" aria-label="translating subtitle">
                        <span className="subtitle-loading-dot" />
                        <span className="subtitle-loading-dot" />
                        <span className="subtitle-loading-dot" />
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
            {previewSegment && (
              <div className="segment preview-segment">
                <span className="ts">hearing…</span>
                <span>{previewSegment.text}</span>
              </div>
            )}
            <div ref={transcriptBottomRef} className="transcript-bottom-anchor" />
            {!transcriptAutoFollow && (
              <button
                type="button"
                className="jump-latest-btn"
                onClick={() => {
                  transcriptAutoFollowRef.current = true;
                  setTranscriptAutoFollow(true);
                  scrollTranscriptToBottom("smooth");
                }}
              >
                Jump to latest
              </button>
            )}
          </div>
        </section>

        <SummaryPanel
          title="Latest Rolling Summary"
          subtitle="most recent 3–5 minutes"
          snapshot={rolling}
          history={historyByType.rolling_summary ?? []}
          sessionStartedAt={sessionStartedAt}
          accent="#2563eb"
          className="summary-panel rolling-panel"
          onSaveToNotes={handleSaveToNotes}
        />

        <SummaryPanel
          title="Current Meeting Summary"
          subtitle="cumulative so far"
          snapshot={cumulative}
          history={historyByType.cumulative_meeting_summary ?? []}
          sessionStartedAt={sessionStartedAt}
          accent="#0f766e"
          className="summary-panel cumulative-panel"
          onSaveToNotes={handleSaveToNotes}
        />

        <section className="panel diagnostics">
          <header>
            <div>
              <h2>Live Diagnostics</h2>
              <p className="subtitle">latency and retry signals</p>
            </div>
            <div className="range">{elapsedSeconds > 0 ? formatElapsedSeconds(elapsedSeconds) : "—"}</div>
          </header>
          <div className="panel-body diagnostics-body">
            <div className="diagnostic-card">
              <span className="diagnostic-label">Active Source</span>
              <strong className="diagnostic-value">{formatSourceLabel(activeSource)}</strong>
              <span className="diagnostic-note">
                {activeSource === "mixed"
                  ? "capturing system audio and microphone"
                  : activeSource === "loopback" || activeSource === "system"
                    ? "capturing system output only"
                    : "capturing microphone only"}
              </span>
            </div>
            <div className={`diagnostic-card gain-card gain-${mixedGainTone}`}>
              <span className="diagnostic-label">Mixed Gain</span>
              <strong className="diagnostic-value">{mixedGainRatio}</strong>
              <span className="diagnostic-note">
                {activeSource === "mixed"
                  ? `mic effective ${formatGain(mixedEffectiveMicGain)} · base ${formatGain(mixedMicGain)} · auto ${mixedAutoBalanceEnabled ? "on" : "off"}`
                  : mixedGainHint}
              </span>
              {activeSource === "mixed" && (
                <span className="diagnostic-note diagnostic-hint">{mixedGainHint}</span>
              )}
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Subtitles</span>
              <strong className="diagnostic-value">{subtitleStatus}</strong>
              <span className="diagnostic-note">{subtitleStatusDetail}</span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Audio Clock</span>
              <strong className="diagnostic-value">{formatElapsedSeconds(audioClockSeconds)}</strong>
              <span className="diagnostic-note">{elapsedSeconds > 0 ? "backend progress" : "frontend estimate"}</span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Committed Transcript</span>
              <strong className="diagnostic-value">{formatElapsedSeconds(visibleTranscriptSeconds)}</strong>
              <span className="diagnostic-note">{latestSegment ? `last committed · ${formatRange(latestSegment.start_time, latestSegment.end_time)}` : "waiting for committed text"}</span>
            </div>
            <div className={`diagnostic-card delay-card delay-${delayTone}`}>
              <span className="diagnostic-label">Realtime Gap</span>
              <strong className="diagnostic-value">{formatLagSeconds(transcriptLagSeconds)}</strong>
              <span className="diagnostic-note">current audio clock minus latest visible transcript</span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Commit Delay</span>
              <strong className="diagnostic-value">{formatLagSeconds(committedLagSeconds)}</strong>
              <span className="diagnostic-note">
                {committedLagSeconds != null
                  ? "reference only · commit time minus segment end"
                  : "waiting for committed text"}
              </span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Retry Summary</span>
              <strong className="diagnostic-value">{retryTotal}</strong>
              <span className="diagnostic-note">improved {retryImproved} · unchanged {retryUnchanged} · diverged {retryDiverged}</span>
            </div>
            <div className={`diagnostic-card delay-card delay-${asrSafeguardActive ? "danger" : (asrRealtimeFactor != null && asrRealtimeFactor > 0.8 ? "warn" : "ok")}`}>
              <span className="diagnostic-label">ASR Realtime</span>
              <strong className="diagnostic-value">{formatRtf(asrRealtimeFactor)}</strong>
              <span className="diagnostic-note">
                audio {formatLagSeconds(asrAudioSeconds)} ? ASR {formatLagSeconds(asrWallSeconds)}
              </span>
            </div>
            <div className={`diagnostic-card delay-card delay-${asrSafeguardActive ? "danger" : "ok"}`}>
              <span className="diagnostic-label">Realtime Safeguard</span>
              <strong className="diagnostic-value">{asrSafeguardActive ? "active" : "clear"}</strong>
              <span className="diagnostic-note">
                {asrSafeguardActive
                  ? (sessionStats?.asr_safeguard_reason ?? "protecting realtime path")
                  : `events ${asrSafeguardEvents}`}
              </span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Weak Voice Rescue</span>
              <strong className="diagnostic-value">{weakRescueAttempts}</strong>
              <span className="diagnostic-note">
                emitted {weakRescueEmitted} · buffer {formatLagSeconds(weakRescueBufferSeconds)}
              </span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Background Queue</span>
              <strong className="diagnostic-value">{translationPendingCount + summaryPendingCount}</strong>
              <span className="diagnostic-note">
                translations {translationPendingCount} ? summaries {summaryPendingCount} ? trims {translationTrimTotal}
              </span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Audio Drops</span>
              <strong className="diagnostic-value">{audioDropTotal}</strong>
              <span className="diagnostic-note">capture queue dropped chunks</span>
            </div>
            <div className="diagnostic-card">
              <span className="diagnostic-label">Backpressure</span>
              <strong className="diagnostic-value">{backpressureAgeSeconds != null ? "recent" : "clear"}</strong>
              <span className="diagnostic-note">
                {backpressureAgeSeconds != null
                  ? `last seen ${formatLagSeconds(backpressureAgeSeconds)} ago at ${formatElapsedSeconds(sessionStats?.last_backpressure_elapsed_seconds)}`
                  : "no recent slow-ASR flush"}
              </span>
            </div>
          </div>
        </section>

        <section className="panel notes">
          <header>
            <div>
              <h2>Notes / Quick Actions</h2>
              <p className="subtitle">manual notes + saved snapshots</p>
            </div>
            <div className="range">{notes.length} saved</div>
          </header>
          <div ref={notesBodyRef} className="panel-body">
            {notes.length === 0 ? (
              <p className="muted">No saved notes yet. Use "Save to notes" on a summary panel.</p>
            ) : (
              <ul className="notes-list">
                {notes.map((n) => (
                  <li key={n.id}>
                    <div className="note-meta">
                      <span>{n.source_type ?? "note"}</span>
                      <span className="muted">
                        {new Date(n.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                    <div>{n.content}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
