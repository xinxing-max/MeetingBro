import { useCallback, useEffect, useRef, useState } from "react";
import type {
  Note,
  SessionEvent,
  SessionStatePayload,
  SessionState,
  SummarySnapshot,
  SummaryType,
  TranscriptSegment,
} from "../types";

const DEFAULT_WS = "ws://127.0.0.1:8765/ws/session";

declare global {
  interface Window {
    meetingbro?: { backendHttp: string; backendWs: string };
  }
}

export interface SaveNoteInput {
  content: string;
  source_type?: string;
  source_id?: string;
}

export interface SessionOptions {
  enabled?: boolean;
  source?: string;           // "mic" | "loopback" | "system" | "mixed"
  summaryLanguage?: string;  // "en" | "zh" | "de"
  speechLanguage?: string;   // "auto" | "en" | "zh" | "de"
  subtitleLanguage?: string; // "off" | "en" | "zh" | "de"
}

export interface SessionView {
  connected: boolean;
  state: SessionState | "disconnected";
  meetingId: string | null;
  sessionStartedAt: string | null;
  elapsedSeconds: number;
  sessionStats: SessionStatePayload | null;
  segments: TranscriptSegment[];
  previewSegment: TranscriptSegment | null;
  latestByType: Partial<Record<SummaryType, SummarySnapshot>>;
  historyByType: Partial<Record<SummaryType, SummarySnapshot[]>>;
  notes: Note[];
  lastError: string | null;
  saveNote: (input: SaveNoteInput) => Promise<void>;
  stopSession: () => void;
}

export function useSessionSocket(options: SessionOptions = {}): SessionView {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SessionView["state"]>("disconnected");
  const [meetingId, setMeetingId] = useState<string | null>(null);
  const [sessionStartedAt, setSessionStartedAt] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sessionStats, setSessionStats] = useState<SessionStatePayload | null>(null);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [previewSegment, setPreviewSegment] = useState<TranscriptSegment | null>(null);
  const [latestByType, setLatestByType] = useState<SessionView["latestByType"]>({});
  const [historyByType, setHistoryByType] = useState<SessionView["historyByType"]>({});
  const [notes, setNotes] = useState<Note[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const previewTimerRef = useRef<number | null>(null);
  const previewHoldTimerRef = useRef<number | null>(null);
  const queuedPreviewRef = useRef<TranscriptSegment | null>(null);
  const previewSegmentRef = useRef<TranscriptSegment | null>(null);

  const enabled = options.enabled ?? true;
  const source = options.source ?? "loopback";
  const summaryLanguage = options.summaryLanguage ?? "en";
  const speechLanguage = options.speechLanguage ?? "auto";
  const subtitleLanguage = options.subtitleLanguage ?? "off";

  useEffect(() => {
    previewSegmentRef.current = previewSegment;
  }, [previewSegment]);

  useEffect(() => {
    const clearQueuedPreview = () => {
      if (previewTimerRef.current != null) {
        window.clearTimeout(previewTimerRef.current);
        previewTimerRef.current = null;
      }
      queuedPreviewRef.current = null;
    };

    const clearPreviewHold = () => {
      if (previewHoldTimerRef.current != null) {
        window.clearTimeout(previewHoldTimerRef.current);
        previewHoldTimerRef.current = null;
      }
    };

    const holdCommittedPreview = (segment: TranscriptSegment) => {
      clearPreviewHold();
      setPreviewSegment(segment);
      previewHoldTimerRef.current = window.setTimeout(() => {
        setPreviewSegment((prev) => {
          if (!prev) return prev;
          if (prev.id !== segment.id) return prev;
          return null;
        });
        previewHoldTimerRef.current = null;
      }, 320);
    };

    const schedulePreviewUpdate = (nextPreview: TranscriptSegment | null) => {
      if (nextPreview == null) {
        clearQueuedPreview();
        setPreviewSegment(null);
        return;
      }

      const currentPreview = previewSegmentRef.current;
      const currentText = currentPreview?.text.trim() ?? "";
      const nextText = nextPreview.text.trim();
      const currentWindow = currentPreview
        ? `${currentPreview.start_time}:${currentPreview.end_time}`
        : "";
      const nextWindow = `${nextPreview.start_time}:${nextPreview.end_time}`;

      if (currentText === nextText && currentWindow === nextWindow) {
        return;
      }

      if (queuedPreviewRef.current?.id === nextPreview.id && queuedPreviewRef.current.text === nextPreview.text) {
        return;
      }

      clearQueuedPreview();

      const sharedPrefixLength = (() => {
        const max = Math.min(currentText.length, nextText.length);
        let index = 0;
        while (index < max && currentText[index] === nextText[index]) {
          index += 1;
        }
        return index;
      })();
      const sharesMostContent =
        currentText.length > 0 &&
        sharedPrefixLength >= Math.floor(Math.min(currentText.length, nextText.length) * 0.7);
      const updateDelayMs = sharesMostContent ? 140 : 220;

      queuedPreviewRef.current = nextPreview;
      previewTimerRef.current = window.setTimeout(() => {
        setPreviewSegment(nextPreview);
        previewTimerRef.current = null;
        queuedPreviewRef.current = null;
      }, updateDelayMs);
    };

    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      clearQueuedPreview();
        clearPreviewHold();
      setConnected(false);
      setState("disconnected");
      return;
    }

    setLastError(null);
    setSegments([]);
    setElapsedSeconds(0);
    setSessionStats(null);
    clearQueuedPreview();
    clearPreviewHold();
    setPreviewSegment(null);
    setLatestByType({});
    setHistoryByType({});
    setNotes([]);
    setMeetingId(null);
    setSessionStartedAt(null);
    setState("starting");

    const base = window.meetingbro?.backendWs ?? "ws://127.0.0.1:8765";
    const params = new URLSearchParams({ source, summary_language: summaryLanguage });
    if (speechLanguage !== "auto") params.set("forced_language", speechLanguage);
    const url = `${base}/ws/session?${params.toString()}`;
    const ws = new WebSocket(url.startsWith("ws") ? url : DEFAULT_WS);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setState("disconnected");
    };
    ws.onerror = () => setLastError("websocket error - check that the backend is running and the selected audio source is available");
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as SessionEvent;
        switch (msg.type) {
          case "session_state":
            setState(msg.payload.state);
            setMeetingId(msg.payload.meeting_id);
            setElapsedSeconds(msg.payload.elapsed_seconds);
            setSessionStats(msg.payload);
            if (msg.payload.state === "running") {
              setSessionStartedAt((prev) => prev ?? new Date().toISOString());
            }
            break;
          case "transcript_segment":
            clearQueuedPreview();
            clearPreviewHold();
            setPreviewSegment((prev) => {
              if (!prev) return prev;
              const matchesCommitted =
                prev.text.trim() === msg.payload.text.trim() &&
                Math.abs(prev.start_time - msg.payload.start_time) < 0.01 &&
                Math.abs(prev.end_time - msg.payload.end_time) < 0.01;
              if (matchesCommitted) {
                holdCommittedPreview(msg.payload);
                return msg.payload;
              }
              return prev.start_time <= msg.payload.end_time ? null : prev;
            });
            setSegments((prev) => [...prev, msg.payload]);
            break;
          case "transcript_translation":
            setSegments((prev) => prev.map((segment) => {
              if (segment.id !== msg.payload.segment_id) {
                return segment;
              }
              return {
                ...segment,
                translations: {
                  ...segment.translations,
                  [msg.payload.language]: msg.payload.text,
                },
              };
            }));
            break;
          case "transcript_preview":
            schedulePreviewUpdate(msg.payload.segment);
            break;
          case "summary_snapshot":
            setLatestByType((prev) => ({
              ...prev,
              [msg.payload.summary_type]: msg.payload,
            }));
            setHistoryByType((prev) => {
              const existing = prev[msg.payload.summary_type] ?? [];
              if (existing.some((snap) => snap.id === msg.payload.id)) {
                return prev;
              }
              return {
                ...prev,
                [msg.payload.summary_type]: [...existing, msg.payload],
              };
            });
            break;
          case "note_saved":
            setNotes((prev) => [...prev, msg.payload]);
            break;
          case "error":
            setLastError(msg.payload.message);
            break;
        }
      } catch (e) {
        setLastError(`parse error: ${(e as Error).message}`);
      }
    };

    return () => {
      clearQueuedPreview();
      clearPreviewHold();
      ws.close();
    };
  }, [enabled]);

  useEffect(() => {
    const ws = wsRef.current;
    if (!enabled || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    ws.send(
      JSON.stringify({
        type: "update_settings",
        payload: {
          source,
          summary_language: summaryLanguage,
          forced_language: speechLanguage,
          subtitle_language: subtitleLanguage,
        },
      }),
    );
  }, [connected, enabled, source, speechLanguage, subtitleLanguage, summaryLanguage]);

  const stopSession = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  }, []);

  const saveNote = useCallback(
    async (input: SaveNoteInput) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !meetingId) {
        setLastError("cannot save note — session not ready");
        return;
      }
      ws.send(
        JSON.stringify({
          type: "save_note",
          payload: {
            content: input.content,
            source_type: input.source_type,
            source_id: input.source_id,
          },
        }),
      );
    },
    [meetingId],
  );

  return {
    connected,
    state,
    meetingId,
    sessionStartedAt,
    elapsedSeconds,
    sessionStats,
    segments,
    previewSegment,
    latestByType,
    historyByType,
    notes,
    lastError,
    saveNote,
    stopSession,
  };
}
