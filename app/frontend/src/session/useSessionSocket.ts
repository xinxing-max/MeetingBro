import { useCallback, useEffect, useRef, useState } from "react";
import type {
  Note,
  SessionEvent,
  SessionState,
  Speaker,
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
  source?: string;           // "mic" | "loopback" | "system"
  summaryLanguage?: string;  // "en" | "zh" | "de"
  forcedLanguage?: string;   // "en" | "zh" | "de" or undefined for auto-detect
}

export interface SessionView {
  connected: boolean;
  state: SessionState | "disconnected";
  meetingId: string | null;
  segments: TranscriptSegment[];
  latestByType: Partial<Record<SummaryType, SummarySnapshot>>;
  speakers: Record<string, Speaker>;
  notes: Note[];
  lastError: string | null;
  saveNote: (input: SaveNoteInput) => Promise<void>;
  stopSession: () => void;
}

export function useSessionSocket(options: SessionOptions = {}): SessionView {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SessionView["state"]>("disconnected");
  const [meetingId, setMeetingId] = useState<string | null>(null);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [latestByType, setLatestByType] = useState<SessionView["latestByType"]>({});
  const [speakers, setSpeakers] = useState<Record<string, Speaker>>({});
  const [notes, setNotes] = useState<Note[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const enabled = options.enabled ?? true;
  const source = options.source ?? "loopback";
  const summaryLanguage = options.summaryLanguage ?? "en";
  const forcedLanguage = options.forcedLanguage;

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnected(false);
      setState("disconnected");
      return;
    }

    setLastError(null);
    setSegments([]);
    setLatestByType({});
    setSpeakers({});
    setNotes([]);
    setMeetingId(null);
    setState("starting");

    const base = window.meetingbro?.backendWs ?? "ws://127.0.0.1:8765";
    const params = new URLSearchParams({ source, summary_language: summaryLanguage });
    if (forcedLanguage) params.set("forced_language", forcedLanguage);
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
            break;
          case "transcript_segment":
            setSegments((prev) => [...prev, msg.payload]);
            break;
          case "summary_snapshot":
            setLatestByType((prev) => ({
              ...prev,
              [msg.payload.summary_type]: msg.payload,
            }));
            break;
          case "speaker_update":
            setSpeakers((prev) => ({
              ...prev,
              [msg.payload.id]: msg.payload,
            }));
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
      ws.close();
    };
  }, [enabled, source, summaryLanguage, forcedLanguage]);

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
    segments,
    latestByType,
    speakers,
    notes,
    lastError,
    saveNote,
    stopSession,
  };
}
