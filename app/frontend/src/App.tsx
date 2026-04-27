import { useMemo, useState } from "react";
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

interface SummaryPanelProps {
  title: string;
  subtitle: string;
  snapshot: SummarySnapshot | undefined;
  accent: string;
  onSaveToNotes: (snapshot: SummarySnapshot) => Promise<void>;
}

function SummaryPanel({ title, subtitle, snapshot, accent, onSaveToNotes }: SummaryPanelProps) {
  const body = snapshot?.content ?? "waiting for the first snapshot…";
  const range = snapshot
    ? formatRange(snapshot.time_start, snapshot.time_end)
    : "—";

  const onCopy = async () => {
    if (!snapshot) return;
    await navigator.clipboard.writeText(snapshot.content);
  };
  const onExpand = () => {
    if (!snapshot) return;
    alert(`${title} — ${range}\n\n${snapshot.content}`);
  };
  const onSave = async () => {
    if (!snapshot) return;
    await onSaveToNotes(snapshot);
  };

  return (
    <section className="panel" style={{ borderTop: `3px solid ${accent}` }}>
      <header>
        <div>
          <h2>{title}</h2>
          <p className="subtitle">{subtitle}</p>
        </div>
        <div className="range">{range}</div>
      </header>
      <div className="panel-body">{body}</div>
      <footer>
        <button onClick={onCopy} disabled={!snapshot}>Copy</button>
        <button onClick={onExpand} disabled={!snapshot}>Expand</button>
        <button onClick={onSave} disabled={!snapshot}>Save to notes</button>
      </footer>
    </section>
  );
}

export default function App() {
  const [source, setSource] = useState("loopback");
  const [summaryLanguage, setSummaryLanguage] = useState("en");
  const [sessionEnabled, setSessionEnabled] = useState(false);
  const { connected, state, meetingId, segments, latestByType, speakers, notes, lastError, saveNote, stopSession } =
    useSessionSocket({ enabled: sessionEnabled, source, summaryLanguage });

  const rolling = latestByType.rolling_summary;
  const cumulative = latestByType.cumulative_meeting_summary;

  const visibleSegments = useMemo(() => segments.slice(-200), [segments]);

  const handleSaveToNotes = async (snapshot: SummarySnapshot) => {
    await saveNote({
      content: snapshot.content,
      source_type: snapshot.summary_type,
      source_id: snapshot.id,
    });
  };

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
          <select value={source} onChange={(e) => setSource(e.target.value)} disabled={state === "running"}>
            <option value="mic">Microphone</option>
            <option value="loopback">System Audio (Loopback)</option>
          </select>
          <select value={summaryLanguage} onChange={(e) => setSummaryLanguage(e.target.value)} disabled={state === "running"}>
            <option value="en">English</option>
            <option value="zh">中文</option>
            <option value="de">Deutsch</option>
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
            </div>
            <div className="range">{segments.length} segments</div>
          </header>
          <div className="panel-body transcript-body">
            {visibleSegments.length === 0 && (
              <div className="muted">waiting for transcript…</div>
            )}
            {visibleSegments.map((s) => (
              <div key={s.id} className="segment">
                <span className="ts">{formatRange(s.start_time, s.end_time)}</span>
                {s.speaker_id && speakers[s.speaker_id] && (
                  <span
                    className="speaker-label"
                    title="Approximate speaker label; not verified identity."
                  >
                    {speakers[s.speaker_id].inferred_label} (approx.)
                  </span>
                )}
                <span>{s.text}</span>
              </div>
            ))}
          </div>
        </section>

        <SummaryPanel
          title="Latest Rolling Summary"
          subtitle="most recent 3–5 minutes"
          snapshot={rolling}
          accent="#2563eb"
          onSaveToNotes={handleSaveToNotes}
        />

        <SummaryPanel
          title="Current Meeting Summary"
          subtitle="cumulative so far"
          snapshot={cumulative}
          accent="#0f766e"
          onSaveToNotes={handleSaveToNotes}
        />

        <section className="panel notes">
          <header>
            <div>
              <h2>Notes / Speakers / Quick Actions</h2>
              <p className="subtitle">manual notes + session controls</p>
            </div>
            <div className="range">{notes.length} saved · {Object.keys(speakers).length} speakers</div>
          </header>
          <div className="panel-body">
            {Object.keys(speakers).length > 0 && (
              <div className="speakers-list">
                <strong style={{ fontSize: 12 }}>Detected speakers:</strong>
                {Object.values(speakers).map((sp) => (
                  <span key={sp.id} className="speaker-label" style={{ marginLeft: 6 }}>
                    {sp.display_name || `${sp.inferred_label} (approx.)`}
                  </span>
                ))}
              </div>
            )}
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
