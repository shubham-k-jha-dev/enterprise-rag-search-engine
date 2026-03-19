import { useState, useRef, useCallback, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const API_KEY = import.meta.env.VITE_API_KEY ?? "Shubh@2005";

const AUTH_HEADERS = {
  "X-API-Key": API_KEY,
};

type Tab = "search" | "ask" | "upload" | "workspaces" | "metrics";

function generateConversationId(): string {
  return crypto.randomUUID();
}

interface SearchResult {
  text: string;
  source: string;
  chunk_id: number;
  rerank_score: number | null;
}

// All state lifted to parent so it persists when switching tabs
export default function Index() {
  const [tab, setTab] = useState<Tab>("search");

  // Conversation ID — generated once per browser session.
  const [conversationId] = useState<string>(generateConversationId);

  // Search state — persists across tab switches
  const [searchQuery, setSearchQuery] = useState("");
  const [searchWorkspace, setSearchWorkspace] = useState("default");
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");

  // Ask state — persists across tab switches
  const [askQuery, setAskQuery] = useState("");
  const [askWorkspace, setAskWorkspace] = useState("default");
  const [askAnswer, setAskAnswer] = useState("");
  const [askPlaceholder, setAskPlaceholder] = useState(true);
  const [askStreaming, setAskStreaming] = useState(false);

  // Upload state — persists across tab switches
  const [uploadWorkspace, setUploadWorkspace] = useState("default");
  const [uploadStatus, setUploadStatus] = useState<{ type: "loading" | "success" | "error"; msg: string } | null>(null);

  return (
    <div className="nexus-root">
      <header className="nexus-header">
        <div className="nexus-container">
          <nav className="nexus-nav">
            <div className="nexus-logo">
              <div className="nexus-logo-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8" />
                  <path d="m21 21-4.3-4.3" />
                </svg>
              </div>
              NeuralSearch
            </div>
            <div className="nexus-tabs">
              {(["search", "ask", "upload", "workspaces", "metrics"] as Tab[]).map(t => (
                <button key={t} className={`nexus-tab${tab === t ? " active" : ""}`} onClick={() => setTab(t)}>
                  {t === "ask" ? "Ask AI" : t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </nav>
        </div>
      </header>
      <main className="nexus-container nexus-main">
        {/* All views always mounted — hidden with display:none when inactive */}
        {/* This preserves state across tab switches */}
        <div style={{ display: tab === "search" ? "block" : "none" }}>
          <SearchView
            query={searchQuery} setQuery={setSearchQuery}
            workspace={searchWorkspace} setWorkspace={setSearchWorkspace}
            results={searchResults} setResults={setSearchResults}
            loading={searchLoading} setLoading={setSearchLoading}
            error={searchError} setError={setSearchError}
          />
        </div>
        <div style={{ display: tab === "ask" ? "block" : "none" }}>
          <AskView
            query={askQuery} setQuery={setAskQuery}
            workspace={askWorkspace} setWorkspace={setAskWorkspace}
            answer={askAnswer} setAnswer={setAskAnswer}
            placeholder={askPlaceholder} setPlaceholder={setAskPlaceholder}
            streaming={askStreaming} setStreaming={setAskStreaming}
            conversationId={conversationId}
          />
        </div>
        <div style={{ display: tab === "upload" ? "block" : "none" }}>
          <UploadView
            workspace={uploadWorkspace} setWorkspace={setUploadWorkspace}
            status={uploadStatus} setStatus={setUploadStatus}
          />
        </div>
        <div style={{ display: tab === "workspaces" ? "block" : "none" }}>
          <WorkspacesView />
        </div>
        <div style={{ display: tab === "metrics" ? "block" : "none" }}>
          <MetricsView />
        </div>
      </main>
    </div>
  );
}

// ─── Search View ──────────────────────────────────────────────────────────────

interface SearchViewProps {
  query: string; setQuery: (v: string) => void;
  workspace: string; setWorkspace: (v: string) => void;
  results: SearchResult[] | null; setResults: (v: SearchResult[] | null) => void;
  loading: boolean; setLoading: (v: boolean) => void;
  error: string; setError: (v: string) => void;
}

function SearchView({ query, setQuery, workspace, setWorkspace, results, setResults, loading, setLoading, error, setError }: SearchViewProps) {
  const search = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true); setError(""); setResults(null);
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...AUTH_HEADERS },
        body: JSON.stringify({ query, workspace }),
      });
      if (res.status === 403) { setError("Invalid API key."); return; }
      if (res.status === 404) { setError(`Workspace "${workspace}" not found.`); return; }
      if (!res.ok) { setError("Server error. Please try again."); return; }
      const data = await res.json();
      setResults(data.results ?? []);
    } catch {
      setError("Cannot connect to NeuralSearch backend.");
    } finally {
      setLoading(false);
    }
  }, [query, workspace]);

  return (
    <div className="nexus-fade">
      <div className="nexus-input-group">
        <div className="nexus-control grow">
          <label>Search Query</label>
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && search()}
            placeholder="Search across your documents..."
          />
        </div>
        <WorkspaceSelect value={workspace} onChange={setWorkspace} />
        <button className="nexus-btn" onClick={search}>Search</button>
      </div>
      {loading && <div className="nexus-card nexus-loading">Searching...</div>}
      {error && <div className="nexus-card nexus-error">Error: {error}</div>}
      {results && results.length === 0 && <div className="nexus-empty">No results found.</div>}
      {results && results.length > 0 && (
        <div className="nexus-results">
          {results.map((r, i) => (
            <div key={i} className="nexus-card nexus-result">
              <div className="nexus-result-header">
                <span className="nexus-source-tag">{r.source}</span>
                <span className="nexus-score">
                  ID: {r.chunk_id} · Relevance: {r.rerank_score !== null ? r.rerank_score.toFixed(4) : "N/A"}
                </span>
              </div>
              <div className="nexus-chunk">
                <TruncatedText text={r.text} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Ask View ─────────────────────────────────────────────────────────────────

interface AskViewProps {
  query: string; setQuery: (v: string) => void;
  workspace: string; setWorkspace: (v: string) => void;
  answer: string; setAnswer: (v: string) => void;
  placeholder: boolean; setPlaceholder: (v: boolean) => void;
  streaming: boolean; setStreaming: (v: boolean) => void;
  conversationId: string;
}

function AskView({ query, setQuery, workspace, setWorkspace, answer, setAnswer, placeholder, setPlaceholder, streaming, setStreaming, conversationId }: AskViewProps) {
  const ask = useCallback(async () => {
    if (!query.trim()) return;
    setPlaceholder(false); setAnswer(""); setStreaming(true);
    try {
      const url = `${API_BASE}/ask?query=${encodeURIComponent(query)}`
        + `&workspace=${encodeURIComponent(workspace)}`
        + `&conversation_id=${encodeURIComponent(conversationId)}`;

      const res = await fetch(url, { headers: AUTH_HEADERS });
      if (res.status === 403) { setAnswer("Invalid API key."); return; }
      if (res.status === 404) { setAnswer(`Workspace "${workspace}" not found.`); return; }
      if (!res.body) { setAnswer("No streaming support."); return; }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let text = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
        setAnswer(text);
      }
    } catch {
      setAnswer("Cannot connect to NeuralSearch backend.");
    } finally {
      setStreaming(false);
    }
  }, [query, workspace, conversationId]);

  return (
    <div className="nexus-fade">
      <div className="nexus-input-group">
        <div className="nexus-control grow">
          <label>Ask a Question</label>
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && ask()}
            placeholder="What would you like to know?"
          />
        </div>
        <WorkspaceSelect value={workspace} onChange={setWorkspace} />
        <button className="nexus-btn" onClick={ask} disabled={streaming}>
          {streaming ? "Thinking..." : "Ask AI"}
        </button>
      </div>
      <div className={`nexus-answer${placeholder ? " placeholder" : ""}`}>
        {placeholder ? "AI-generated answers will appear here..." : answer}
        {streaming && <span className="nexus-cursor">|</span>}
      </div>
    </div>
  );
}

// ─── Upload View ──────────────────────────────────────────────────────────────

interface UploadViewProps {
  workspace: string; setWorkspace: (v: string) => void;
  status: { type: "loading" | "success" | "error"; msg: string } | null;
  setStatus: (v: { type: "loading" | "success" | "error"; msg: string } | null) => void;
}

function UploadView({ workspace, setWorkspace, status, setStatus }: UploadViewProps) {
  const fileRef = useRef<HTMLInputElement>(null);

  const upload = useCallback(async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setStatus({ type: "loading", msg: "Uploading and indexing..." });
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload?workspace=${workspace}`, {
        method: "POST",
        headers: AUTH_HEADERS,
        body: form,
      });
      const data = await res.json();
      if (res.status === 403) {
        setStatus({ type: "error", msg: "Invalid API key." });
      } else if (res.status === 404) {
        setStatus({ type: "error", msg: `Workspace "${workspace}" not found. Create it first.` });
      } else if (res.ok) {
        setStatus({ type: "success", msg: `${data.message} — ${data.chunks_created} chunks created` });
      } else {
        throw new Error(data.detail ?? "Upload failed");
      }
    } catch (e: any) {
      setStatus({ type: "error", msg: `Error: ${e.message}` });
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  }, [workspace]);

  return (
    <div className="nexus-fade">
      <div className="nexus-card nexus-upload-card">
        <div className="nexus-control" style={{ marginBottom: "1.5rem" }}>
          <label>Target Workspace</label>
          <WorkspaceSelect value={workspace} onChange={setWorkspace} />
        </div>
        <div className="nexus-upload-zone" onClick={() => fileRef.current?.click()}>
          <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p style={{ fontWeight: 600, marginBottom: 4 }}>Click to upload a document</p>
          <p className="nexus-sub">PDF, TXT, or DOCX</p>
          <input ref={fileRef} type="file" accept=".pdf,.txt,.docx" style={{ display: "none" }} onChange={upload} />
        </div>
        {status && <div className={`nexus-status ${status.type}`}>{status.msg}</div>}
      </div>
    </div>
  );
}

// ─── Metrics View ─────────────────────────────────────────────────────────────

function MetricsView() {
  const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/metrics`, { headers: AUTH_HEADERS });
        if (!res.ok) { setError("Failed to load metrics."); return; }
        setMetrics(await res.json());
      } catch {
        setError("Cannot connect to NeuralSearch backend.");
      }
    })();
  }, []);

  return (
    <div className="nexus-fade">
      {!metrics && !error && <div className="nexus-card nexus-loading">Loading metrics...</div>}
      {error && <div className="nexus-card nexus-error">{error}</div>}
      {metrics && (
        <div className="nexus-metrics-grid">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key} className="nexus-card nexus-metric">
              <div className="nexus-metric-label">{key.replace(/_/g, " ")}</div>
              <div className="nexus-metric-value">
                {typeof value === "number" ? value.toLocaleString() : String(value)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Workspaces View (Phase D) ────────────────────────────────────────────────

function WorkspacesView() {
  const [workspaces, setWorkspaces] = useState<string[]>([]);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [statusMsg, setStatusMsg] = useState("");
  const [statusType, setStatusType] = useState<"success" | "error" | "">("");

  const DEFAULT_WS = new Set(["default", "got", "dexter"]);

  const loadWorkspaces = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/workspaces`);
      if (!res.ok) return;
      const data = await res.json();
      setWorkspaces(data.workspaces ?? []);
    } catch {
      setStatusMsg("Cannot connect to backend.");
      setStatusType("error");
    }
  }, []);

  useEffect(() => {
    loadWorkspaces();
  }, [loadWorkspaces]);

  const createWorkspace = useCallback(async () => {
    if (!newName.trim()) return;
    setStatusMsg(""); setStatusType("");
    try {
      const res = await fetch(`${API_BASE}/workspaces`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...AUTH_HEADERS },
        body: JSON.stringify({ name: newName.trim(), description: newDescription.trim() }),
      });
      const data = await res.json();
      if (res.status === 201) {
        setStatusMsg(`Workspace "${newName}" created successfully.`);
        setStatusType("success");
        setNewName("");
        setNewDescription("");
        loadWorkspaces();
      } else if (res.status === 409) {
        setStatusMsg(`Workspace "${newName}" already exists.`);
        setStatusType("error");
      } else {
        setStatusMsg(data.detail ?? "Failed to create workspace.");
        setStatusType("error");
      }
    } catch {
      setStatusMsg("Cannot connect to backend.");
      setStatusType("error");
    }
  }, [newName, newDescription, loadWorkspaces]);

  const deleteWorkspace = useCallback(async (name: string) => {
    if (!window.confirm(`Delete workspace "${name}"? This is permanent.`)) return;
    setStatusMsg(""); setStatusType("");
    try {
      const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(name)}`, {
        method: "DELETE",
        headers: AUTH_HEADERS,
      });
      const data = await res.json();
      if (res.ok) {
        setStatusMsg(`Workspace "${name}" deleted.`);
        setStatusType("success");
        loadWorkspaces();
      } else {
        setStatusMsg(data.detail ?? "Failed to delete workspace.");
        setStatusType("error");
      }
    } catch {
      setStatusMsg("Cannot connect to backend.");
      setStatusType("error");
    }
  }, [loadWorkspaces]);

  return (
    <div className="nexus-fade">

      {/* Create Workspace */}
      <div className="nexus-card" style={{ marginBottom: "1.5rem" }}>
        <h3 style={{ marginBottom: "1rem", fontWeight: 600 }}>Create Workspace</h3>
        <div className="nexus-input-group">
          <div className="nexus-control grow">
            <label>Name</label>
            <input
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => e.key === "Enter" && createWorkspace()}
              placeholder="e.g. ca-finals, class8-science"
            />
          </div>
          <div className="nexus-control grow">
            <label>Description (optional)</label>
            <input
              value={newDescription}
              onChange={e => setNewDescription(e.target.value)}
              placeholder="e.g. CA Final year exam notes"
            />
          </div>
          <button className="nexus-btn" onClick={createWorkspace}>Create</button>
        </div>
        <p style={{ fontSize: "0.8rem", color: "#888", marginTop: "0.5rem" }}>
          Name rules: lowercase letters, digits, hyphens only. No spaces.
        </p>
        {statusMsg && (
          <div className={`nexus-status ${statusType}`} style={{ marginTop: "0.75rem" }}>
            {statusMsg}
          </div>
        )}
      </div>

      {/* Workspace List */}
      <div className="nexus-card">
        <h3 style={{ marginBottom: "1rem", fontWeight: 600 }}>
          All Workspaces ({workspaces.length})
        </h3>
        {workspaces.length === 0 && (
          <div className="nexus-empty">Loading workspaces...</div>
        )}
        {workspaces.map(ws => (
          <div key={ws} className="nexus-result-header" style={{ padding: "0.75rem 0", borderBottom: "1px solid #eee" }}>
            <span>
              <strong>{ws}</strong>
              {DEFAULT_WS.has(ws) && (
                <span style={{ marginLeft: "0.5rem", fontSize: "0.75rem", color: "#888" }}>
                  (default — protected)
                </span>
              )}
            </span>
            {!DEFAULT_WS.has(ws) && (
              <button
                onClick={() => deleteWorkspace(ws)}
                style={{
                  background: "none",
                  border: "1px solid #e53e3e",
                  color: "#e53e3e",
                  borderRadius: "4px",
                  padding: "0.2rem 0.6rem",
                  cursor: "pointer",
                  fontSize: "0.8rem",
                }}
              >
                Delete
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Shared Components ────────────────────────────────────────────────────────

function TruncatedText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const limit = 300;
  if (text.length <= limit) return <span>{text}</span>;
  return (
    <span>
      {expanded ? text : text.slice(0, limit) + "..."}
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          marginLeft: "8px",
          background: "none",
          border: "none",
          color: "#3b82f6",
          cursor: "pointer",
          fontSize: "0.85rem",
          padding: 0,
          textDecoration: "underline",
        }}
      >
        {expanded ? "Show less" : "Show more"}
      </button>
    </span>
  );
}

interface WorkspaceSelectProps {
  value: string;
  onChange: (v: string) => void;
}

function WorkspaceSelect({ value, onChange }: WorkspaceSelectProps) {
  const [workspaces, setWorkspaces] = useState<string[]>(["default"]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/workspaces`);
        if (!res.ok) return;
        const data = await res.json();
        setWorkspaces(data.workspaces ?? ["default"]);
      } catch {
        // Network error — keep the fallback ["default"] list.
      }
    })();
  }, []);

  return (
    <div className="nexus-control">
      <label>Workspace</label>
      <select value={value} onChange={e => onChange(e.target.value)}>
        {workspaces.map(ws => (
          <option key={ws} value={ws}>{ws}</option>
        ))}
      </select>
    </div>
  );
}