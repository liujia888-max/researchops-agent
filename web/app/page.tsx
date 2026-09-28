"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type Step =
  | { kind: "plan"; plan: string[] }
  | { kind: "tool_call"; name: string; arguments: unknown }
  | { kind: "tool_result"; name: string; arguments: unknown; output: string };

type TraceSummary = {
  trace_id?: string;
  llm_calls?: number;
  tool_calls?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  cost_usd?: number;
};

type Metric = {
  name: string;
  value: number | string;
  dataset?: string;
  sigma?: number | string;
};

type Experiment = {
  id: number;
  name: string;
  runs: {
    status: string;
    metrics: Metric[];
  }[];
};

type Doc = {
  doc_id: string;
  chunks: number;
};

type MemoryItem = {
  id: number;
  task: string;
  result: string;
  kind: string;
  created_at: string;
};

type PendingApproval = {
  request_id: string;
  tool_name: string;
  arguments: unknown;
};

// Chat-message model: every turn (live run or recalled history) is one user bubble
// plus one assistant bubble, so the conversation reads like ChatGPT/DeepSeek.
type UserMsg = { id: string; role: "user"; content: string; at: string };

type AssistantMsg = {
  id: string;
  role: "assistant";
  at: string;
  status: "running" | "done" | "error";
  steps: Step[];
  report: string;
  error: string;
  trace: TraceSummary | null;
  langfuseUrl: string;
};

type ChatMsg = UserMsg | AssistantMsg;

function now(): string {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function fmtIso(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 16).replace("T", " ");
  return d.toLocaleString("zh-CN", { hour12: false });
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function inlineMarkdown(s: string): string {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

// Minimal, XSS-safe markdown renderer. Escapes everything first, then re-applies
// a small, safe subset of block + inline elements.
function renderMarkdown(src: string): string {
  const lines = escapeHtml(src).split("\n");
  const out: string[] = [];
  let inTable = false;
  let table: string[] = [];

  const flushTable = () => {
    if (!inTable) return;
    inTable = false;
    const rows = table.map((row) => {
      const cells = row
        .split("|")
        .slice(1, -1)
        .map((c) => c.trim());
      return `<tr>${cells.map((c) => `<td>${inlineMarkdown(c)}</td>`).join("")}</tr>`;
    });
    out.push(`<table><tbody>${rows.join("")}</tbody></table>`);
    table = [];
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (/^\|.*\|$/.test(line) && line.replace(/[|\-\s]/g, "") !== "") {
      if (!inTable) inTable = true;
      if (/^\|[\s:\-|]+\|$/.test(line)) continue; // skip separator row
      table.push(line);
      continue;
    }
    flushTable();
    if (!line) continue;
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      const level = h[1].length;
      out.push(`<h${level}>${inlineMarkdown(h[2])}</h${level}>`);
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      out.push(`<li>${inlineMarkdown(line.replace(/^[-*]\s+/, ""))}</li>`);
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      out.push(`<li>${inlineMarkdown(line.replace(/^\d+\.\s+/, ""))}</li>`);
      continue;
    }
    out.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  flushTable();
  return out.join("");
}

function StepView({ step }: { step: Step }) {
  if (step.kind === "plan") {
    return (
      <div className="step plan">
        <div className="head">
          计划 <span className="tag">planner</span>
        </div>
        <ol>
          {step.plan.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ol>
      </div>
    );
  }
  if (step.kind === "tool_call") {
    return (
      <div className="step tool">
        <div className="head">
          调用工具 <span className="tag">{step.name}</span>
        </div>
        <div className="args">{JSON.stringify(step.arguments)}</div>
      </div>
    );
  }
  return (
    <div className="step tool">
      <div className="head">
        工具结果 <span className="tag">{step.name}</span>
      </div>
      <div className="output">{step.output}</div>
    </div>
  );
}

function TraceCard({ trace, langfuseUrl }: { trace: TraceSummary; langfuseUrl: string }) {
  return (
    <div className="trace-card">
      <div className="trace-grid">
        <div className="cell">
          <div className="k">LLM 调用</div>
          <div className="v">{trace.llm_calls ?? "-"}</div>
        </div>
        <div className="cell">
          <div className="k">工具调用</div>
          <div className="v">{trace.tool_calls ?? "-"}</div>
        </div>
        <div className="cell">
          <div className="k">总 Token</div>
          <div className="v">{trace.total_tokens ?? "-"}</div>
        </div>
        <div className="cell">
          <div className="k">成本</div>
          <div className="v">${(trace.cost_usd ?? 0).toFixed(4)}</div>
        </div>
      </div>
      {langfuseUrl && (
        <p className="muted" style={{ marginTop: 10 }}>
          <a href={langfuseUrl} target="_blank" rel="noreferrer">
            在 Langfuse 查看完整 trace →
          </a>
        </p>
      )}
    </div>
  );
}

function UserBubble({ msg }: { msg: UserMsg }) {
  return (
    <div className="msg user">
      <div className="bubble">
        <div className="meta">{msg.at}</div>
        {msg.content}
      </div>
      <div className="avatar">我</div>
    </div>
  );
}

function AssistantBubble({ msg }: { msg: AssistantMsg }) {
  const toolCalls = msg.steps.filter((s) => s.kind === "tool_call").length;
  return (
    <div className="msg assistant">
      <div className="avatar">🤖</div>
      <div className="bubble">
        <div className="meta">ResearchOps Agent · {msg.at}</div>
        {msg.steps.length > 0 && (
          <details className="steps-box" open={msg.status === "running"}>
            <summary>执行过程 · {toolCalls} 次工具调用</summary>
            {msg.steps.map((s, i) => (
              <StepView key={i} step={s} />
            ))}
          </details>
        )}
        {msg.report ? (
          <div className="report" dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.report) }} />
        ) : msg.status === "running" ? (
          <div className="typing">
            <span className="spinner" />
            {msg.steps.length === 0 ? "正在规划…" : "正在检索 / 执行，稍候…"}
          </div>
        ) : null}
        {msg.error && <div className="error">{msg.error}</div>}
        {msg.trace && <TraceCard trace={msg.trace} langfuseUrl={msg.langfuseUrl} />}
      </div>
    </div>
  );
}

export default function Home() {
  const [task, setTask] = useState("");
  const [maxIterations, setMaxIterations] = useState(10);
  const [langfuse, setLangfuse] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [running, setRunning] = useState(false);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [pending, setPending] = useState<PendingApproval | null>(null);

  // Auto-follow the stream only while the user is already at the bottom; a manual
  // scroll up disables it so reading isn't yanked back down mid-generation.
  const chatRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const idSeq = useRef(0);
  // The assistant bubble the SSE stream is currently writing into (by id, so a
  // history click mid-run can't redirect events into the wrong bubble).
  const activeId = useRef<string | null>(null);

  function onChatScroll() {
    const el = chatRef.current;
    if (!el) return;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }

  useEffect(() => {
    if (stickToBottom.current && chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  function pushMessages(add: ChatMsg[]) {
    setMessages((ms) => [...ms, ...add]);
  }

  function updateActive(updater: (m: AssistantMsg) => AssistantMsg) {
    setMessages((ms) =>
      ms.map((m) =>
        m.role === "assistant" && m.id === activeId.current ? updater(m) : m
      )
    );
  }

  async function refreshExperiments() {
    try {
      const res = await fetch(`${API_BASE}/experiments`);
      if (res.ok) setExperiments(await res.json());
    } catch {
      // backend not reachable — keep last list
    }
  }

  async function refreshDocuments() {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) setDocuments(await res.json());
    } catch {
      // Qdrant not reachable — keep last list
    }
  }

  async function refreshMemories() {
    try {
      const res = await fetch(`${API_BASE}/memory`);
      if (res.ok) setMemories(await res.json());
    } catch {
      // backend not reachable — keep last list
    }
  }

  useEffect(() => {
    refreshExperiments();
    refreshDocuments();
    refreshMemories();
  }, []);

  async function uploadFiles(files: FileList | null) {
    if (!files || files.length === 0 || uploading) return;
    setUploading(true);
    setUploadMsg("");
    const results: string[] = [];
    for (const f of Array.from(files)) {
      const form = new FormData();
      form.append("file", f);
      try {
        const res = await fetch(`${API_BASE}/documents`, { method: "POST", body: form });
        const body = await res.json().catch(() => null);
        if (res.ok && body) {
          results.push(`${body.filename} → ${body.chunks} chunks`);
        } else {
          results.push(`${f.name} 失败：${body?.detail ?? res.status}`);
        }
      } catch (e) {
        results.push(`${f.name} 失败：${e instanceof Error ? e.message : String(e)}`);
      }
    }
    setUploadMsg(results.join("\n"));
    setUploading(false);
    refreshDocuments();
  }

  async function deleteDocument(docId: string) {
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
      });
      if (res.ok) refreshDocuments();
    } catch {
      // keep last list on failure
    }
  }

  async function decideApproval(approve: boolean) {
    if (!pending) return;
    const { request_id } = pending;
    setPending(null);
    try {
      await fetch(`${API_BASE}/approvals/${request_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approve }),
      });
    } catch {
      // backend unreachable — the stream will time out and treat it as rejected
    }
  }

  // Replay a stored Q&A into the conversation area as a user bubble + answer
  // bubble, and also backfill the composer so the question can be followed up.
  function loadMemory(m: MemoryItem) {
    const report = m.result || "（这条记忆只保存了问题，没有报告内容）";
    setTask(m.task);
    stickToBottom.current = true;
    pushMessages([
      { id: `u${++idSeq.current}`, role: "user", content: m.task, at: fmtIso(m.created_at) },
      {
        id: `a${++idSeq.current}`,
        role: "assistant",
        at: fmtIso(m.created_at),
        status: "done",
        steps: [],
        report,
        error: "",
        trace: null,
        langfuseUrl: "",
      },
    ]);
  }

  async function run() {
    const question = task.trim();
    if (!question || running) return;
    setRunning(true);
    setTask("");
    stickToBottom.current = true;

    const uid = `u${++idSeq.current}`;
    const aid = `a${++idSeq.current}`;
    activeId.current = aid;
    pushMessages([
      { id: uid, role: "user", content: question, at: now() },
      {
        id: aid,
        role: "assistant",
        at: now(),
        status: "running",
        steps: [],
        report: "",
        error: "",
        trace: null,
        langfuseUrl: "",
      },
    ]);

    try {
      const res = await fetch(`${API_BASE}/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: question, max_iterations: maxIterations, langfuse }),
      });
      if (!res.ok || !res.body) {
        updateActive((m) => ({ ...m, status: "error", error: `后端返回 ${res.status}` }));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") continue;
          let ev: Record<string, unknown>;
          try {
            ev = JSON.parse(payload);
          } catch {
            continue;
          }
          handleEvent(ev);
        }
      }
    } catch (e) {
      updateActive((m) => ({
        ...m,
        status: "error",
        error: e instanceof Error ? e.message : String(e),
      }));
    } finally {
      updateActive((m) => (m.status === "running" ? { ...m, status: "done" } : m));
      activeId.current = null;
      setRunning(false);
      refreshExperiments();
      refreshMemories();
    }
  }

  function handleEvent(ev: Record<string, unknown>) {
    switch (ev.event) {
      case "plan":
        updateActive((m) => ({
          ...m,
          steps: [...m.steps, { kind: "plan", plan: (ev.plan as string[]) ?? [] }],
        }));
        break;
      case "tool_call":
        updateActive((m) => ({
          ...m,
          steps: [...m.steps, { kind: "tool_call", name: ev.name as string, arguments: ev.arguments }],
        }));
        break;
      case "tool_result":
        updateActive((m) => ({
          ...m,
          steps: [
            ...m.steps,
            {
              kind: "tool_result",
              name: ev.name as string,
              arguments: ev.arguments,
              output: ev.output as string,
            },
          ],
        }));
        break;
      case "report":
        updateActive((m) => ({ ...m, report: ev.report as string, status: "done" }));
        break;
      case "trace":
        updateActive((m) => ({ ...m, trace: ev as unknown as TraceSummary }));
        break;
      case "langfuse":
        updateActive((m) => ({ ...m, langfuseUrl: ev.url as string }));
        break;
      case "pending_approval":
        setPending({
          request_id: ev.request_id as string,
          tool_name: ev.tool_name as string,
          arguments: ev.arguments,
        });
        break;
      case "error":
        updateActive((m) => ({ ...m, status: "error", error: ev.message as string }));
        break;
      case "done":
        updateActive((m) => ({ ...m, status: m.status === "error" ? "error" : "done" }));
        break;
    }
  }

  return (
    <div className="container">
      <header className="brand">
        <h1>ResearchOps Agent</h1>
        <p>一句话任务 → 自主检索 / 提交 / 出报告</p>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <div className="card documents">
            <div className="head" style={{ fontWeight: 600, marginBottom: 10 }}>
              文档库（RAG 检索源）
            </div>
            <div className="row" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
              <label className="btn secondary" style={{ cursor: "pointer" }}>
                {uploading ? "上传中…" : "上传文档"}
                <input
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt,.md,.markdown"
                  style={{ display: "none" }}
                  disabled={uploading}
                  onChange={(e) => {
                    uploadFiles(e.target.files);
                    e.target.value = "";
                  }}
                />
              </label>
              <button className="btn secondary" onClick={refreshDocuments}>
                刷新
              </button>
              <span className="muted">PDF / Word(.docx) / txt / md</span>
            </div>
            {uploadMsg && (
              <pre className="output" style={{ marginTop: 10 }}>
                {uploadMsg}
              </pre>
            )}
            {documents.length === 0 ? (
              <p className="muted" style={{ marginTop: 10 }}>
                暂无已入库文档
              </p>
            ) : (
              <div style={{ marginTop: 10 }}>
                {documents.map((d) => (
                  <div
                    key={d.doc_id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "6px 0",
                      borderBottom: "1px solid var(--border)",
                    }}
                  >
                    <span style={{ wordBreak: "break-all", paddingRight: 8 }}>{d.doc_id}</span>
                    <span style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
                      <span className="muted">{d.chunks} chunks</span>
                      <button
                        className="btn secondary"
                        style={{ padding: "2px 10px", fontSize: 12 }}
                        onClick={() => deleteDocument(d.doc_id)}
                      >
                        删除
                      </button>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card memories">
            <div className="head" style={{ fontWeight: 600, marginBottom: 10 }}>
              历史问答（记忆）
            </div>
            {memories.length === 0 ? (
              <p className="muted" style={{ marginTop: 10 }}>
                暂无历史任务——每跑完一次任务，问题与报告会自动记在这里
              </p>
            ) : (
              <div style={{ marginTop: 10 }}>
                {memories.map((m) => (
                  <button
                    key={m.id}
                    className="memory-item"
                    title={`${fmtIso(m.created_at)} · 点击在对话区查看完整问答`}
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      padding: "6px 4px",
                      border: "none",
                      borderBottom: "1px solid var(--border)",
                      background: "none",
                      cursor: "pointer",
                      fontSize: 13,
                    }}
                    onClick={() => loadMemory(m)}
                  >
                    {m.task.length > 42 ? `${m.task.slice(0, 42)}…` : m.task}
                    <span className="muted" style={{ float: "right", fontSize: 11, marginLeft: 8 }}>
                      {m.kind === "experiment" ? "实验" : m.created_at.slice(0, 10)}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="card experiments">
            <div className="head" style={{ fontWeight: 600, marginBottom: 10 }}>
              历史实验
            </div>
            {experiments.length === 0 ? (
              <p className="muted">暂无实验记录</p>
            ) : (
              experiments.map((exp) => (
                <div className="exp" key={exp.id}>
                  <div className="name">
                    #{exp.id} {exp.name}
                  </div>
                  {exp.runs.map((run, i) => (
                    <div className="run" key={i}>
                      <span className="meta">状态</span>{" "}
                      <span className={`badge ${run.status === "success" ? "ok" : "other"}`}>
                        {run.status}
                      </span>
                      {run.metrics.length > 0 && (
                        <div className="metrics">
                          {run.metrics.map((m, i) => (
                            <span className="m" key={i}>
                              {m.name}
                              {m.sigma != null ? ` σ${m.sigma}` : ""}:{" "}
                              {typeof m.value === "number" ? m.value.toFixed(3) : m.value}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        </aside>

        <main className="main">
          <div className="chat" ref={chatRef} onScroll={onChatScroll}>
            {messages.length === 0 ? (
              <div className="empty">
                <h2>👋 你好，我是 ResearchOps Agent</h2>
                <p>一句话任务 → 自主检索 / 提交 / 出报告</p>
                <p className="muted" style={{ marginTop: 8 }}>
                  例如：复现 Restormer 的 Gaussian Color Blind 在 CBSD68 σ=25 上的结果，并和
                  model_v3_rgb 对比，出报告
                </p>
              </div>
            ) : (
              messages.map((m) =>
                m.role === "user" ? (
                  <UserBubble key={m.id} msg={m} />
                ) : (
                  <AssistantBubble key={m.id} msg={m} />
                )
              )
            )}
          </div>

          <div className="card composer">
            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              onKeyDown={(e) => {
                // Enter sends, Shift+Enter inserts a newline; isComposing guards the
                // Chinese IME so confirming a candidate doesn't fire the run.
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  run();
                }
              }}
              placeholder="输入科研任务，Enter 发送，Shift+Enter 换行"
            />
            <div className="row">
              <button className="btn" onClick={run} disabled={running || !task.trim()}>
                {running ? (
                  <>
                    <span className="spinner" /> 运行中…
                  </>
                ) : (
                  "发送"
                )}
              </button>
              <label>
                最大步数
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value) || 10)}
                  style={{ width: 60, padding: "4px 8px", border: "1px solid var(--border)", borderRadius: 6 }}
                />
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={langfuse}
                  onChange={(e) => setLangfuse(e.target.checked)}
                />
                Langfuse 追踪
              </label>
              <span className="muted">后端 {API_BASE}</span>
            </div>
          </div>
        </main>
      </div>

      {pending && (
        <div className="modal-backdrop">
          <div className="modal">
            <h3>⚠ 需要批准才能执行</h3>
            <p>
              Agent 请求执行危险操作 <code>{pending.tool_name}</code>，是否允许？
            </p>
            <pre className="approval-args">{JSON.stringify(pending.arguments, null, 2)}</pre>
            <div className="modal-actions">
              <button className="btn" onClick={() => decideApproval(true)}>
                批准执行
              </button>
              <button className="btn secondary" onClick={() => decideApproval(false)}>
                拒绝
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
