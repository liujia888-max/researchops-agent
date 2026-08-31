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

export default function Home() {
  const [task, setTask] = useState("");
  const [maxIterations, setMaxIterations] = useState(10);
  const [langfuse, setLangfuse] = useState(false);
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<Step[]>([]);
  const [report, setReport] = useState("");
  const [trace, setTrace] = useState<TraceSummary | null>(null);
  const [langfuseUrl, setLangfuseUrl] = useState("");
  const [error, setError] = useState("");
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    refreshExperiments();
    refreshDocuments();
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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps, report]);

  async function run() {
    if (!task.trim() || running) return;
    setRunning(true);
    setError("");
    setSteps([]);
    setReport("");
    setTrace(null);
    setLangfuseUrl("");

    try {
      const res = await fetch(`${API_BASE}/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task, max_iterations: maxIterations, langfuse }),
      });
      if (!res.ok || !res.body) {
        setError(`后端返回 ${res.status}`);
        setRunning(false);
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
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      refreshExperiments();
    }
  }

  function handleEvent(ev: Record<string, unknown>) {
    switch (ev.event) {
      case "plan":
        setSteps((s) => [...s, { kind: "plan", plan: (ev.plan as string[]) ?? [] }]);
        break;
      case "tool_call":
        setSteps((s) => [
          ...s,
          { kind: "tool_call", name: ev.name as string, arguments: ev.arguments },
        ]);
        break;
      case "tool_result":
        setSteps((s) => [
          ...s,
          {
            kind: "tool_result",
            name: ev.name as string,
            arguments: ev.arguments,
            output: ev.output as string,
          },
        ]);
        break;
      case "report":
        setReport(ev.report as string);
        break;
      case "trace":
        setTrace(ev as unknown as TraceSummary);
        break;
      case "langfuse":
        setLangfuseUrl(ev.url as string);
        break;
      case "error":
        setError(ev.message as string);
        break;
    }
  }

  return (
    <div className="container">
      <header className="brand">
        <h1>ResearchOps Agent</h1>
        <p>一句话任务 → 自主检索 / 提交 / 出报告</p>
      </header>

      <div className="card composer">
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="例如：复现 Restormer 的 Gaussian Color Blind 在 CBSD68 σ=25 上的结果，并和 model_v3_rgb 对比，出报告"
        />
        <div className="row">
          <button className="btn" onClick={run} disabled={running || !task.trim()}>
            {running ? (
              <>
                <span className="spinner" /> 运行中…
              </>
            ) : (
              "开始"
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

      {error && <div className="error">{error}</div>}

      {steps.length > 0 && (
        <div className="card">
          {steps.map((s, i) => (
            <StepView key={i} step={s} />
          ))}
        </div>
      )}

      {report && (
        <div className="card">
          <div className="head" style={{ fontWeight: 600, marginBottom: 8 }}>
            最终报告
          </div>
          <div className="report" dangerouslySetInnerHTML={{ __html: renderMarkdown(report) }} />
        </div>
      )}

      {trace && (
        <div className="card">
          <div className="head" style={{ fontWeight: 600, marginBottom: 10 }}>
            本次 Trace
          </div>
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
      )}

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
          <span className="muted">支持 PDF / Word(.docx) / txt / md</span>
        </div>
        {uploadMsg && (
          <pre className="output" style={{ marginTop: 10 }}>
            {uploadMsg}
          </pre>
        )}
        {documents.length === 0 ? (
          <p className="muted" style={{ marginTop: 10 }}>
            暂无已入库文档（上传后这里会列出，并成为 Agent 检索的语料）
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
                <span>{d.doc_id}</span>
                <span style={{ display: "flex", alignItems: "center", gap: 12 }}>
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

      <div ref={bottomRef} />
    </div>
  );
}
