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
  const bottomRef = useRef<HTMLDivElement>(null);

  async function refreshExperiments() {
    try {
      const res = await fetch(`${API_BASE}/experiments`);
      if (res.ok) setExperiments(await res.json());
    } catch {
      // backend not reachable — keep last list
    }
  }

  useEffect(() => {
    refreshExperiments();
  }, []);

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

      <div ref={bottomRef} />
    </div>
  );
}
