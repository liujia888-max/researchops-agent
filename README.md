# ResearchOps Agent

面向深度学习实验的科研智能体（AI 应用工程师 / Agent 工程师方向的项目）。

给它一篇论文 PDF 和一句「复现 Restormer 在 CBSD68 σ=25 上的结果，并和 model_v3_rgb 对比，出报告」，Agent 自主完成：任务拆解 → 检索论文库(RAG) → 读代码/历史实验 → 经 MCP 提交任务到远程 GPU → 轮询日志/解析指标 → 落库并生成带引用的对比报告。

> 当前状态：**Phase 1（RAG）✅ · Phase 2（Agent + MCP）✅ · Phase 3 可观测 + 评测 ✅**
>
> - **RAG**：Recall@5=0.857 / Hit@5=1.0 / MRR@5=0.621；RAGAS faithfulness=0.958 / answer_relevancy=0.837（[phase1](docs/phase1-rag-results.md)）
> - **Agent 端到端**：一句话任务 → 自主检索/提交/轮询/落库/出报告；复现 Restormer vs model_v3_rgb 在 CBSD68 σ=25 领先 **+1.81 dB**（[phase2](docs/phase2-repro-results.md)）
> - **Agent 轨迹评测**（golden set 3 任务）：完成率 **1.0** / 答案正确率 **1.0** / 工具召回 **1.0** / 工具精确率 **0.83** / 平均步数 **1.33** / 单任务成本 **~$0.0027** / 平均延迟 **9.7s**
>
> 完整路线见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 能力三角（对齐 2026 年 AI 应用/Agent 岗 JD）

- **RAG**：论文 PDF → 结构感知分块 → bge-m3 双向量(dense+sparse) → Qdrant 混合检索 + RRF → bge-reranker-v2-m3 重排 → 带引用生成
- **Agent**：LangGraph 状态机（Plan → Retrieve → Execute → Monitor → Report）+ human-in-the-loop 审批 + 长短期记忆
- **工程化**：FastAPI + SSE 流式、Docker Compose、可观测（自研 trace + Langfuse 云面板：token/成本/延迟）、RAGAS/golden-set 评测、自研 MCP Server `labops`

## 快速开始

```bash
# 1. 环境（Python 3.12）
conda create -y -p .venv python=3.12 pip
.venv/Scripts/activate

# 2. 安装
pip install -e ".[dev]"

# 3. 跑起来
cp .env.example .env   # 填入 LLM API Key
uvicorn researchops.server.main:app --reload
```

> 更完整的「自己怎么用 / 别人怎么用」分层说明（按外部依赖分层、命令一览、网页版、Docker）见 [docs/USAGE.md](docs/USAGE.md)。

## 网页版（Web App）

后端起好后，再起前端（Next.js + React），浏览器打开 `http://localhost:3000`：

```bash
cd web
npm install
npm run dev        # 开发模式
# 或 npm run build && npm start  生产模式
```

> Windows 本机注意：`node`/`npm` 不在系统 PATH 里（`node.exe` 在 `D:\node.exe`，npm 在 `D:\node_modules\npm\bin`）。两种方式任选：
>
> ```powershell
> # 方式 A：临时加到当前终端 PATH，再照常跑
> $env:Path = "D:\;" + $env:Path
> npm install
> npm run dev
>
> # 方式 B：不碰 PATH，直接指定 node 调 npm-cli
> & "D:\node.exe" "D:\node_modules\npm\bin\npm-cli.js" install
> & "D:\node.exe" "D:\node_modules\npm\bin\npm-cli.js" run dev
> ```

前端默认连 `http://localhost:8000`，如需改后端地址，用环境变量 `NEXT_PUBLIC_API_BASE` 覆盖：

```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
```

界面支持：SSE 流式查看 Agent 每一步（计划 → 工具调用 → 工具结果 → 最终报告）、勾选 Langfuse 追踪、展示本次 Trace 的 token/成本、以及历史实验列表。

## 架构

```
Windows 本地（控制面）              远程 AutoDL GPU（执行面）
┌─────────────────────┐           ┌──────────────────────────┐
│ Next.js 前端 (:3000) │           │ 训练/测试任务 (screen)     │
│ FastAPI + SSE (:8000)│  ──SSH──▶ │ vLLM 开源模型 (supervisord)│
│  ├─ LangGraph Agent │           │ bge-m3 embedding/reranker │
│  ├─ RAG (Qdrant)    │           │ MinerU 论文解析            │
│  └─ labops MCP      │           └──────────────────────────┘
└─────────────────────┘
```

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。
