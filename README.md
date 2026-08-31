# ResearchOps Agent

面向深度学习实验的科研智能体（AI 应用工程师 / Agent 工程师方向的项目）。

给它一篇论文 PDF 和一句「复现 Restormer 在 CBSD68 σ=25 上的结果，并和 model_v3_rgb 对比，出报告」，Agent 自主完成：任务拆解 → 检索论文库(RAG) → 读代码/历史实验 → 经 MCP 提交任务到远程 GPU → 轮询日志/解析指标 → 落库并生成带引用的对比报告。

> 当前状态：**Phase 1（RAG）✅ · Phase 2（Agent + MCP）✅ · Phase 3 可观测 + 评测 ✅**
>
> - **RAG**：Recall@5=0.857 / Hit@5=1.0 / MRR@5=0.621；RAGAS faithfulness=0.958 / answer_relevancy=0.837（[phase1](docs/phase1-rag-results.md)）
> - **Agent 端到端**：一句话任务 → 自主检索/提交/轮询/落库/出报告；复现 Restormer vs model_v3_rgb 在 CBSD68 σ=25 领先 **+1.81 dB**（[phase2](docs/phase2-repro-results.md)）
> - **Agent 轨迹评测**（golden set 18 任务，LLM-judge）：完成率 **1.0** / 答案正确率 **0.889** / 工具召回 **1.0** / 工具精确率 **0.778** / 平均步数 **2.06** / 单任务成本 **~$0.0042** / 平均延迟 **14.1s**（[phase3](docs/phase3-agent-eval-results.md)）
>
> 完整路线见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 能力三角（对齐 2026 年 AI 应用/Agent 岗 JD）

- **RAG**：论文 PDF → 结构感知分块 → bge-m3 双向量(dense+sparse) → Qdrant 混合检索 + RRF → bge-reranker-v2-m3 重排 → 带引用生成
- **Agent**：LangGraph 状态机（Plan → Retrieve → Execute → Monitor → Report）+ human-in-the-loop 审批 + 长短期记忆
- **工程化**：FastAPI + SSE 流式、Docker Compose、可观测（自研 trace + Langfuse 云面板：token/成本/延迟）、RAGAS/golden-set 评测、自研 MCP Server `labops`

## 快速开始

### 一键跑通（Docker Compose，无 GPU 也能跑）

```bash
git clone https://github.com/liujia888-max/researchops-agent.git
cd researchops-agent
cp .env.example .env        # 填入一个 LLM API Key（DeepSeek / Qwen 二选一）
docker compose up --build
```

起好后：

- **网页版**：`http://localhost:3000`（上传文档 → 输入任务 → 看流式报告）
- **API 文档**：`http://localhost:8000/docs`
- **Langfuse 面板**：`http://localhost:3001`（需在 `.env` 填 Langfuse key）

一键灌入示例文档，立刻体验 RAG 问答：

```bash
docker compose exec api researchops ingest examples/sample_document.md
docker compose exec api researchops search "denoising method"
```

> **为什么没有 GPU 也能跑通 RAG？** 默认 `RAG_FALLBACK_LOCAL=true`：embedding/reranker
> 推理服务不可达时，自动退化为零依赖的 feature-hash 嵌入（词法检索），链路不报错。
> 想要语义检索质量，把 `INFERENCE_BASE_URL` 指向一台跑着 `scripts/inference_server.py`
> 的主机即可（bge-m3 + reranker）。GPU 在本项目里始终是**可选资源**。

### 本地开发路径（conda + uvicorn）

```bash
# 1. 环境（Python 3.12）
conda create -y -p .venv python=3.12 pip
.venv/Scripts/activate

# 2. 安装
pip install -e ".[dev]"

# 3. 自检外部依赖（LLM / Qdrant / embedding / GPU 主机逐项报告）
cp .env.example .env        # 填入 LLM API Key
researchops doctor

# 4. 跑起来
uvicorn researchops.server.main:app --reload
```

> 更完整的「自己怎么用 / 别人怎么用」分层说明（按外部依赖分层、命令一览、网页版、Docker）见 [docs/USAGE.md](docs/USAGE.md)。

## 网页版（Web App）

> `docker compose up` 已内置前端（`http://localhost:3000`）。下面是自己手动起前端的方式。

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
