# ResearchOps Agent

面向深度学习实验的科研智能体（AI 应用工程师 / Agent 工程师方向的项目）。

给它一篇论文 PDF 和一句「复现 Restormer 在 CBSD68 σ=25 上的结果，并和 model_v3_rgb 对比，出报告」，Agent 自主完成：任务拆解 → 检索论文库(RAG) → 读代码/历史实验 → 经 MCP 提交任务到远程 GPU → 轮询日志/解析指标 → 落库并生成带引用的对比报告。

> 当前状态：**Phase 1 —— RAG 子系统已完成**（解析 → 结构分块 → 表格结构化解析 → 混合检索 + RRF + 重排 → 带引用生成 + grounding；golden set 评测 Recall@5=0.857 / Hit@5=1.0 / MRR@5=0.621 / 表格行召回@5=0.50，RAGAS faithfulness=0.958 / answer_relevancy=0.837）。详见 [docs/phase1-rag-results.md](docs/phase1-rag-results.md)。
> 完整路线见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 能力三角（对齐 2026 年 AI 应用/Agent 岗 JD）

- **RAG**：论文 PDF → 结构感知分块 → bge-m3 双向量(dense+sparse) → Qdrant 混合检索 + RRF → bge-reranker-v2-m3 重排 → 带引用生成
- **Agent**：LangGraph 状态机（Plan → Retrieve → Execute → Monitor → Report）+ human-in-the-loop 审批 + 长短期记忆
- **工程化**：FastAPI + SSE 流式、Docker Compose、Langfuse 可观测、RAGAS/golden-set 评测、自研 MCP Server `labops`

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

## 架构

```
Windows 本地（控制面）              远程 AutoDL GPU（执行面）
┌─────────────────────┐           ┌──────────────────────────┐
│ FastAPI + SSE       │           │ 训练/测试任务 (screen)     │
│  ├─ LangGraph Agent │  ──SSH──▶ │ vLLM 开源模型 (supervisord)│
│  ├─ RAG (Qdrant)    │           │ bge-m3 embedding/reranker │
│  └─ labops MCP      │           │ MinerU 论文解析            │
└─────────────────────┘           └──────────────────────────┘
```

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。
