# 使用指南（自己 / 别人）

同一份代码，差别只在**外部依赖**谁配好了：LLM key、Qdrant、embedding/reranker、GPU 主机、Langfuse 账号。按「你自己（依赖已就绪）」和「别人（全新 clone）」两种身份分开讲。

> **先读结论**：没有 GPU、没有 embedding 服务，也能把 RAG + Agent 核心链路跑通。默认
> `RAG_FALLBACK_LOCAL=true`，推理服务不可达时自动退化为零依赖的 feature-hash 嵌入（词法
> 检索），链路不报错。GPU 只是「可选资源」，用于语义检索质量 + 私有化/成本对比。

## 最快路径：Docker 一键起全套（含前端）

```bash
git clone git@github.com:liujia888-max/researchops-agent.git
cd researchops-agent
cp .env.example .env         # 填一个 LLM API Key（DeepSeek / Qwen 二选一）
docker compose up --build
```

起好后：

| 服务 | 地址 |
|---|---|
| 网页版（上传文档 + 聊天 + 报告） | http://localhost:3000 |
| API 文档（FastAPI Swagger） | http://localhost:8000/docs |
| Langfuse 可观测面板 | http://localhost:3001 |

灌入示例文档，立刻验证 RAG：

```bash
docker compose exec api researchops ingest examples/sample_document.md
docker compose exec api researchops search "denoising method"
```

## 本地开发路径（conda + uvicorn，从零）

```bash
# Python 3.12 环境
conda create -y -p .venv python=3.12 pip
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate

# 安装（含 dev 依赖：ruff/mypy/pytest）
pip install -e ".[dev]"

# 填 key（.env 不入库，模板见 .env.example）
cp .env.example .env
```

装完后自检：`researchops doctor` 逐项报告 LLM / Qdrant / embedding / GPU 主机是否就绪，再单独测 LLM：

```bash
researchops doctor               # 依赖体检
researchops ping                 # 期望输出 [deepseek-chat] pong + token 数
```

## 计算在哪里跑（控制面 vs 执行面）

- **控制面（你本机 / 部署机）**：FastAPI + LangGraph Agent + RAG(Qdrant) + labops MCP，负责思考、检索、编排、出报告。
- **执行面（任意 GPU 主机）**：真正的训练/测试任务，经 `labops` MCP 走 SSH 提交到远程 GPU 上跑；embedding/reranker 的 bge-m3 服务也可跑在这里。
- 两者通过 `LABOPS_HOST/PORT/...` 解耦：GPU 主机关了也不崩，Agent 降级为纯检索问答。

## 按能力分层：配多少外部依赖，就能跑到哪一层

| 层 | 需要的外部依赖 | 能跑什么 |
|---|---|---|
| ① LLM | 一个 DeepSeek / Qwen key（`.env`） | `researchops ping` ✅ |
| ② RAG | Qdrant（`QDRANT_URL`，默认 `http://127.0.0.1:6333`） | `researchops ingest <file>` / `researchops search "<query>"` / 带引用问答 ✅ |
| ③ RAG（语义质量） | 再加 bge-m3/reranker 推理服务（`INFERENCE_BASE_URL`） | 混合检索 dense+sparse + rerank（缺省时自动退化词法检索）✅ |
| ④ 完整 Agent | 自己的 GPU 主机 + SSH 免密（`LABOPS_HOST/PORT/USER/KEY_PATH/WORKDIR`） | `researchops agent "<task>"` 端到端：检索→读代码→提交任务→轮询→出报告 ✅ |
| ⑤ 可观测 | 自己的 Langfuse 账号（`LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL`） | `--langfuse` 把 trace 导出到云面板 ✅ |

> **没有 GPU 也能演示核心链路**：Agent 对 GPU 不可达做了结构化降级——labops 工具返回可读错误并建议「开机 / 换只读任务」，Agent 回退为纯检索问答，不会崩。

## 命令一览

```bash
researchops doctor                                   # 依赖体检（LLM/Qdrant/embedding/GPU 逐项）
researchops ping                                     # 测 LLM provider
researchops ingest <paper.pdf>                       # 文档入库 RAG（PDF/Word/txt/md）
researchops search "<query>" --top-k 5               # 混合检索
researchops mcp                                      # labops MCP server（stdio）
researchops agent "<task>"                           # 端到端 agent
researchops agent "<task>" --max-iterations 10       # 步数上限
researchops agent "<task>" --interactive-approval    # 危险操作（submit/cancel）人工审批
researchops agent "<task>" --multi                   # 多智能体（supervisor + 专家）
researchops agent "<task>" --trace                   # 本地 token/成本/延迟摘要
researchops agent "<task>" --langfuse                # 导出 trace 到 Langfuse 并打印 URL
```

## 网页版（Web App）

`docker compose up` 已内置前端。手动起（本地开发）用两个终端：

```bash
# 终端 1：后端（SSE 流式）
uvicorn researchops.server.main:app --reload        # http://localhost:8000

# 终端 2：前端（Next.js + React）
cd web
npm install
npm run dev                                         # http://localhost:3000
```

前端默认连 `http://localhost:8000`，换后端地址用环境变量覆盖：

```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
```

> Windows 本机注意：`node`/`npm` 不在 PATH，用 `$env:Path = "D:\;" + $env:Path` 或直接 `& "D:\node.exe" "D:\node_modules\npm\bin\npm-cli.js" run dev`。

网页端支持**直接上传文档入库**：点击「文档库」面板的上传按钮，选择 PDF / Word(.docx) / txt / md 文件，上传后即解析→分块→向量化→入库 Qdrant，成为 Agent `rag_search` 的检索语料。也提供 `POST /documents`（multipart 上传）和 `GET /documents`（列出已入库文档）两个 REST 接口。

## 别人用 Docker 起全套（可选）

```bash
docker compose up -d
```

`docker-compose.yml` 起 api + 前端(Next.js) + Qdrant(`:6333`) + Langfuse(`:3001`) + Postgres + Redis。**embedding/reranker 推理服务和 GPU 任务提交仍需一台 GPU**——但缺省时 RAG 自动退化词法检索、Agent 对 GPU 不可达降级，所以无 GPU 环境也能完整演示检索问答链路。GPU 在本项目里是「可选资源」：云 API 即可跑通核心链路，GPU 只用于语义检索 + 私有化/成本对比。

## 开发自检

```bash
make check      # = ruff check . + mypy src + pytest
```

## 你自己（依赖已就绪）的日常流程

你的 `.env` 已填 DeepSeek + Langfuse key、SSH 已配免密，因此直接：

```bash
# 一条命令完整跑
researchops agent "复现 Restormer 在 CBSD68 σ=25 的结果并与 model_v3_rgb 对比" --trace --langfuse

# 或网页版（两个终端，见上）
```
