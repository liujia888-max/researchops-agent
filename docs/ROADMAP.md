# Roadmap

分四期，每期结束都可运行、可演示。完整决策背景见项目根 README。

## Phase 0 — 环境与骨架 ✅

- [x] C 盘清理（回收 ~53GB）
- [x] git（MinGit 便携版，`D:\Users\刘嘉\tools\git`）
- [x] 项目骨架 + ruff/mypy/pytest + CI + Docker Compose（CI 验证）
- [x] LLM Provider 抽象层（deepseek / qwen / vllm 三后端）

## Phase 1 — RAG 子系统

- [ ] 论文 PDF 解析：远程跑 MinerU（公式/表格结构化），本地 PyMuPDF 兜底
- [ ] 结构感知分块（公式/表格原子化，parent-child）
- [ ] bge-m3 双向量（dense+sparse）入库 Qdrant
- [ ] 混合检索 + RRF 融合 + bge-reranker-v2-m3 重排
- [ ] 带引用生成 + 程序化 grounding 校验
- [ ] golden set + RAGAS 评测

## Phase 2 — Agent 运行时 + MCP

- [ ] LangGraph 状态机：Plan → Retrieve → Execute → Monitor → Report
- [ ] 工具层：rag_search / arxiv_search / read_file / python_sandbox
- [ ] 自研 MCP Server `labops`（SSH 到 autodl，screen/supervisord 管长任务）
- [ ] human-in-the-loop 审批 + 步数上限 + 失败降级
- [ ] 长短期记忆

## Phase 3 — 可观测 + 评测 + 交付

- [x] 可观测：轻量 trace（token/cost/latency，`researchops agent --trace`）+ Langfuse 云导出（`--langfuse`，token/成本/延迟面板，trace 上云）
- [x] Agent 轨迹评测（工具召回/精确率、完成率、答案正确率、平均步数/token/成本/延迟，`scripts/eval_agent.py` + `golden_set/`）
- [x] Next.js + React 前端（`web/`，SSE 流式查看 Agent 每步 + Langfuse 追踪开关 + Trace token/成本 + 历史实验列表）
- [ ] vLLM 私有化演示 + 成本对比
- [ ] README 量化指标 + 录屏 demo

## 环境约束备忘

- 本机无 Docker，Dockerfile/compose 由 CI 验证（`docker compose build api`）
- conda 需配国内镜像（清华 anaconda 已停用，见 CLAUDE.md）
- 远程 GPU 主机 `autodl-new5` 可能随时关机，Agent 需容忍其不可达
