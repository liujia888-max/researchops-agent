# Phase 1 · RAG 子系统结果

> 时间：2026-08-29（检索/RAGAS 首版）→ 2026-08-30（表格结构化解析 + 表格行召回）。数据在远程 GPU 主机上运行得到，论文为 Restormer（arXiv 2111.09881）。

## 检索链路

PDF → PyMuPDF 解析 → 结构感知分块（按 section 标题切分 + 页内按行打包，带重叠）→ **表格结构化解析（`find_tables` 单元格网格 → method × dataset × σ 行）** → bge-m3 双向量（dense 1024 维 + sparse 词法）→ Qdrant 混合检索（dense + sparse，RRF 融合）→ bge-reranker-v2-m3 重排 → 带引用生成。

- 分块：Restormer 论文 → **307 chunks** = 216 prose + **91 表格行**。section 标签正确（Abstract / 1. Introduction / 3. Method / 4. Experiments / 5. Conclusion / 4.5 Ablation Studies…）。

## 表格结构化解析（本次新增）

论文里的 SOTA 对比表（method × dataset × σ）在 PyMuPDF 默认 `get_text` 下会被抽成无结构数字流——列头（数据集）和行标签（方法）丢失，于是「Restormer 在 CBSD68 σ=25 的 PSNR」无法映射到 31.79。

`src/researchops/rag/tables.py` 用 `page.find_tables()` 的单元格网格（保留行列对齐与 `-` 占位符）重新解析这类表，把每个「方法 × 数据集 × σ → 值」组合还原成一条自包含、可检索的行：

```
Table 5. Gaussian color image denoising ... Restormer on CBSD68: σ=15 PSNR 34.40, σ=25 PSNR 31.79, σ=50 PSNR 28.60
```

- 解析出 **91 行**（灰度表 13 方法 × 2 数据集 + 彩色表 13 方法 × 5 数据集），`-` 占位符（如 RPCNN σ=15 无值）正确保留。
- 每行作为 `chunk_type="table_row"` 的独立 chunk 入库，与 prose chunk 一起参与检索；`Chunk.chunk_type` 进 payload、retriever 回传，供指标区分「结构化行」vs「数字流」。
- 两种 body 布局都处理：列拆分（方法名列 + 每数据集一列）与折叠（整个方法表落进第一列，逐行「方法 + 值」）。
- 范围：denoising/restoration 常见的「method × dataset × σ」结果表；不匹配此形状的表跳过（内容仍由 prose chunk 覆盖）。MinerU 会在上游替换掉这套启发式。

## 检索评测（golden set: 14 条手标问答）

| 指标 | 数值 |
|---|---|
| Recall@5 | **0.857** |
| Hit@5 | **1.000** |
| MRR@5 | **0.621** |
| rerank 提升（ΔRecall@5） | **+0.036** |
| **表格行召回@5** | **0.500** |

含义：

- **Hit@5 = 100%**：14 条问答里，top-5 每条都命中了正确答案所在页。
- **表格行召回@5 = 50%**（新增指标）：表格类问题里，结构化表格行（非 prose 数字流）在 top-5 出现的比例。r005（明确问 CBSD68 σ=25 的数字）**命中**——「Restormer on CBSD68」行排第 1；r006（问 Restormer 与 SwinIR 的对比）未命中——对比类 query 里 prose 数字流 + SwinIR 的**彩色**表行占满 top-5，灰度表那一行没进前 5（见下「遗留缺口」）。
- Recall@5 相对上一版 0.893 微降至 0.857：来自 4 条「答案横跨两页」的条目（r002/r007/r009/r013）其中一页没进 top-5；r005 反而从 0 升到 1（表格页被召回）。净效果是「精确数字问题」被修好，代价是「多页概念题」边界的轻微晃动。

## 带引用生成 demo（真实 LLM，DeepSeek）

`python e2e_citation.py "query"` 走完整链路：混合检索 → 带 `[n]` 引用生成 → grounding 校验 → 页码追溯。

- **概念类问题**（例：`How does the MDTA module achieve linear complexity?`）→ 正确答案，4 条引用全部 grounded，核心定位到 `3.1. Multi-Dconv Head Transposed Attention`「apply SA across channels rather than spatial」。
- **数字表格类问题**（例：`CBSD68 σ=25 的 PSNR`）→ 表格结构化解析前，PyMuPDF 把表抽成无结构数字流，模型说 `33.04 dB`（错）。表格行入库后，检索 top-1 直接命中 `Restormer on CBSD68: σ=25 PSNR 31.79`。注意正确值是 **31.79 dB**（σ=25，单模型 31.78），不是 34.67——34.67 是 DSNet 在 McMaster σ=15 的值，别张冠李戴。

## RAGAS 生成质量评测（LLM judge）

对同 14 条 golden set，走完整链路（混合检索 → 重排 → top-5 → 生成无引用答案），再用 RAGAS 的 LLM judge 打分。`scripts/eval_ragas.py` 一次性跑完检索 + 生成 + 评测。

| 指标 | mean | min |
|---|---|---|
| **faithfulness**（答案每句话是否有检索上下文支撑，NLI 判真） | **0.9583** | 0.6667 |
| **answer_relevancy**（答案是否真的回应了问题） | **0.8366** | 0.0000 |

表格行召回带来的变化（对比上一版 faithfulness=0.9464 / relevancy=0.7701）：

- **r005（CBSD68 PSNR）从「答非所问」到「答对」**：上一版检索没召回表格页，模型只能诚实说「not explicitly stated」（faithfulness=1.0 但 relevancy=0.0）。现在 top-1 命中表格行，模型答出 **34.40 / 31.79 / 28.60**（σ=15/25/50），faithfulness=1.00 / relevancy=0.99。
- **r009、r014 也改善**：faithfulness 升到 1.0，r009 relevancy 0.0→0.84。
- **遗留缺口 r006（Restormer vs SwinIR 灰度对比）**：relevancy 仍 0.00。根因是对比类 query（提到「SwinIR」「grayscale」）会让 prose 数字流 + SwinIR 的**彩色**表行占满 top-5，灰度表「Restormer on BSD68」那行没进前 5。这是 chunk 粒度的检索质量问题（表格行自包含但缺少「SwinIR」这类对照上下文），下一步方向是表格行加权 / 查询扩展，而非解析 bug。

### ragas 0.4.3 的四个坑（已解决，记录备查）

1. **`ModuleNotFoundError: langchain_community.chat_models.vertexai`**：ragas 0.4.3 无条件 `from langchain_community.chat_models.vertexai import ChatVertexAI`，而 langchain-community 0.4.x 已删该模块。解法：`pip install langchain-google-vertexai` 后在该路径建 shim 文件 re-export。
2. **DeepSeek 只支持 n=1**：ragas 默认 `n=3` 做 statement/question 生成，DeepSeek 直接 400 报错。解法：`LangchainLLMWrapper(..., bypass_n=True)`。
3. **faithfulness 全 TimeoutError**：faithfulness 每条要串行多次 NLI 调用，默认 `RunConfig(timeout=180)` 不够。解法：`RunConfig(timeout=600, max_workers=8)`（并发也别开太高，避免 429）。
4. **两套并行 API**：0.4 有 legacy `evaluate()` + `ragas.metrics`（`Metric` 子类）和新的 `@experiment` + `ragas.metrics.collections`（`SimpleBaseMetric`）。`evaluate()` 只认 legacy 那套，混用会 `TypeError: All metrics must be initialised metric objects`。本脚本走 legacy 路径。

## 复现

```bash
# 远程主机（Qdrant + bge-m3 已就绪）
cd /root/autodl-tmp
python e2e_rag.py            # 灌入论文（prose + 表格行）+ 混合检索 demo
python eval_retrieval.py     # 跑 golden set，输出 Recall@k/MRR/rerank delta/表格行召回
python eval_ragas.py /root/autodl-tmp/golden_set/restormer.json 5   # 检索+生成+RAGAS 打分
```

本地单测（无 GPU / 无网络）：

```bash
python -m pytest tests/test_chunking.py tests/test_citation.py tests/test_metrics.py
```

## 已交付组件

- `src/researchops/rag/parser.py` —— PyMuPDF 按页解析
- `src/researchops/rag/chunking.py` —— 结构感知分块（行级扫描，section 标注）
- `src/researchops/rag/tables.py` —— **表格结构化解析**（`find_tables` 网格 → `TableRow`，`chunk_type="table_row"`）
- `src/researchops/rag/embedder.py` —— bge-m3 + reranker HTTP 客户端
- `src/researchops/rag/qdrant_store.py` —— Qdrant 双向量入库（uuid5 幂等，payload 带 `chunk_type`）
- `src/researchops/rag/retriever.py` —— `fuse()`（RRF 混合）+ `rerank()`（重排）
- `src/researchops/rag/citation.py` —— 带 `[n]` 引用生成 + grounding 校验
- `src/researchops/eval/metrics.py` —— Recall@k / Hit@k / MRR / rerank delta / **表格行召回**（纯函数）
- `golden_set/restormer.json` —— 14 条手标问答（含 `gold_table` 标注；修正 r005 错误答案 34.67→31.79）
- `scripts/eval_ragas.py` —— RAGAS faithfulness/answer_relevancy 评测（检索→生成→LLM judge 打分）
