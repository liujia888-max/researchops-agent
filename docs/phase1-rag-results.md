# Phase 1 · RAG 子系统结果

> 时间：2026-08-29。数据在远程 GPU 主机（autodl-new5）上运行得到，论文为 Restormer（arXiv 2111.09881）。

## 检索链路

PDF → PyMuPDF 解析 → 结构感知分块（按 section 标题切分 + 页内按行打包，带重叠）→ bge-m3 双向量（dense 1024 维 + sparse 词法）→ Qdrant 混合检索（dense + sparse，RRF 融合）→ bge-reranker-v2-m3 重排 → 带引用生成。

- 分块：Restormer 论文 → **216 chunks**，section 标签正确（Abstract / 1. Introduction / 3. Method / 4. Experiments / 5. Conclusion / 4.5 Ablation Studies…），表格数字行不再被误判为标题。

## 检索评测（golden set: 14 条手标问答）

| 指标 | 数值 |
|---|---|
| Recall@5 | **0.893** |
| Hit@5 | **1.000** |
| MRR@5 | **0.657** |
| rerank 提升（ΔRecall@5） | **+0.071** |

含义：

- **Hit@5 = 100%**：14 条问答里，top-5 结果每条都命中了正确答案所在页。
- **Recall@5 = 89.3%**：平均把 89% 的相关页召回到 top-5。
- **rerank +7.1pp**：混合检索（RRF）已覆盖大部分价值，cross-encoder 重排在 top-5 边界上带来约 7 个百分点的召回提升 —— 面试可讲的「为什么做混合检索 + 重排」量化依据。

## 复现

```bash
# 远程主机（Qdrant + bge-m3 已就绪）
cd /root/autodl-tmp
python e2e_rag.py            # 灌入论文 + 混合检索 demo
python eval_retrieval.py     # 跑 golden set，输出 Recall@k/MRR/rerank delta
```

本地单测（无 GPU / 无网络）：

```bash
python -m pytest tests/test_chunking.py tests/test_citation.py tests/test_metrics.py
```

## 已交付组件

- `src/researchops/rag/parser.py` —— PyMuPDF 按页解析
- `src/researchops/rag/chunking.py` —— 结构感知分块（行级扫描，section 标注）
- `src/researchops/rag/embedder.py` —— bge-m3 + reranker HTTP 客户端
- `src/researchops/rag/qdrant_store.py` —— Qdrant 双向量入库（uuid5 幂等）
- `src/researchops/rag/retriever.py` —— `fuse()`（RRF 混合）+ `rerank()`（重排）
- `src/researchops/rag/citation.py` —— 带 `[n]` 引用生成 + grounding 校验
- `src/researchops/eval/metrics.py` —— Recall@k / Hit@k / MRR / rerank delta（纯函数）
- `golden_set/restormer.json` —— 14 条手标问答
- `scripts/eval_ragas.py` —— RAGAS faithfulness/answer_relevancy 评测（检索→生成→LLM judge 打分）

## 带引用生成 demo（真实 LLM，DeepSeek）

`python e2e_citation.py "query"` 走完整链路：混合检索 → 带 `[n]` 引用生成 → grounding 校验 → 页码追溯。

- **概念类问题**（例：`How does the MDTA module achieve linear complexity?`）→ 正确答案，4 条引用全部 grounded，核心定位到 `3.1. Multi-Dconv Head Transposed Attention`（p4）「apply SA across channels rather than spatial」。
- **数字表格类问题**（例：`CBSD68 σ=25 的 PSNR`）→ 模型把 PSNR 说成 `33.04 dB`，正确值是 `34.67 dB`。根因：PyMuPDF 把表格抽成无结构数字流，列头/行标签丢失，LLM 无法映射「数字 → 数据集/σ」；词法级 grounding 只验「引用块里有没有数字」，验不了「数字对不对」。

这正好量化了下一步的必要性：**MinerU 结构化表格解析**（保列头→行标签映射）+ **RAGAS faithfulness**（NLI 判真）。

## RAGAS 生成质量评测（LLM judge）

对同 14 条 golden set，走完整链路（混合检索 → 重排 → top-5 → 生成无引用答案），再用 RAGAS 的 LLM judge 打分。`scripts/eval_ragas.py` 一次性跑完检索 + 生成 + 评测。

| 指标 | mean | min |
|---|---|---|
| **faithfulness**（答案每句话是否有检索上下文支撑，NLI 判真） | **0.9464** | 0.7500 |
| **answer_relevancy**（答案是否真的回应了问题） | **0.7701** | 0.0000 |

faithfulness=0.95 说明「不编造」prompt 生效了：r005（CBSD68 PSNR）现在诚实回答「not explicitly stated」，faithfulness=1.00 —— 上一节 `33.04 dB` 的数字幻觉被压住了。

answer_relevancy 被两条拖低，根因都是**检索/范围错配**，不是生成问题：

- **r005**（CBSD68 PSNR）：faithfulness=1.00 / relevancy=0.00 —— 检索没把 PSNR 表格那一页召回 top-5，模型只能如实说「未提及」。忠实但答非所问，是**召回缺口**。
- **r009**（full image vs patches）：faithfulness=0.75 / relevancy=0.00 —— 答的是「训练用 patches」，而问题问的是「推理时处理整图还是 patch」，**范围错配**。
- r006 / r014 faithfulness=0.75：答案下「state-of-the-art」这类断言时，检索块里没有对应的具体数字做支撑。

结论与上节数字幻觉的根因一致：**表格结构化解析（MinerU）+ 更精准的表格行召回**是当前最值得投入的方向，faithfulness 本身已经足够好。

### ragas 0.4.3 的四个坑（已解决，记录备查）

1. **`ModuleNotFoundError: langchain_community.chat_models.vertexai`**：ragas 0.4.3 无条件 `from langchain_community.chat_models.vertexai import ChatVertexAI`，而 langchain-community 0.4.x 已删该模块。解法：`pip install langchain-google-vertexai` 后在该路径建 shim 文件 re-export。
2. **DeepSeek 只支持 n=1**：ragas 默认 `n=3` 做 statement/question 生成，DeepSeek 直接 400 报错。解法：`LangchainLLMWrapper(..., bypass_n=True)`。
3. **faithfulness 全 TimeoutError**：faithfulness 每条要串行多次 NLI 调用，默认 `RunConfig(timeout=180)` 不够。解法：`RunConfig(timeout=600, max_workers=8)`（并发也别开太高，避免 429）。
4. **两套并行 API**：0.4 有 legacy `evaluate()` + `ragas.metrics`（`Metric` 子类）和新的 `@experiment` + `ragas.metrics.collections`（`SimpleBaseMetric`）。`evaluate()` 只认 legacy 那套，混用会 `TypeError: All metrics must be initialised metric objects`。本脚本走 legacy 路径。

## 复现（RAGAS 追加）

在「检索评测」复现命令基础上追加：

```bash
cd /root/autodl-tmp
python eval_ragas.py /root/autodl-tmp/golden_set/restormer.json 5   # 检索+生成+RAGAS 打分
```
