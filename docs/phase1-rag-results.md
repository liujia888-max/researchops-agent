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

## 带引用生成 demo（真实 LLM，DeepSeek）

`python e2e_citation.py "query"` 走完整链路：混合检索 → 带 `[n]` 引用生成 → grounding 校验 → 页码追溯。

- **概念类问题**（例：`How does the MDTA module achieve linear complexity?`）→ 正确答案，4 条引用全部 grounded，核心定位到 `3.1. Multi-Dconv Head Transposed Attention`（p4）「apply SA across channels rather than spatial」。
- **数字表格类问题**（例：`CBSD68 σ=25 的 PSNR`）→ 模型把 PSNR 说成 `33.04 dB`，正确值是 `34.67 dB`。根因：PyMuPDF 把表格抽成无结构数字流，列头/行标签丢失，LLM 无法映射「数字 → 数据集/σ」；词法级 grounding 只验「引用块里有没有数字」，验不了「数字对不对」。

这正好量化了下一步的必要性：**MinerU 结构化表格解析**（保列头→行标签映射）+ **RAGAS faithfulness**（NLI 判真）。

## 待办（依赖外部资源）

- RAGAS faithfulness / answer_relevancy：需 LLM judge + `ragas` 依赖（解决上述数字幻觉问题）。
