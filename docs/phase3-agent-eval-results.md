# Phase 3 · Agent 轨迹评测结果

> 时间：2026-08-31。真实调用 DeepSeek，18 条 golden task 一次跑完，`answer_accuracy` 从子串匹配升级为 LLM-judge（faithfulness 判断）。

## 评测设置

- **任务集**：`golden_set/agent_tasks.json`，从 3 条扩充到 **18 条**，覆盖三类：
  - **检索类（8 条）**：纯 `rag_search`，问论文的方法名/子网络/核心挑战/评估指标等，每条带 `expected_facts`（用备选列表容错，如 `["ground truth","ground-truth"]`）。
  - **只读 labops 类（4 条）**：`gpu_info` / `list_experiments` / `job_status` / `tail_log`，`expected_facts` 为空（只判「是否调对工具并跑完」）。
  - **多步类（6 条）**：检索 + 只读工具组合，如 `rag_search+gpu_info`、`gpu_info+list_experiments`、`rag_search+job_status+tail_log`。
- **judge 升级**：`answer_accuracy` 不再用「报告里出现 `31.79` 就算对」的子串匹配，改成对 agent 最终报告做 LLM-judge——判断「是否回应了任务 + 事实是否与 `expected_facts` 一致 + 是否凭空捏造数字/论断」，只回 `PASS`/`FAIL`。子串匹配保留为 `--no-judge` 的兜底基线（`task_passed`）。
- **运行环境**：本地 `eval_agent.py --judge`，DeepSeek（`llm/providers.py` 默认后端）+ Qdrant（127.0.0.1:6333）+ bge-m3（:8001）真实就绪；GPU 主机本次**在线**（`gpu_info` 返回 RTX 5090、`list_experiments` 返回目录列表），只读工具拿到真实数据而非退化错误。
- **只读护栏**：eval 不注册 `run_experiment`、无 approver，`submit_job`/`cancel_job` deny-by-default——`check-job-status` 里 agent 尝试 `submit_job` 被 REJECTED 后自动退回 `job_status`/`tail_log`，正好验证了「危险操作 deny-by-default + 降级」这条设计。

## 总分

| 指标 | 数值 | 说明 |
|---|---|---|
| 任务数 | 18 | 检索 8 / 只读 labops 4 / 多步 6 |
| **completion_rate**（跑完率） | **1.000** | 18/18 全部到达终态，无卡死/超步数 |
| **answer_accuracy**（LLM-judge） | **0.889** | 16/18 判对，2 条判错（见下） |
| **tool_recall**（工具召回） | **1.000** | 每条 `expected_tools` 都被调过，无漏调 |
| **tool_precision**（工具精确） | **0.778** | 平均每任务多调了约 0.22 个「golden 之外」的工具 |
| avg_steps（平均步数） | 2.06 | 检索类多为 1 步，多步类 2–4 步 |
| avg_tokens（平均 token） | 11 605 | 单任务 in+out 合计 |
| avg_cost_usd（平均成本） | $0.00424 | 18 任务合计约 **$0.076** |
| avg_wall_s（平均耗时） | 14.10 s | 18 任务合计约 4.2 分钟 |

## 分任务明细

| id | 类型 | steps | 调用工具 | passed |
|---|---|---|---|---|
| n2n-title-method | 检索 | 1 | rag_search | ✅ |
| n2n-two-subnets | 检索 | 1 | rag_search | ✅ |
| n2n-problem | 检索 | 1 | rag_search | ✅ |
| n2n-no-ground-truth | 检索 | 2 | rag_search ×2 | ✅ |
| n2n-metrics | 检索 | 1 | rag_search | ✅ |
| n2n-applications | 检索 | 2 | rag_search ×2 | ❌ |
| n2n-authors | 检索 | 1 | rag_search | ✅ |
| n2n-augmentation-role | 检索 | 1 | rag_search | ❌ |
| gpu-inventory | 只读 labops | 1 | gpu_info | ✅ |
| list-workdir | 只读 labops | 1 | list_experiments | ✅ |
| check-job-status | 只读 labops | 3 | submit_job → job_status → tail_log | ✅ |
| tail-job-log | 只读 labops | 3 | list_experiments → job_status → tail_log | ✅ |
| multi-rag-gpu | 多步 | 3 | rag_search + gpu_info + list_experiments | ✅ |
| multi-rag-workdir | 多步 | 3 | rag_search ×2 + list_experiments | ✅ |
| multi-gpu-workdir | 多步 | 2 | gpu_info + list_experiments | ✅ |
| multi-rag-jobstatus | 多步 | 4 | rag_search + job_status + tail_log + fetch_metrics | ✅ |
| multi-full-sweep | 多步 | 3 | rag_search + gpu_info + list_experiments | ✅ |
| multi-rag-tail | 多步 | 4 | rag_search + list_experiments + job_status + tail_log | ✅ |

> `check-job-status` 的 `submit_job` 是被 deny-by-default 拦掉的（工具确实被调用了，只是返回 REJECTED），随后 agent 自动改用 `job_status` + `tail_log` 完成了任务——这正是「只读优先 + 危险操作需审批」的预期行为。

## 两条失败分析（answer_accuracy 丢的 2 分）

两条都是**检索类**，`expected_facts` 是英文 token，最终报告却是中文，judge 判了 FAIL：

- **n2n-applications**（`expected_facts=["medical images"]`）：报告写的是「该方法可用于**医学图像**去噪……CT……」，并称「可应用于**多个实际领域**」。judge 判 FAIL 的主因是**过度断言**——报告把「医学图像」扩写成了「多个领域 + CT」等具体应用，超出了 `expected_facts`（仅 medical images）能支撑的范围，触犯了「不凭空捏造论断」这条。
- **n2n-augmentation-role**（`expected_facts=["validation data","validation training"]`）：报告只讲到自增强网络「从已有噪声图像集生成新噪声图像以解决盲噪声局限」，**没有落到「用于验证集/验证训练」这个具体用途**上，缺了关键事实 → FAIL。

两点结论：

1. **LLM-judge 确实比子串匹配更严格**：这两条若用子串匹配，`n2n-applications` 会因「medical images」未以英文原样出现而同样判错，但 `n2n-augmentation-role` 这类「事实在报告里缺位」的情况子串匹配未必能干净地区分——judge 能明确判「答得不完整/答偏」而非「恰好没出现那个词」。
2. **golden fact 本身要再核**：这两条恰好是「论文是否真的把应用落到 medical images / 是否真的说生成图用于 validation」边界上，需要回论文原文核对 `expected_facts` 是否准确，再决定是修 fact 还是修报告的忠实度。下一步把这两条连同 `restormer.json` 的 gap 一起过一遍。

## 关于当前语料（重要）

`agent_tasks.json` 的检索事实是按**当前 Qdrant 里实际入库的论文**写的——即 *Self-Augmented Noisy Image for Noise2Noise Image Denoising*（IEEE Access 2024，单文档、96 prose chunk），**不是** Phase 1 文档里的 Restormer。原 `agent_tasks.json` 引用的 Restormer 事实（31.79 等）已随语料替换而失配，本阶段按现语料重写了检索类 `expected_facts`。portfolio 里若要把「Restormer 复现 + 对比」讲成主线，需要先把 Restormer 重新灌回 Qdrant，再回填这批数字类 fact。

## 复现

```bash
# 环境就绪（.env 有 DeepSeek key；Qdrant :6333 + bge-m3 :8001 在线；GPU 主机可选）
python scripts/eval_agent.py --judge        # 18 条，LLM-judge
python scripts/eval_agent.py --no-judge     # 兜底：子串匹配
```

无 LLM / 无 GPU 的纯函数与脚本化 harness 单测：

```bash
python -m pytest tests/test_agent_eval.py tests/test_agent.py tests/test_mcp_adapter.py
```

## 已交付组件

- `golden_set/agent_tasks.json` —— 18 条任务（检索/只读 labops/多步，`expected_tools` + `expected_facts` 备选列表容错）
- `src/researchops/eval/agent_eval.py` —— 轨迹指标纯函数 + `run_eval`；新增 `judge_report`（LLM-judge，faithfulness）、`TaskOutcome.passed`、`_render_facts`；`_fact_matches` 改为大小写不敏感
- `scripts/eval_agent.py` —— 修掉 `build_default_tools` 的 async 调用点，加 `--judge`/`--no-judge`
- `tests/test_agent_eval.py` —— 新增 `judge_report`、大小写不敏感、judge 覆盖子串匹配等用例
- 同期：`graph.py` 重复调用护栏 + 失败重试、`build_default_tools` 全 async 化后的调用点修复（`scripts/agent_demo.py` 等）
