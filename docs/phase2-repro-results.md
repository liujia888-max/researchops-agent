# Phase 2 — 复现 Restormer 与 model_v3_rgb 对比（真提交 + 轮询 + 落库）

日期：2026-08-30

## 目标

用 ResearchOps Agent 的确定性链路，在远程 GPU 主机（AutoDL RTX 5090）上：

1. 复现 **Restormer**（官方 Gaussian Color Blind 预训练模型）在 **CBSD68**（彩色，σ=15/25/50）上的 PSNR/SSIM；
2. 复现用户自研模型 **model_v3_rgb** 在同一数据集上的结果；
3. 全程走「真提交 → 轮询 → 落库」链路：`submit_job`（screen 后台任务）→ `job_status` 轮询至结束 → `tail_log` + `parse_metrics` 解析 → `ExperimentStore` 持久化。

全程无 LLM 参与热路径——命令预先确定，Pipeline 只负责可靠执行与落库。

## 执行命令

> `submit_job` 会把命令包装为 `cd /root/autodl-tmp && <command> >> <log> 2>&1` 并以 detached screen 会话运行。

- **model_v3_rgb**：`cd pythonProject4 && python3 test_rgb.py`
- **Restormer**：
  ```bash
  cd pythonProject4/Restormer/Denoising && \
    PYTHONPATH=/root/autodl-tmp/pythonProject4/Restormer \
    python3 test_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50 && \
    PYTHONPATH=/root/autodl-tmp/pythonProject4/Restormer \
    python3 evaluate_restormer_cbsd68_psnr_ssim.py --model_type blind --sigmas 15,25,50
  ```

## 结果（PSNR / SSIM，CBSD68）

| σ | model_v3_rgb PSNR | model_v3_rgb SSIM | Restormer PSNR | Restormer SSIM |
|---|------------------:|------------------:|---------------:|---------------:|
| 15 | 32.48 | 0.8966 | 34.39 | 0.9384 |
| 25 | 29.96 | 0.8458 | 31.78 | 0.8982 |
| 50 | 26.23 | 0.6961 | 28.59 | 0.8172 |

## 对比（Restormer − model_v3_rgb）

| σ | Δ PSNR (dB) |
|---|------------:|
| 15 | +1.91 |
| 25 | +1.81 |
| 50 | +2.36 |

Restormer 作为 SOTA Transformer 去噪基线，在三个噪声水平上均显著高于 model_v3_rgb，
且差距随噪声水平升高而扩大（σ=50 时最大 +2.36 dB）。这与历史实验记录一致
（Restormer σ=25 参考值 31.78 dB，model_v3_rgb σ=25 参考值 29.96 dB）。

## 落库记录

`ExperimentStore`（SQLAlchemy 2.0 + SQLite）持久化了完整链路结果：

- 2 条 `Experiment`：`repro_cbsd68_model_v3_rgb`、`repro_restormer_cbsd68`
- 2 条 `JobRun`（均 `completed`，含 `started_at` / `finished_at`）
- 12 条 `Metric`（每条 run 6 条：PSNR/SSIM × σ=15/25/50）

## 复现方式

```bash
# 前提：远程主机在线、~/.ssh/id_rsa 可免密登录、.env 已配置（可选覆盖 labops_*）
python scripts/run_repro.py     # 一键走完 preflight → submit → poll → 落库 → 对比
python scripts/preflight.py     # 仅做只读体检（GPU 余量 / 训练进程 / 脚本与权重存在性）
```

## 说明

- 运行期间远程训练进程 `train_v6_noiseadaptive_subband.py`（PID 3606）未被触碰，
  两条 eval 仅复用空闲显存（约 24.9 GB 可用）。
- 修复了一个真实环境问题：AutoDL 较新 `screen -ls` 输出带 `(MM/DD/YY HH:MM:SS)`
  时间戳字段，旧解析正则无法识别为 live 会话，导致 `job_status` 误判「已结束」而提前拉日志。
  已改为兼容新旧两种格式（见 `src/researchops/labops/client.py` 的 `parse_screen_sessions`）。
