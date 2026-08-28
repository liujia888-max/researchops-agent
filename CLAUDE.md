# CLAUDE.md — ResearchOps Agent 项目

面向深度学习实验的 LLM Agent（RAG + Agent + MCP + 可观测）。求职定位：AI 应用/Agent 工程师（应届校招）。

## 项目根目录
`D:\Users\刘嘉\ResearchOps-Agent`（D 盘根不可写，所有工具/数据都放 D 盘用户目录下）。

## 本地开发环境（已实测，勿再踩坑）

- **Python**：用 conda 环境 `fastapi-env`（Python 3.13.14，`D:\Users\miniconda\envs\fastapi-env\python.exe`）。
  运行时依赖（fastapi/uvicorn/pydantic/pydantic-settings/httpx）和 dev 工具（pytest/ruff/mypy）都已装好。
  py3.12 环境未建成，Phase 1 上 ML 库时再解决（本地无 GPU，ML 依赖主要在远程跑）。
- **conda**：默认国外源超时；清华 anaconda 通道已停用（403）。已写入 `C:\Users\刘嘉\.condarc` 但清华 anaconda 不可用，
  建环境改用 pip（默认 PyPI 可用，无需镜像）。不要用 `conda create`。
- **git**：MinGit 便携版，`D:\Users\刘嘉\tools\git\cmd\git.exe`（不在 PATH，用绝对路径或手动加 PATH）。
  仓库默认分支 `main`。本地身份是占位符 `ResearchOps Agent <researchops@example.com>`，待用户改真实身份。
- **Node/npm**：原 Node 装在已删除的 WorkBuddy 目录里，**已失效**。前端 Phase 3 前需重装 Node。
- **Docker**：本机无 Docker。`Dockerfile`/`docker-compose.yml` 由 GitHub Actions CI 的 `docker compose build api` 验证。
- **磁盘**：C 盘已清到 ~53GB 余量；D 盘剩 ~65GB。所有缓存/数据落 D 盘，别往 C 盘写。

## 远程 GPU 主机

- 别名 `autodl-new5`（`connect.westd.seetacloud.com:49830`，root，`~/.ssh/id_rsa` 免密）。
  RTX 5090 32GB，`/root/autodl-tmp` 剩 ~49GB。**按量计费，可能随时关机，Agent 必须容忍其不可达**。
- 远程有去噪项目 `/root/autodl-tmp/pythonProject4`（Restormer/DnCNN/wavelet-transformer）。
- 网络：github.com / hf-mirror.com / modelscope.cn / api.deepseek.com / pypi 清华源 均可达。

## 关键命令（先 cd 到项目根，用 fastapi-env 的 python）

```powershell
$py = 'D:\Users\miniconda\envs\fastapi-env\python.exe'
& $py -m ruff check .        # lint
& $py -m mypy src            # typecheck
& $py -m pytest              # test（6 个用例）
& $py -m researchops ping --provider deepseek --message "hi"   # 测 LLM（需 .env 配 key）
```

## 代码结构

- `src/researchops/llm/providers.py` — LLM Provider 抽象（deepseek/qwen/vllm 三后端，OpenAI 兼容协议）
- `src/researchops/server/main.py` — FastAPI 应用（`/health`、`/chat/completions`）
- `src/researchops/config.py` — pydantic-settings，读 `.env`（key 不入库）

## 约定

- Python 3.12+，ruff（line-length 100）+ mypy strict，pytest + asyncio auto 模式
- 密钥走环境变量/`.env`，绝不入库（`.gitignore` 已排除）
- 提交前跑 `ruff check` + `mypy src` + `pytest` 三绿
