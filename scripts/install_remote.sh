#!/usr/bin/env bash
# Install remote inference deps on autodl-new5 (RTX 5090).
# Run: bash /root/autodl-tmp/install_remote.sh
set -euo pipefail

# Prefer Tsinghua PyPI mirror (faster in CN); falls back transparently.
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

python -m pip install --upgrade pip -i "$PIP_INDEX"
python -m pip install -r /root/autodl-tmp/requirements-remote.txt -i "$PIP_INDEX"

echo "=== done. torch: $(python -c 'import torch; print(torch.__version__)') ==="
