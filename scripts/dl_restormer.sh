#!/usr/bin/env bash
# Download the Restormer paper (arxiv 2111.09881) with retries and size check.
set -euo pipefail
cd /root/autodl-tmp

for i in 1 2 3 4 5; do
    wget -q -c --tries=3 --timeout=90 -O restormer.pdf https://arxiv.org/pdf/2111.09881 || true
    sz=$(stat -c %s restormer.pdf 2>/dev/null || echo 0)
    echo "attempt $i size=$sz"
    # Full PDF is ~7.57MB; treat >=7MB as complete.
    if [ "$sz" -ge 7000000 ]; then
        echo "DOWNLOAD_OK size=$sz"
        exit 0
    fi
    sleep 3
done
echo "DOWNLOAD_INCOMPLETE final_size=$(stat -c %s restormer.pdf 2>/dev/null || echo 0)"
