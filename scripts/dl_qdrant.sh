#!/usr/bin/env bash
# Download Qdrant binary via gh-proxy (github is slow from autodl), then extract.
set -euo pipefail
cd /root/autodl-tmp
curl -sL -o qdrant.tar.gz \
  "https://gh-proxy.com/https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-unknown-linux-gnu.tar.gz"
tar xzf qdrant.tar.gz
echo OK > qdrant.done
echo "QDRANT_DONE"
