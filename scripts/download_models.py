import os

os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# hf-mirror cannot proxy xet CAS requests (401); force plain HTTP download.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

# hf-mirror 403s on junk metadata files (.DS_Store etc.); skip them and only
# pull the files actually needed to load the model.
ignore = ["*.DS_Store", "*.h5", "*.ot", "*.msgpack", "*.tiktoken", "*.pdf", "*.png", "*.jpg"]

for repo in ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"]:
    print(f"downloading {repo} ...", flush=True)
    p = snapshot_download(repo_id=repo, ignore_patterns=ignore)
    print(f"  done -> {p}", flush=True)

print("ALL_MODELS_DOWNLOADED", flush=True)
