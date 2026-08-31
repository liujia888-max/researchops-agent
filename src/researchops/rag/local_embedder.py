"""Offline embedding fallback: feature-hashed vectors when no inference service is up.

The RAG pipeline normally embeds with bge-m3 over the remote inference service. To keep
``docker compose up`` + ``ingest`` + ``search`` working on a machine with no GPU (and no
embedding service at all), this module provides a zero-dependency fallback:

* ``hash_embed`` — a deterministic, unit-normalised bag-of-character-ngrams hashed into
  ``DIM`` bins. ``DIM`` matches bge-m3's dense vector size, so Qdrant needs no schema
  change when you later switch to the real embedder.
* ``sparse_weights`` — lexical weights (hashed token id -> occurrence count) as a
  BM25-style sparse vector.

Retrieval quality here is lexical, not semantic: near-synonyms and paraphrases will not
match. It exists so the pipeline *runs* end-to-end offline; point ``INFERENCE_BASE_URL``
at a bge-m3 service for full semantic quality. Vectors are deterministic across runs and
machines (Python's ``hashlib``, not ``hash()``, is used).
"""

from __future__ import annotations

import hashlib
import math
import re

# Must match ``qdrant_store.DENSE_DIM`` so existing collections keep working.
DIM = 1024
_WORD = re.compile(r"[a-z0-9]+")


def _h(token: str, salt: int) -> int:
    """Deterministic 64-bit hash of ``token`` under ``salt`` (stable across runs)."""
    return int.from_bytes(hashlib.sha256(f"{salt}:{token}".encode()).digest()[:8], "little")


def _char_ngrams(text: str, n: int) -> list[str]:
    """Whitespace-stripped lowercased character n-grams of ``text``."""
    s = re.sub(r"\s+", "", text.lower())
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def hash_embed(text: str, *, dim: int = DIM) -> list[float]:
    """Deterministic, unit-normalised bag-of-ngrams hash embedding of ``text``."""
    if not text.strip():
        return [0.0] * dim
    vec = [0.0] * dim
    for n in (2, 3, 4):
        for ng in _char_ngrams(text, n):
            idx = _h(ng, n) % dim
            sign = 1.0 if _h(ng, 100 + n) & 1 else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def sparse_weights(text: str) -> tuple[list[int], list[float]]:
    """Lexical sparse vector: hashed token id -> occurrence count."""
    counts: dict[int, int] = {}
    for word in _WORD.findall(text.lower()):
        idx = _h(word, 7) % 100_000
        counts[idx] = counts.get(idx, 0) + 1
    indices = sorted(counts)
    return indices, [float(counts[i]) for i in indices]


def embed(texts: list[str]) -> list[tuple[list[float], list[int], list[float]]]:
    """Embed a batch in the same ``(dense, sparse_indices, sparse_values)`` shape as
    ``Embedder.embed``, so the RAG pipeline is agnostic to which embedder ran."""
    return [(hash_embed(t), *sparse_weights(t)) for t in texts]


def rerank(query: str, passages: list[str]) -> list[float]:
    """Lexical-fallback relevance: cosine of hash embeddings (both unit-normalised)."""
    q = hash_embed(query)
    return [
        max(0.0, sum(x * y for x, y in zip(q, hash_embed(p), strict=True)))
        for p in passages
    ]
