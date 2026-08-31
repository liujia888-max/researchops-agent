"""Tests for the zero-dependency feature-hash fallback embedder."""

from __future__ import annotations

from researchops.rag.local_embedder import DIM, embed, hash_embed, rerank, sparse_weights


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_dense_vector_is_deterministic_and_unit_normed() -> None:
    a = hash_embed("wavelet transformer denoising")
    b = hash_embed("wavelet transformer denoising")
    assert a == b
    assert len(a) == DIM
    assert abs(sum(v * v for v in a) - 1.0) < 1e-6


def test_empty_text_returns_zero_vector() -> None:
    assert hash_embed("   ") == [0.0] * DIM


def test_similar_texts_are_closer_than_unrelated() -> None:
    q = hash_embed("image denoising with transformer")
    close = hash_embed("image denoising with transformer")
    far = hash_embed("quantum chemistry molecular dynamics")
    assert _cos(q, close) > _cos(q, far)


def test_sparse_weights_are_lexical_counts() -> None:
    indices, values = sparse_weights("denoising denoising image")
    assert len(indices) == 2  # two unique words
    assert sum(values) == 3.0  # three tokens total


def test_embed_shape_matches_pipeline_contract() -> None:
    out = embed(["hello world", "another text"])
    assert len(out) == 2
    for dense, indices, values in out:
        assert len(dense) == DIM
        assert len(indices) == len(values)


def test_rerank_prefers_matching_passage() -> None:
    scores = rerank("wavelet transformer", ["wavelet transformer detail", "gpu cluster"])
    assert scores[0] > scores[1]
    assert all(s >= 0.0 for s in scores)
