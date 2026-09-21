"""Tests for the hybrid retriever's failure modes (no live Qdrant needed)."""

from __future__ import annotations

import pytest

from researchops.config import Settings
from researchops.rag.retriever import Retriever


async def test_retrieve_raises_actionable_error_when_qdrant_down() -> None:
    """An unreachable Qdrant surfaces as a readable, actionable tool error —
    not a raw client exception the agent can't interpret."""
    settings = Settings(qdrant_url="http://127.0.0.1:9")  # nothing listens here
    retriever = Retriever(settings)
    try:
        with pytest.raises(RuntimeError, match="Qdrant unreachable"):
            await retriever.retrieve("Restormer CBSD68 PSNR")
    finally:
        await retriever.close()
