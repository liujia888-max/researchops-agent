from __future__ import annotations

import pytest

from researchops.config import Settings
from researchops.llm.providers import VLLMLLM, DeepSeekLLM, QwenLLM, build_llm


def test_build_deepseek() -> None:
    s = Settings(llm_provider="deepseek", deepseek_api_key="k")
    assert isinstance(build_llm(s), DeepSeekLLM)


def test_build_qwen() -> None:
    s = Settings(llm_provider="qwen", qwen_api_key="k")
    assert isinstance(build_llm(s), QwenLLM)


def test_build_vllm() -> None:
    s = Settings(llm_provider="vllm")
    assert isinstance(build_llm(s), VLLMLLM)


def test_provider_override_wins_over_settings() -> None:
    s = Settings(llm_provider="deepseek", deepseek_api_key="k")
    assert isinstance(build_llm(s, provider="qwen"), QwenLLM)


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown llm_provider"):
        build_llm(Settings(), provider="nope")
