"""Tests for the doctor setup check (offline pieces only; network checks are live)."""

from __future__ import annotations

from researchops.config import Settings
from researchops.doctor import check_env, format_report


def test_check_env_reports_provider_and_key_without_leaking_key() -> None:
    settings = Settings(llm_provider="deepseek", deepseek_api_key="sk-secret")
    env = check_env(settings)
    assert env["provider"] == "deepseek"
    assert env["key_set"] is True
    assert "sk-secret" not in str(env)


def test_check_env_reports_missing_key() -> None:
    settings = Settings(llm_provider="qwen", qwen_api_key="")
    env = check_env(settings)
    assert env["provider"] == "qwen"
    assert env["key_set"] is False


def test_format_report_marks_missing_key_and_fallback() -> None:
    result = {
        "env": {"provider": "deepseek", "key_set": False, "dotenv": False},
        "llm": {"ok": False, "error": "boom"},
        "qdrant": {"ok": True, "url": "http://x:6333"},
        "inference": {"ok": False, "url": "http://x:8001", "error": "refused", "fallback": True},
        "labops": {"ok": False, "host": "h", "error": "refused"},
    }
    text = format_report(result)
    assert "MISSING" in text
    assert "OK" in text  # qdrant is reachable
    assert "feature-hash" in text  # inference fallback note
