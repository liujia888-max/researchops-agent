"""``researchops doctor`` — a read-only setup check for a fresh clone.

Runs each external dependency the agent needs and prints a per-layer OK/WARN report, so
a new user sees in one command what is ready and what still needs wiring (LLM key ->
Qdrant -> embedding service -> GPU host). Nothing is mutated and every network check is
best-effort with a short timeout; failures are reported, never raised.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from researchops.config import Settings, get_settings


def check_env(settings: Settings) -> dict[str, Any]:
    """Report the selected LLM provider and whether its key is present (never the key)."""
    if settings.llm_provider == "deepseek":
        key = settings.deepseek_api_key
    elif settings.llm_provider == "qwen":
        key = settings.qwen_api_key
    else:
        key = settings.vllm_api_key
    return {
        "provider": settings.llm_provider,
        "key_set": bool(key),
        "dotenv": Path(".env").exists(),
    }


async def check_llm(settings: Settings) -> dict[str, Any]:
    """Ping the configured LLM with a minimal prompt (verifies the key actually works)."""
    from researchops.llm import build_llm
    from researchops.llm.providers import ChatMessage

    try:
        llm = build_llm(settings)
        resp = await asyncio.wait_for(
            llm.chat([ChatMessage(role="user", content="Reply with exactly: pong")]),
            timeout=20,
        )
        return {"ok": True, "model": resp.model}
    except Exception as exc:  # noqa: BLE001 — a doctor reports, never raises
        return {"ok": False, "error": str(exc)[:200]}


async def check_qdrant(settings: Settings) -> dict[str, Any]:
    """Verify the Qdrant server is reachable (collection is created lazily on ingest)."""
    from qdrant_client import AsyncQdrantClient

    try:
        client = AsyncQdrantClient(url=settings.qdrant_url, check_compatibility=False)
        async with asyncio.timeout(5):
            await client.get_collections()
        await client.close()
        return {"ok": True, "url": settings.qdrant_url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": settings.qdrant_url, "error": str(exc)[:200]}


async def check_inference(settings: Settings) -> dict[str, Any]:
    """Check the bge-m3/reranker inference service; report the offline fallback state."""
    import httpx

    url = settings.inference_base_url.rstrip("/")
    result: dict[str, Any] = {"url": url, "fallback": settings.rag_fallback_local}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/health")
        result["ok"] = resp.status_code == 200
        if resp.status_code != 200:
            result["error"] = f"status {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = str(exc)[:200]
    return result


async def check_labops(settings: Settings) -> dict[str, Any]:
    """Verify SSH + ``nvidia-smi`` access to the GPU host (optional dependency)."""
    from researchops.labops import LabClient, SshConnection

    try:
        client = LabClient(SshConnection())
        async with asyncio.timeout(15):
            gpus = await client.gpu_info()
        await client.close()
        return {"ok": True, "gpus": [g.name for g in gpus]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "host": settings.labops_host, "error": str(exc)[:200]}


async def run_doctor() -> dict[str, Any]:
    """Run every check concurrently and return the raw result dict."""
    settings = get_settings()
    env = check_env(settings)
    llm, qdrant, inference, labops = await asyncio.gather(
        check_llm(settings),
        check_qdrant(settings),
        check_inference(settings),
        check_labops(settings),
    )
    return {"env": env, "llm": llm, "qdrant": qdrant, "inference": inference, "labops": labops}


def format_report(result: dict[str, Any]) -> str:
    """Render the doctor result as a plain-text report."""
    env = result["env"]
    lines: list[str] = [
        "== ResearchOps Agent doctor ==",
        (
            f"LLM provider : {env['provider']}"
            f" (key {'set' if env['key_set'] else 'MISSING — set it in .env'})"
        ),
        (
            f".env file     : "
            f"{'present' if env['dotenv'] else 'MISSING — run: cp .env.example .env'}"
        ),
    ]

    rows = (
        ("LLM reachable", result["llm"], ("model",)),
        ("Qdrant", result["qdrant"], ("url",)),
        ("Embedding", result["inference"], ("url",)),
        ("GPU host", result["labops"], ("host", "gpus")),
    )
    for label, r, show_keys in rows:
        if r.get("ok"):
            extra = ", ".join(str(r[k]) for k in show_keys if k in r and r[k]) or ""
            lines.append(f"{label:<13}: OK   {extra}".rstrip())
        else:
            note = r.get("error", "unreachable")
            suffix = ""
            if label == "Embedding" and r.get("fallback"):
                suffix = " (RAG will use offline feature-hash embedding)"
            lines.append(f"{label:<13}: WARN {note}{suffix}")
    return "\n".join(lines)
