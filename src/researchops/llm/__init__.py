"""LLM provider package.

A single async interface (`BaseLLM.chat`) over three interchangeable backends:
- `deepseek` / `qwen`: cloud APIs
- `vllm`: self-hosted OpenAI-compatible server (private deployment demo)

This is the seam that lets the rest of the agent be provider-agnostic, and is the
foundation for later cost/latency comparison work.
"""

from researchops.llm.providers import BaseLLM, build_llm

__all__ = ["BaseLLM", "build_llm"]
