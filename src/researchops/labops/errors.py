"""Structured errors for the labops layer.

Every failure the MCP tools can hit is mapped to one of these, so the consuming
agent (and the LLM) gets an actionable, machine-readable error instead of a raw
traceback. The LangGraph layer (Phase 2) can `except` these programmatically to
decide whether to retry, prompt the user, or give up.
"""

from __future__ import annotations


class LabopsError(Exception):
    """Base class for all labops failures."""


class HostUnreachableError(LabopsError):
    """The GPU host is down: powered off, network unreachable, or key auth failed."""


class CommandFailedError(LabopsError):
    """A remote command exited non-zero or could not be executed."""


class JobNotFoundError(LabopsError):
    """The requested job id has no screen session and no log file."""


class InvalidJobIdError(LabopsError):
    """A job id failed the ``[A-Za-z0-9_-]`` allowlist (path-traversal guard)."""
