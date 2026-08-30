"""Entrypoint: run the labops MCP server over stdio.

Usage: ``python -m researchops.mcp`` (or ``researchops mcp``).
"""

from __future__ import annotations

from researchops.mcp.server import mcp

if __name__ == "__main__":
    mcp.run()
