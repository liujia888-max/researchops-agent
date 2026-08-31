"""Command policy for ``submit_job``: a conservative denylist that narrows blast radius.

``submit_job`` remains arbitrary remote execution by design — that is what a training
job *is*. This module is NOT a sandbox and is not a substitute for human approval. It
rejects the most obviously destructive commands *before* they reach the host, so a bad
LLM proposal or an unapproved caller cannot trivially wipe the machine, exfiltrate data,
or take the box down.

The list is deliberately small and conservative: a research lab legitimately runs
``rm -rf ./checkpoints``, ``pip install``, and ``python train.py``, so those are NOT
blocked. Blocking is reserved for actions whose blast radius escapes the working
directory, escalates privilege, opens a reverse shell, or shuts the host down. (A
recursive ``rm`` of ``/``, ``/*``, ``~``, or ``$HOME`` is blocked; ``rm -rf <workdir>/x``
is not — the command already runs ``cd <workdir>`` first, so relative deletes stay inside
the sandbox by construction.)
"""

from __future__ import annotations

import re

# Each rule is a compiled pattern plus a short human-readable reason. Patterns are
# matched case-insensitively and anchored to whole words where possible (``\b``) to
# avoid false positives on harmless substrings.
_DENYLIST: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsudo\b", re.IGNORECASE), "privilege escalation"),
    (re.compile(r"\bmkfs\b", re.IGNORECASE), "filesystem creation"),
    (re.compile(r"\bdd\b\s+.*\bof=/dev/", re.IGNORECASE), "raw device write"),
    (re.compile(r">\s*/dev/(?:sd|nvme|hd|vd|mmcblk)", re.IGNORECASE), "raw device write"),
    (re.compile(r"\breboot\b|\bshutdown\b|\bpoweroff\b|\bhalt\b", re.IGNORECASE), "host shutdown"),
    (re.compile(r"\bsystemctl\b", re.IGNORECASE), "service control"),
    (re.compile(r"/dev/(?:tcp|udp)/", re.IGNORECASE), "reverse shell"),
    (re.compile(r"\b(?:nc|ncat|netcat)\b.*-e\b", re.IGNORECASE), "reverse shell"),
    (re.compile(r"\b(?:curl|wget)\b.*\|\s*(?:sh|bash)\b", re.IGNORECASE), "pipe-to-shell"),
    (re.compile(r"\bpkill\b|\bkillall\b", re.IGNORECASE), "process termination"),
    (
        re.compile(
            r"\brm\s+(?:-[a-zA-Z]*[rf][a-zA-Z]*\s+)+"
            r"(?:--(?:no-preserve-root|preserve-root)\s+)*"
            r"(?:/\*|/|\*|~|\$HOME|\$\(HOME\))"
            r"(?=\s|$|&&|;|\||>)",
            re.IGNORECASE,
        ),
        "recursive delete of filesystem root/home",
    ),
]


def validate_command(command: str) -> str | None:
    """Return a human-readable violation for a dangerous command, else ``None``.

    Pure and unit-testable: no I/O, no host access. Called by ``LabClient.submit_job``
    (the transport choke point) and by the agent tool layer (to give the LLM a clean,
    non-retry rejection).
    """
    for pattern, reason in _DENYLIST:
        if pattern.search(command):
            return f"command violates policy ({reason})"
    return None
