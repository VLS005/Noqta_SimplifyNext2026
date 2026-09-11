"""Load AWS credentials from local .env files without printing secret values."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env() -> list[str]:
    """Parse KEY=value / export KEY=value files into os.environ.

    Does not override variables already set in the process. Returns the
    names of keys that were newly applied (never the values).
    """
    here = Path(__file__).resolve().parent
    agent_root = here.parent
    repo_root = agent_root.parent.parent
    candidates = [
        agent_root / ".env",
        repo_root / ".env",
    ]
    applied: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if not key:
                continue
            # Repo .env should win over a stale shell session.
            os.environ[key] = value
            applied.append(key)
    return applied
