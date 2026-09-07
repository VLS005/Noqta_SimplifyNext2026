"""One-shot: create obstacle-memory-index on DoraDB if missing, then poll until ACTIVE.

Usage (from agents/personalisation-agent):
    python scripts/create_vector_index.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.aws_env import load_local_env  # noqa: E402

load_local_env()

from app.config import Settings  # noqa: E402
from app.tools.memory_store import MemoryStore  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    settings = Settings(create_vector_index_if_missing=True, demo_mode=False)
    store = MemoryStore(settings)

    print(
        f"Table={settings.dynamodb_table_name} region={settings.aws_region} "
        f"index={settings.vector_index_name}",
        flush=True,
    )
    status = store.ensure_vector_index()
    print(f"ensure_vector_index -> {status}", flush=True)
    store.wait_until_vector_index_active(poll_seconds=5.0, timeout_seconds=900.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
