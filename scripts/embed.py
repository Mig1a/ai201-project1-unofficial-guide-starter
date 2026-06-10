from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag import build_vector_store


def main() -> None:
    try:
        count = build_vector_store(reset=True)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Built vector store with {count} chunks in vectordb/.")


if __name__ == "__main__":
    main()
