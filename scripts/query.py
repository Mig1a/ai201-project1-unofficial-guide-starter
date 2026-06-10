from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask The Unofficial Guide a question.")
    parser.add_argument("question", nargs="*", help="Question to ask the RAG system.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    result = answer_question(question)
    print(result["answer"])
    print("\nRetrieved chunks:")
    for index, chunk in enumerate(result["chunks"], start=1):
        metadata = chunk["metadata"]
        print(
            f"{index}. {metadata['professor']} | {metadata['source_file']} | "
            f"chunk {metadata['chunk_number']}/{metadata['total_chunks']}"
        )


if __name__ == "__main__":
    main()
