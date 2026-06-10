from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
COLLECTION_NAME = "unofficial_guide_reviews"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


def require_openai_key() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")


def professor_from_filename(path: Path) -> str:
    base = path.stem.replace("_reviews", "").replace("_summary", "")
    return " ".join(part.capitalize() for part in base.split("_"))


def doc_type_from_filename(path: Path) -> str:
    return "summary" if path.stem.endswith("_summary") else "reviews"


def load_processed_documents() -> list[dict[str, Any]]:
    paths = sorted(PROCESSED_DIR.glob("*.md")) + sorted(PROCESSED_DIR.glob("*.txt"))
    if not paths:
        raise RuntimeError("No processed documents found. Run `python scripts/ingest.py` first.")
    docs = []
    for path in paths:
        docs.append(
            {
                "path": path,
                "text": path.read_text(encoding="utf-8"),
                "source_file": path.name,
                "professor": professor_from_filename(path),
                "document_type": doc_type_from_filename(path),
            }
        )
    return docs


def split_text_recursively(text: str) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            chunks = []
            start = 0
            while start < len(text):
                chunks.append(text[start : start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
            return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def split_documents() -> tuple[list[str], list[dict[str, Any]], list[str]]:
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for doc in load_processed_documents():
        chunks = split_text_recursively(doc["text"])
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            safe_source = re.sub(r"[^a-zA-Z0-9_]+", "_", doc["source_file"])
            ids.append(f"{safe_source}_{index}")
            texts.append(chunk)
            metadatas.append(
                {
                    "source_file": doc["source_file"],
                    "professor": doc["professor"],
                    "document_type": doc["document_type"],
                    "chunk_number": index,
                    "total_chunks": total,
                }
            )
    return texts, metadatas, ids


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_vector_store(reset: bool = True) -> int:
    from langchain_openai import OpenAIEmbeddings

    require_openai_key()
    if reset and VECTORDB_DIR.exists():
        shutil.rmtree(VECTORDB_DIR)
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    texts, metadatas, ids = split_documents()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(texts)

    collection = get_collection()
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
    return len(texts)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    from langchain_openai import OpenAIEmbeddings

    require_openai_key()
    if not VECTORDB_DIR.exists():
        raise RuntimeError("vectordb/ does not exist. Run `python scripts/embed.py` first.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = embeddings.embed_query(query)
    collection = get_collection()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append({"text": text, "metadata": metadata, "distance": distance})
    return chunks


def build_context(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        lines.append(
            f"[Chunk {index}] Professor: {metadata['professor']} | "
            f"Source: {metadata['source_file']} | Type: {metadata['document_type']} | "
            f"Chunk: {metadata['chunk_number']}/{metadata['total_chunks']}\n{chunk['text']}"
        )
    return "\n\n".join(lines)


def answer_question(query: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    from langchain_openai import ChatOpenAI

    chunks = retrieve(query, top_k=TOP_K)
    context = build_context(chunks)
    sources = sorted({chunk["metadata"]["source_file"] for chunk in chunks})

    llm = ChatOpenAI(model=model, temperature=0)
    messages = [
        (
            "system",
            "You answer questions about GMU Computer Science professors using only the retrieved "
            "student review context. Do not use outside knowledge. Do not invent claims. If the "
            "retrieved context is insufficient, say there is not enough information. Cite professor "
            "and document names. Mention mixed or polarized evidence when the retrieved reviews "
            "conflict. Use this exact structure:\n\nAnswer:\n[Grounded answer]\n\nEvidence:\n"
            "- [Short cited evidence from source/chunk]\n\nSources:\n- [source file]",
        ),
        ("human", f"Question: {query}\n\nRetrieved context:\n{context}"),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "chunks": chunks, "sources": sources}
