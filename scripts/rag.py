from __future__ import annotations

import os
import re
import shutil
import json
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
COLLECTION_NAME = "unofficial_guide_reviews"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")
        return

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def setting(name: str, default: str) -> str:
    load_environment()
    return os.getenv(name, default).strip()


def require_api_key(provider: str) -> None:
    key_name = f"{provider.upper()}_API_KEY"
    if not os.getenv(key_name):
        raise RuntimeError(
            f"{key_name} is missing. Copy .env.example to .env and add the key, "
            f"or choose a local provider in .env."
        )


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
    vector_store = setting("VECTOR_STORE", "chroma").lower()
    if vector_store != "chroma":
        raise RuntimeError(
            f"VECTOR_STORE={vector_store} is configured, but this implementation currently "
            "includes only the ChromaDB adapter. Add a FAISS, Qdrant, Pinecone, or Weaviate "
            "adapter before using that setting."
        )
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    provider = setting("EMBEDDING_PROVIDER", "openai").lower()
    model = setting("EMBEDDING_MODEL", "text-embedding-3-small")

    if provider == "openai":
        require_api_key("openai")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model).embed_documents(texts)

    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model)
        return encoder.encode(texts, normalize_embeddings=True).tolist()

    raise RuntimeError(
        f"Unsupported EMBEDDING_PROVIDER={provider}. Supported providers in this "
        "implementation are openai and sentence_transformers."
    )


def embed_query(query: str) -> list[float]:
    provider = setting("EMBEDDING_PROVIDER", "openai").lower()
    model = setting("EMBEDDING_MODEL", "text-embedding-3-small")

    if provider == "openai":
        require_api_key("openai")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model).embed_query(query)

    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model)
        return encoder.encode(query, normalize_embeddings=True).tolist()

    raise RuntimeError(
        f"Unsupported EMBEDDING_PROVIDER={provider}. Supported providers in this "
        "implementation are openai and sentence_transformers."
    )


def build_vector_store(reset: bool = True) -> int:
    if reset and VECTORDB_DIR.exists():
        shutil.rmtree(VECTORDB_DIR)
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    texts, metadatas, ids = split_documents()
    vectors = embed_documents(texts)

    collection = get_collection()
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
    return len(texts)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    if not VECTORDB_DIR.exists():
        raise RuntimeError("vectordb/ does not exist. Run `python scripts/embed.py` first.")
    query_vector = embed_query(query)
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


def grounded_prompt(query: str, context: str) -> str:
    return (
        "You answer questions about GMU Computer Science professors using only the retrieved "
        "student review context. Do not use outside knowledge. Do not invent claims. If the "
        "retrieved context is insufficient, say there is not enough information. Cite professor "
        "and document names. Mention mixed or polarized evidence when the retrieved reviews "
        "conflict. Use this exact structure:\n\nAnswer:\n[Grounded answer]\n\nEvidence:\n"
        "- [Short cited evidence from source/chunk]\n\nSources:\n- [source file]\n\n"
        f"Question: {query}\n\nRetrieved context:\n{context}"
    )


def generate_answer(query: str, context: str, model: str | None = None) -> str:
    provider = setting("LLM_PROVIDER", "openai").lower()
    selected_model = model or setting("LLM_MODEL", "gpt-4o-mini")
    prompt = grounded_prompt(query, context)

    if provider == "openai":
        require_api_key("openai")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=selected_model, temperature=0)
        response = llm.invoke(
            [
                ("system", "Follow the user's grounding and citation instructions exactly."),
                ("human", prompt),
            ]
        )
        return response.content

    if provider == "ollama":
        payload = json.dumps({"model": selected_model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            setting("OLLAMA_URL", "http://localhost:11434/api/generate"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("response", "")

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER={provider}. Supported providers in this implementation "
        "are openai and ollama."
    )


def answer_question(query: str, model: str | None = None) -> dict[str, Any]:
    chunks = retrieve(query, top_k=TOP_K)
    context = build_context(chunks)
    sources = sorted({chunk["metadata"]["source_file"] for chunk in chunks})
    answer = generate_answer(query, context, model=model)
    return {"answer": answer, "chunks": chunks, "sources": sources}
