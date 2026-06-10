# The Unofficial Guide

The Unofficial Guide is a production-style RAG system for asking plain-language questions about George Mason University Computer Science professors using student review documents.

The domain is GMU CS professor student reviews. Rate My Professors is only the collection source for the manually saved PDFs; the system does not scrape Rate My Professors or use outside professor knowledge.

## Project Overview

Students can ask questions such as:

- Which professor gives useful feedback?
- Which professor explains difficult CS concepts clearly?
- Which professor has the hardest exams?
- Which professor is described as disorganized?
- Which professor has mixed or polarized reviews?
- Which professor has reviews recommending office hours?

The system ingests documents, chunks them, embeds chunks into a vector store, retrieves relevant evidence, and generates grounded answers with source attribution. The implementation includes default technology choices, but the assignment does not require any specific vendor, model, framework, or database.

## Data Source

The raw PDFs were manually collected from Rate My Professors and saved into `data/raw/`.

| # | Processed document | Type | Source PDF |
|---|--------------------|------|------------|
| 1 | `ahmed_zaman_reviews.md` | cleaned reviews | Ahmed Zaman PDF |
| 2 | `ahmed_zaman_summary.md` | generated theme summary | Ahmed Zaman PDF |
| 3 | `sanjeev_setia_reviews.md` | cleaned reviews | Sanjeev Setia PDF |
| 4 | `sanjeev_setia_summary.md` | generated theme summary | Sanjeev Setia PDF |
| 5 | `jana_kosecka_reviews.md` | cleaned reviews | Jana Kosecka PDF |
| 6 | `jana_kosecka_summary.md` | generated theme summary | Jana Kosecka PDF |
| 7 | `alexander_laufer_reviews.md` | cleaned reviews | Alexander Laufer PDF |
| 8 | `alexander_laufer_summary.md` | generated theme summary | Alexander Laufer PDF |
| 9 | `wes_masri_reviews.md` | cleaned reviews | Wassim (Wes) Masri PDF |
| 10 | `wes_masri_summary.md` | generated theme summary | Wassim (Wes) Masri PDF |

The summary documents are generated from extracted review text only. They organize evidence around lecture clarity, difficulty, grading style, exams/quizzes, office hours/helpfulness, organization, workload, and student sentiment.

## Setup

Use Python 3.12.

```bash
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` to select providers. The default path is fully local and does not require OpenAI:

```txt
EMBEDDING_PROVIDER=local_tfidf
EMBEDDING_MODEL=local_tfidf
LLM_PROVIDER=extractive
LLM_MODEL=extractive
VECTOR_STORE=json
```

This local baseline uses TF-IDF-style sparse vectors stored in a JSON vector index and an extractive grounded answer generator. It is intended to be reproducible with no API key. You can switch to OpenAI, Ollama, ChromaDB, or Sentence Transformers by changing `.env` and installing the corresponding dependencies.

For local generation with Ollama, set `LLM_PROVIDER=ollama`, choose an installed Ollama model, and set `OLLAMA_URL` if your server is not on the default local endpoint.

## Run Ingestion

The ingestion pipeline loads PDFs from `data/raw/`, extracts text, removes obvious page noise, and writes cleaned review and summary files to `data/processed/`.

```bash
python scripts/ingest.py
```

Expected output: 5 raw PDFs become 10 processed Markdown documents.

## Build Embeddings

The embedding pipeline chunks processed documents with:

```python
chunk_size = 500
chunk_overlap = 100
```

The current default implementation uses LangChain's `RecursiveCharacterTextSplitter` when available, with a small built-in fallback splitter for reproducibility. Chunks include metadata for source file, professor, document type, chunk number, and total chunks.

```bash
python scripts/embed.py
```

## Run The App

The simplest no-API demo is the command-line interface:

```bash
python scripts/query.py "Which professor explains difficult CS concepts clearly?"
```

This uses the local vector index and extractive answer generator.

The project also includes a Streamlit interface, but it requires installing Streamlit:

```bash
streamlit run app.py
```

The interface includes a question input, generated answer, cited source list, retrieved chunk expanders, and a warning that answers are based only on the collected review documents.

## Run Evaluation

```bash
python scripts/evaluate.py
```

The evaluation script runs five test questions when dependencies, a built vector store, and the selected provider credentials are available. If setup is incomplete, it still writes `docs/evaluation_report.md` with the test cases and the blocking setup error.

## Configurable Technology Choices

### Embeddings

The embedding implementation is configurable through:

```txt
EMBEDDING_PROVIDER=
EMBEDDING_MODEL=
```

Implemented providers:

- `local_tfidf` by default for a no-API reproducible baseline
- `openai` with models such as `text-embedding-3-small`
- `sentence_transformers` for local models such as BGE, E5, or MiniLM when `sentence-transformers` is installed

Alternatives considered include OpenAI embeddings, Sentence Transformers, BGE models, E5 models, Cohere embeddings, and Voyage embeddings.

Default rationale: `local_tfidf` is free, reproducible, and requires no API key or model download, which makes it the safest baseline for the assignment demo. It is less semantically rich than OpenAI, BGE, E5, Cohere, or Voyage embeddings, so it may miss paraphrases. A local BGE or E5 model would improve retrieval quality while remaining offline, but it requires local disk, memory, and model downloads. API embedding models may improve subtle comparative retrieval but add cost, latency, and vendor dependency. Multilingual support is not a major requirement for the current English corpus, but would matter if reviews in other languages were added.

### LLM

Grounded generation is configurable through:

```txt
LLM_PROVIDER=
LLM_MODEL=
```

Implemented providers:

- `extractive` by default for a no-API grounded baseline
- `openai` with models such as `gpt-4o-mini`
- `ollama` for local generation through a running Ollama server

Alternatives considered include OpenAI, Gemini, Claude, Ollama, and local HuggingFace models.

Default rationale: `extractive` generation is not as fluent as an LLM, but it is maximally grounded because it builds the answer directly from retrieved review sentences and requires no API key. `gpt-4o-mini` is a practical optional model because it is low-latency, affordable, and reliable at following citation/grounding instructions. Ollama improves privacy and avoids API costs, but output quality and speed depend on the local model and hardware. Claude or Gemini could be strong alternatives if their APIs are preferred.

### Vector Store

Vector storage is configurable through:

```txt
VECTOR_STORE=
```

This implementation includes a JSON local vector index by default and a ChromaDB adapter for richer vector storage. Alternatives considered include FAISS, Pinecone, Weaviate, and Qdrant.

Default rationale: the JSON vector index is dependency-free and easiest to reproduce. ChromaDB is still a good optional local vector database because it is simple to persist and reset. FAISS would be faster and lighter for local dense vectors, but requires more custom metadata handling. Pinecone, Weaviate, and Qdrant are stronger production choices for hosted scaling, access control, monitoring, and larger corpora, but they add operational setup that is unnecessary for this small project.

### Frameworks

The project uses a small custom Python architecture with optional LangChain helpers and a Streamlit UI. This keeps the pipeline readable while still satisfying ingestion, chunking, embeddings, vector storage, semantic retrieval, grounded generation, source attribution, evaluation, and documentation requirements.

LangChain and LlamaIndex were considered because they provide many RAG utilities. A custom implementation was kept for the project flow so each stage is visible and easy to explain. Streamlit was chosen over Flask/FastAPI because the assignment needs a demonstrable query interface more than a production HTTP API.

## Design Decisions

- The retriever returns top 5 chunks so the LLM has enough evidence without flooding the prompt.
- Smaller 500-character chunks fit short opinion-heavy reviews where individual claims about exams, grading, lectures, or office hours matter.
- 100-character overlap preserves context when a review or theme sentence crosses a chunk boundary.
- Metadata is attached to every chunk so answers can cite source files and professors.
- Provider settings are explicit in `.env` so the same architecture can be run with API-hosted or local models.

## Limitations

- The corpus covers only five manually collected Rate My Professors PDF pages.
- The system cannot answer questions unsupported by those PDFs.
- PDF extraction can leave small fragments of site metadata even after cleaning.
- Rate My Professors reviews are subjective and may be biased or unrepresentative.
- End-to-end retrieval and generation require installing dependencies and configuring the selected providers.

## Future Improvements

- Add more manually collected professor review documents.
- Improve review-boundary detection so each review can become a cleaner unit.
- Add reranking or source diversity constraints for broad comparison questions.
- Add adapters for FAISS, Qdrant, Claude, Gemini, Cohere, and Voyage.
- Add human-verified evaluation labels after running the app with the selected model stack.
