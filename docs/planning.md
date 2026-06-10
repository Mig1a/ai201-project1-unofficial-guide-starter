# Project Planning

## Domain Choice

The domain is student-generated reviews of Computer Science professors at George Mason University. This is useful because official course catalogs rarely describe lecture clarity, exam style, grading strictness, workload, office-hour helpfulness, or how students feel after taking a course.

Rate My Professors is a source, not the domain. The domain is the set of student review evidence about GMU CS professors; Rate My Professors is only where these manually collected PDFs came from.

## Data Collection Method

Five professor pages were manually saved as PDFs and placed in `data/raw/`:

- Ahmed Zaman
- Sanjeev Setia
- Jana Kosecka
- Alexander Laufer
- Wassim (Wes) Masri

The project does not scrape Rate My Professors. The ingestion script only processes local PDFs.

## Ingestion Plan

`scripts/ingest.py` loads every PDF from `data/raw/` with `pypdf`, extracts text, cleans it, and writes two documents per professor:

- a cleaned full-review document
- a structured summary/theme document

This produces 10 processed documents in `data/processed/`, satisfying the minimum document requirement.

## Cleaning Plan

The cleaner removes obvious page noise:

- ad and sale text
- repeated Rate My Professors headers
- URL-like fragments
- page numbers
- footer policy text
- duplicate whitespace

The cleaner is intentionally conservative. It removes repeated site boilerplate while preserving review wording that may be useful evidence.

## Chunking Strategy

The system uses:

```python
chunk_size = 500
chunk_overlap = 100
```

`scripts/rag.py` uses `RecursiveCharacterTextSplitter` when LangChain is installed and falls back to a simple overlapping character splitter when it is not. The chunk metadata includes source file, professor name, document type, chunk number, and total chunks.

Student reviews are short, opinion-heavy, and topic-specific. Smaller chunks help the retriever find precise evidence about feedback, grading, exams, lecture clarity, organization, or office hours. The overlap preserves context across review boundaries.

Alternatives considered:

- Larger chunks: better global context but weaker precision for short review claims.
- Review-level chunks: cleaner if review boundaries are reliable, but PDF extraction does not preserve those boundaries consistently.
- Token-based splitting: useful for strict model context budgeting, but character-based splitting is simpler and reproducible for this small corpus.

In the current processed corpus, the chunking configuration produces 160 chunks in this environment.

## Embedding Model Choice

The assignment does not require a specific embedding provider. The implementation is configurable with `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL`.

Current default:

```txt
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

Why this default was chosen: it is fast, inexpensive, and strong enough for a small English review corpus. The corpus is short and mostly natural-language opinion text, so a general-purpose embedding model is appropriate.

Alternatives considered:

- Sentence Transformers: local, reproducible, no API cost, but requires model downloads and local compute.
- BGE models: strong retrieval quality, good local option, more setup and disk use.
- E5 models: strong query/document retrieval framing, good local option, requires prompt-format care.
- Cohere embeddings: strong hosted retrieval models, API dependency and cost.
- Voyage embeddings: high retrieval quality, API dependency and cost.

Production tradeoffs:

- Cost: local models avoid per-call fees; hosted APIs charge by usage.
- Retrieval quality: larger or retrieval-specialized models can improve subtle comparisons.
- Multilingual support: not critical for this English corpus, but important if multilingual reviews are added.
- Local vs API: local improves privacy and offline reproducibility; API models simplify setup and often improve quality.
- Performance: local latency depends on hardware; hosted latency depends on network and provider.

## Vector Database Choice

The assignment does not require a specific vector database. The implementation is configurable with `VECTOR_STORE`.

Current default:

```txt
VECTOR_STORE=chroma
```

Why this default was chosen: ChromaDB is easy to persist locally in `vectordb/`, simple to reset, and appropriate for a small demo corpus.

Alternatives considered:

- FAISS: fast and lightweight for local search, but requires more custom metadata persistence.
- Pinecone: hosted and scalable, but adds external service setup and cost.
- Weaviate: production-oriented with rich metadata and hybrid search, but heavier operationally.
- Qdrant: strong vector database with local and hosted options, but more setup than needed here.

Production tradeoffs:

- Local stores are easier for reproducibility and demos.
- Hosted stores are better for scale, monitoring, backups, concurrent users, and access control.
- Metadata support matters because every answer must cite source documents.

## Retrieval Strategy

For each query, the system embeds the question and retrieves the top 5 semantically similar chunks from the configured vector store. The UI shows those retrieved chunks and metadata so the evidence trail is visible.

Top 5 balances answer coverage and prompt size. Fewer chunks can miss comparison evidence; many more chunks can dilute the prompt and make citations harder to inspect.

## LLM Choice

The assignment does not require a specific LLM. The implementation is configurable with `LLM_PROVIDER` and `LLM_MODEL`.

Current default:

```txt
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

Why this default was chosen: it is affordable, responsive, and good at following structured grounding and citation instructions for a class demo.

Alternatives considered:

- Gemini: strong general model quality, API dependency.
- Claude: strong instruction following and synthesis, API dependency.
- Ollama: local and private, but quality/speed depend on hardware and selected model.
- Local HuggingFace models: maximum control and offline operation, but more setup and likely more prompt-tuning.

Production tradeoffs:

- API models simplify deployment but create cost, privacy, and vendor-dependency concerns.
- Local models reduce data-sharing concerns but require hardware, monitoring, and model maintenance.
- Stronger models may handle mixed or conflicting reviews better, but the grounding prompt and retrieval quality matter more than raw model size.

## Framework Choice

The solution may use LangChain, LlamaIndex, or custom Python. This implementation uses custom Python orchestration with optional LangChain helpers for splitting and provider wrappers. That keeps each RAG stage visible for grading while still using proven library components where useful.

The query interface may be Streamlit, Flask, FastAPI, notebook, or CLI. This implementation uses Streamlit because it is quick to demo and naturally supports input, answer output, source lists, and expandable retrieved chunks.

## Grounding Strategy

The LLM prompt requires the model to:

- answer only from retrieved context
- avoid outside knowledge
- avoid invented claims
- say when evidence is missing
- cite professor and document names
- mention mixed or polarized evidence when retrieved reviews conflict

The response format is:

```txt
Answer:
[Grounded answer]

Evidence:
- [Short cited evidence from source/chunk]

Sources:
- [source file]
```

## Evaluation Plan

The five evaluation questions are:

| # | Question | Expected answer basis |
|---|----------|-----------------------|
| 1 | Which professor is most often described as making difficult CS concepts easier to understand? | Evidence about clear explanations, concepts, and understanding. |
| 2 | Which professor has reviews mentioning very hard exams or tough grading? | Evidence about hard exams, test-heavy courses, tough grading, or grading inconsistency. |
| 3 | Which professor has reviews that mention disorganization or poor structure? | Evidence about disorganized assignments, confusing course structure, or poor organization. |
| 4 | Which professor has mixed or polarized reviews? | Evidence showing both strongly positive and strongly negative reviews. |
| 5 | Which professor has reviews recommending office hours? | Evidence about office hours, asking questions, availability, or help. |

Each evaluation records the question, ground truth, generated answer, retrieved chunks, retrieved sources, retrieval accuracy, response accuracy, notes, and one failure case.

## Anticipated Challenges

PDF extraction can merge headers, ratings, and review text into long sentences, making cleaning imperfect.

Broad comparison questions may retrieve evidence for multiple professors but still lack enough balanced evidence to rank them confidently.
