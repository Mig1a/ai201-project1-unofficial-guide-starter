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

`scripts/rag.py` uses `RecursiveCharacterTextSplitter` when LangChain is installed. The chunk metadata includes source file, professor name, document type, chunk number, and total chunks.

Student reviews are short, opinion-heavy, and topic-specific. Smaller chunks help the retriever find precise evidence about feedback, grading, exams, lecture clarity, organization, or office hours. The overlap preserves context across review boundaries.

In the current processed corpus, the chunking configuration produces 160 chunks in this environment.

## Embedding Model Choice

The embedding model is OpenAI `text-embedding-3-small`.

This is a good fit for the project because it balances cost, speed, and quality. A larger model such as `text-embedding-3-large` may improve retrieval quality, especially for subtle comparison questions, but it costs more and is likely unnecessary for this small corpus. A local embedding model would reduce API dependency but may produce weaker retrieval quality without tuning.

## Vector Database Choice

The vector store is ChromaDB persisted to `vectordb/`. ChromaDB is simple to run locally, easy to reset during development, and appropriate for a class project demo.

## Retrieval Strategy

For each query, the system embeds the question and retrieves the top 5 semantically similar chunks from ChromaDB. The UI shows those retrieved chunks and metadata so the evidence trail is visible.

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
