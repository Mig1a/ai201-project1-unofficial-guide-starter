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

The answer generator is instructed to use only retrieved chunks from the processed documents and cite the professor/document sources used.

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

Then edit `.env`:

```txt
OPENAI_API_KEY=your_key_here
```

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

It uses `RecursiveCharacterTextSplitter`, embeds chunks with OpenAI `text-embedding-3-small`, and persists the ChromaDB vector database to `vectordb/`.

```bash
python scripts/embed.py
```

## Run The App

```bash
streamlit run app.py
```

The interface includes a question input, generated answer, cited source list, retrieved chunk expanders, and a warning that answers are based only on the collected review documents.

## Run Evaluation

```bash
python scripts/evaluate.py
```

The evaluation script runs five test questions when dependencies, `vectordb/`, and `OPENAI_API_KEY` are available. If setup is incomplete, it still writes `docs/evaluation_report.md` with the test cases and the blocking setup error.

## Design Decisions

- `text-embedding-3-small` was chosen because it is fast and cost-effective for a small student-review corpus.
- ChromaDB was chosen because it is simple to persist locally in `vectordb/` and easy to demo.
- The retriever returns top 5 chunks so the LLM has enough evidence without flooding the prompt.
- Smaller 500-character chunks fit short opinion-heavy reviews where individual claims about exams, grading, lectures, or office hours matter.
- 100-character overlap preserves context when a review or theme sentence crosses a chunk boundary.

## Limitations

- The corpus covers only five manually collected Rate My Professors PDF pages.
- The system cannot answer questions unsupported by those PDFs.
- PDF extraction can leave small fragments of site metadata even after cleaning.
- Rate My Professors reviews are subjective and may be biased or unrepresentative.
- Building `vectordb/` and generating answers require an OpenAI API key.

## Future Improvements

- Add more manually collected professor review documents.
- Improve review-boundary detection so each review can become a cleaner unit.
- Add reranking or source diversity constraints for broad comparison questions.
- Add human-verified evaluation labels after running the app with a real API key.
