# Evaluation Report

This report defines the required five-question evaluation. Generated answers were not auto-filled in this environment because end-to-end retrieval and generation require installing the selected provider dependencies, building the vector store, and configuring the selected model credentials or local model runtime. Running `python scripts/evaluate.py` currently reports the missing setup blocker cleanly.

| # | Question | Ground-truth answer | Generated system answer | Retrieved chunks | Retrieved source files | Retrieval accuracy | Response accuracy | Notes |
|---|----------|---------------------|-------------------------|------------------|------------------------|--------------------|-------------------|-------|
| 1 | Which professor is most often described as making difficult CS concepts easier to understand? | Ahmed Zaman has repeated evidence about making CS330 concepts easier, simplifying unintuitive concepts, and explaining clearly. Sanjeev Setia also has some positive clarity evidence, but fewer extracted reviews. | Not run. | Not run. | Not run. | Not run | Not run | Run `python scripts/embed.py`, then `python scripts/evaluate.py` after installing dependencies and setting `OPENAI_API_KEY`. |
| 2 | Which professor has reviews mentioning very hard exams or tough grading? | Sanjeev Setia and Jana Kosecka have strong evidence about tough grading or hard exams. Wassim Masri also has evidence about test-heavy or unclear exam preparation. | Not run. | Not run. | Not run. | Not run | Not run | Ground truth is based on processed summaries and cleaned review text. |
| 3 | Which professor has reviews that mention disorganization or poor structure? | Alexander Laufer has direct evidence about unorganized assignments and an insanely disorganized class. Jana Kosecka also has evidence about disorganization and unclear grading criteria. | Not run. | Not run. | Not run. | Not run | Not run | A good answer should cite `alexander_laufer_summary.md` and possibly `jana_kosecka_summary.md`. |
| 4 | Which professor has mixed or polarized reviews? | Sanjeev Setia, Jana Kosecka, Alexander Laufer, and Wassim Masri have mixed evidence. Ahmed Zaman is mostly positive but includes some negative or critical comments. | Not run. | Not run. | Not run. | Not run | Not run | The answer should avoid over-ranking without retrieved evidence. |
| 5 | Which professor has reviews recommending office hours? | Ahmed Zaman has direct evidence that office hours are essential and that students should go to office hours. Jana Kosecka has mixed office-hours evidence, including a positive review saying she tries to help and negative reviews about office-hour frustration. | Not run. | Not run. | Not run. | Not run | Not run | Good retrieval should surface office-hour chunks, not just generic helpful labels. |

## Failure Case

A likely failure case is a broad ranking question such as "Who is the most beginner-friendly professor?" The corpus contains subjective student reviews and only five professor pages. Retrieval may return positive chunks from several professors, but the evidence may not support a confident ranking.

## Setup Note

In this workspace, installing the full demo dependency stack failed because the runtime ran out of disk space during package download. The ingestion pipeline, processed corpus, evaluation script fallback path, and Python syntax were verified locally. End-to-end generated answers should be run after installing `requirements.txt`, selecting providers in `.env`, configuring any required API keys or local model servers, and building `vectordb/`.
