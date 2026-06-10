from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess import (
    build_reviews_document,
    build_summary_document,
    clean_text,
    infer_professor_name,
    professor_slug,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    page_text = []
    for page in reader.pages:
        page_text.append(page.extract_text() or "")
    return "\n".join(page_text)


def ingest_pdf(pdf_path: Path) -> list[Path]:
    professor_name = infer_professor_name(pdf_path)
    slug = professor_slug(professor_name)
    cleaned = clean_text(extract_pdf_text(pdf_path))
    if not cleaned:
        raise ValueError(f"No extractable text found in {pdf_path.name}")

    reviews_path = PROCESSED_DIR / f"{slug}_reviews.md"
    summary_path = PROCESSED_DIR / f"{slug}_summary.md"
    reviews_path.write_text(build_reviews_document(professor_name, cleaned), encoding="utf-8")
    summary_path.write_text(build_summary_document(professor_name, cleaned), encoding="utf-8")
    return [reviews_path, summary_path]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in data/raw/. Add the manually collected Rate My Professors PDFs first.")

    written = []
    for pdf_path in pdfs:
        written.extend(ingest_pdf(pdf_path))

    print(f"Ingested {len(pdfs)} PDFs and wrote {len(written)} processed documents:")
    for path in written:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
