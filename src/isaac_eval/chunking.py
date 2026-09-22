from __future__ import annotations
import re
from pathlib import Path


def extract_pdf_text(path: str | Path) -> str:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(parts)
    # PDF extraction leaves hyphenated line breaks and ragged whitespace that
    # would otherwise inflate word counts and pollute chunk boundaries.
    text = text.replace("­", "").replace("‐", "-")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_paper(text: str, size_words: int, overlap_words: int, paper_id: str = "") -> list[dict]:
    words = text.split()
    if size_words <= 0:
        raise ValueError("size_words must be positive")
    if overlap_words >= size_words:
        raise ValueError("overlap_words must be smaller than size_words")

    step = size_words - overlap_words
    chunks: list[dict] = []
    for start in range(0, len(words), step):
        window = words[start:start + size_words]
        if not window:
            break
        body = " ".join(window)
        chunks.append({
            "chunk_id": f"{paper_id}::{len(chunks):04d}" if paper_id else f"{len(chunks):04d}",
            "paper_id": paper_id,
            "text": body,
            "n_words": len(window),
            "n_chars": len(body),
        })
        if start + size_words >= len(words):
            break
    return chunks
