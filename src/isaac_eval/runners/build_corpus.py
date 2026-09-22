"""RUNBOOK STEP 2 (문헌 파트) — PDF 청킹 + 임베딩 인덱스 구축.

산출:
  data/corpus/chunks.jsonl        청크 원문
  data/corpus/chunk_index/        청크 단위 인덱스 (경로 A/B용)
  data/corpus/paper_index/        논문 단위 인덱스 (경로 C용)
  results/tables/corpus_stats.csv 4.3절 청크 수·평균 길이
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.chunking import extract_pdf_text, chunk_paper
from isaac_eval.retrieval import build_index, save_index


def main():
    load_env()
    cfg = load_settings()
    papers_dir = Path(get(cfg, "paths.papers_dir"))
    corpus_dir = Path("data/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    size_words = get(cfg, "chunking.size_words")
    overlap_words = get(cfg, "chunking.overlap_words")
    embed_model = get(cfg, "embedding.model")

    all_chunks: list[dict] = []
    paper_records: list[dict] = []

    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        paper_id = pdf_path.stem
        text = extract_pdf_text(pdf_path)
        chunks = chunk_paper(text, size_words, overlap_words, paper_id=paper_id)
        all_chunks.extend(chunks)
        # 논문 단위 인덱스(경로 C)는 앞부분(제목·초록·서론)을 논문 대표 텍스트로 쓴다.
        paper_records.append({
            "paper_id": paper_id,
            "n_words": len(text.split()),
            "n_chunks": len(chunks),
            "summary_text": " ".join(text.split()[:400]),
        })
        print(f"{paper_id}: {len(text.split())} words -> {len(chunks)} chunks")

    with open(corpus_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nembedding {len(all_chunks)} chunks with {embed_model} ...")
    chunk_index = build_index(all_chunks, text_key="text", id_key="chunk_id", model=embed_model, verbose=True)
    save_index(chunk_index, corpus_dir / "chunk_index")

    print(f"embedding {len(paper_records)} paper-level records ...")
    paper_index = build_index(paper_records, text_key="summary_text", id_key="paper_id", model=embed_model)
    save_index(paper_index, corpus_dir / "paper_index")

    tables_dir = Path(get(cfg, "paths.results_dir")) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "corpus_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "n_words", "n_chunks"])
        for record in paper_records:
            writer.writerow([record["paper_id"], record["n_words"], record["n_chunks"]])
        writer.writerow(["TOTAL", sum(r["n_words"] for r in paper_records), len(all_chunks)])
        mean_words = sum(c["n_words"] for c in all_chunks) / len(all_chunks)
        writer.writerow(["mean_words_per_chunk", round(mean_words, 2), ""])
        writer.writerow(["chunk_size_words", size_words, ""])
        writer.writerow(["overlap_words", overlap_words, ""])
        writer.writerow(["embedding_model", embed_model, chunk_index.vectors.shape[1]])

    print(f"\ntotal chunks: {len(all_chunks)}")
    print(f"embedding dim: {chunk_index.vectors.shape[1]}")
    print(f"wrote {tables_dir / 'corpus_stats.csv'}")


if __name__ == "__main__":
    main()
