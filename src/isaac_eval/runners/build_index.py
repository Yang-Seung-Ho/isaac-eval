"""RUNBOOK STEP 2 — 카탈로그 + 표 2(마스킹 실측) + 재현성 스냅샷 생성.

청킹/임베딩 인덱스는 비정형 질의 범위(STEP 5)에 속하므로 여기서는 만들지 않는다.
"""
from __future__ import annotations
import csv
import json
import platform
import sys
from pathlib import Path

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.catalog import build_catalog_entry, save_catalog
from isaac_eval.gold import load_dataset
from isaac_eval.ontology import extract_structural_ontology, masking_report


def main():
    load_env()
    cfg = load_settings()

    entry = build_catalog_entry(cfg)
    catalog_path = f"{get(cfg, 'paths.catalog_dir')}/jeju_phytoplankton.json"
    save_catalog(entry, catalog_path)

    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
    ontology = extract_structural_ontology(df)
    report = masking_report(df, ontology)

    results_dir = Path(get(cfg, "paths.results_dir"))
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(tables_dir / "table2_masking.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "value"])
        writer.writerow(["columns_total", report["columns_total"]])
        writer.writerow(["numeric_masked", report["numeric_masked"]])
        writer.writerow(["numeric_masked_pct", report["numeric_masked_pct"]])
        writer.writerow(["categorical_enumerated", report["categorical_enumerated"]])
        writer.writerow(["payload_chars", report["payload_chars"]])
        writer.writerow(["leaked_values_count", len(report["leaked_values"])])
        writer.writerow(["nan_cells_in_stations", report["nan_cells_in_stations"]])
        for col, values in report["enumerated_values"].items():
            writer.writerow([f"enumerated::{col}", " | ".join(map(str, values))])

    import pandas as pd

    # 논문 5.7절이 요구하는 항목을 한 파일에 모은다: 모델·디코딩 파라미터, 임베딩과
    # 벡터 차원, 검색 상위 K, 청크 크기와 중첩, DOMAIN_TERMS 전체 목록과 확정 임계값,
    # 샌드박스 격리 방식·제한 시간·재시도 예산.
    domain_terms = json.loads(Path("config/domain_terms.json").read_text(encoding="utf-8"))
    corpus_stats = {}
    chunks_path = Path("data/corpus/chunks.jsonl")
    if chunks_path.exists():
        chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        papers = sorted({c["paper_id"] for c in chunks})
        corpus_stats = {
            "n_papers": len(papers),
            "papers": papers,
            "n_chunks": len(chunks),
            "total_words": sum(c["n_words"] for c in chunks),
            "mean_words_per_chunk": round(sum(c["n_words"] for c in chunks) / len(chunks), 2),
        }

    queries = {}
    for name, path in (("structured", get(cfg, "paths.queries_structured")),
                       ("unstructured", "queries/unstructured.jsonl"),
                       ("hybrid", "queries/hybrid.jsonl")):
        p = Path(path)
        if p.exists():
            queries[name] = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())

    models_used = sorted({
        json.loads(p.read_text(encoding="utf-8")).get("config_snapshot", {}).get("model")
        for p in (Path(get(cfg, "paths.results_dir")) / "runs").glob("*.json")
    } - {None}) if (Path(get(cfg, "paths.results_dir")) / "runs").exists() else []

    reproducibility = {
        "config": cfg,
        "dataset": {
            "path": cfg["paths"]["dataset"],
            "sheet": cfg["paths"]["sheet"],
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
        },
        "corpus": corpus_stats,
        "queries": queries,
        "masking_report": report,
        "domain_terms": {"count": len(domain_terms), "terms": domain_terms},
        "models_used_in_runs": models_used,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "notes": [
            "sandbox.mode=subprocess (Docker not installed on the measurement host); "
            "isolated interpreter (python -I), network blocked, temp cwd",
            "retrieval: exhaustive cosine over numpy — no approximate index, so no ANN seed variance",
            "LLM API daily quota forced model rotation across runs; per-record model is in "
            "config_snapshot.model",
            "complexity threshold fixed by calibration before the main measurement (RUNBOOK STEP 3)",
        ],
    }
    (results_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"catalog written to {catalog_path}")
    print(f"wrote {tables_dir / 'table2_masking.csv'}")
    print(f"wrote {results_dir / 'reproducibility.json'}")
    print(f"masking: {report['numeric_masked']}/{report['columns_total']} cols, "
          f"{report['numeric_masked_pct']}% numeric masked, "
          f"{len(report['leaked_values'])} leaked, {report['payload_chars']} payload chars")


if __name__ == "__main__":
    main()
