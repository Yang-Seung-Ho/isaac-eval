"""RUNBOOK STEP 3 — 복잡도 임계값 캘리브레이션 (v1·v2 동시).

산출:
  results/tables/calibration_report.csv   5.5절 (분포·임계값·F1·도달가능성)
  results/tables/table3_complexity.csv    표 3 (질의별 v1/v2 차원 점수)
  results/calibration.json                원자료
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.retrieval import load_index, search
from isaac_eval.complexity import score_v1, score_v2, score_v3
from isaac_eval.calibration import calibrate


def main():
    load_env()
    cfg = load_settings()
    embed_model = get(cfg, "embedding.model")
    top_k = get(cfg, "retrieval.top_k")

    chunk_index = load_index("data/corpus/chunk_index")

    with open("queries/unstructured.jsonl", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    rows = []
    for q in queries:
        hits = search(chunk_index, q["text"], top_k, embed_model)
        chunks = [payload["text"] for _, _, payload in hits]
        top1_cos = hits[0][1] if hits else 0.0

        v1 = score_v1(chunks, q["text"])
        v2 = score_v2(chunks, q["text"], top1_cos)
        v3 = score_v3(chunks, q["text"], top1_cos)
        rows.append({
            "qid": q["qid"],
            "label": q["complexity_label"],
            "gold_paper_route": q["gold_paper_route"],
            "top1_cos": round(top1_cos, 4),
            "retrieved": [cid for cid, _, _ in hits],
            "v1": v1.as_dict(),
            "v2": v2.as_dict(),
            "v3": v3.as_dict(),
        })
        print(f"{q['qid']} [{q['complexity_label']}] cos={top1_cos:.3f} v1={v1.total:.2f} v2={v2.total:.2f} v3={v3.total:.2f}")

    results_dir = Path(get(cfg, "paths.results_dir"))
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for version in ("v1", "v2", "v3"):
        scored = [{"qid": r["qid"], "label": r["label"], "score": r[version]["total"]} for r in rows]
        reports[version] = calibrate(scored, version=version, dev_size=20, seed=get(cfg, "llm.seed"))

    (results_dir / "calibration.json").write_text(
        json.dumps({"per_query": rows, "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(tables_dir / "calibration_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "version", "n_dev", "n_test",
            "low_min", "low_q1", "low_median", "low_q3", "low_max",
            "high_min", "high_q1", "high_median", "high_q3", "high_max",
            "selected_threshold", "dev_f1", "test_f1", "route_a_reachable", "max_low_score_all",
            "direction_correct", "separation_auc", "median_gap",
        ])
        for version, rep in reports.items():
            low, high = rep["dist_low"], rep["dist_high"]
            writer.writerow([
                version, rep["n_dev"], rep["n_test"],
                low["min"], low["q1"], low["median"], low["q3"], low["max"],
                high["min"], high["q1"], high["median"], high["q3"], high["max"],
                rep["selected_threshold"], rep["dev_f1"], rep["test_f1"],
                rep["route_a_reachable"], rep["max_low_score_all"],
                rep["direction_correct"], rep["separation_auc"], rep["median_gap"],
            ])

    with open(tables_dir / "table3_complexity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "qid", "complexity_label", "gold_paper_route", "top1_cos",
            "v1_dim1", "v1_dim2", "v1_dim3", "v1_total",
            "v2_dim1", "v2_dim2", "v2_dim3", "v2_total",
            "v3_dim1", "v3_dim2", "v3_dim3", "v3_total",
        ])
        for r in rows:
            writer.writerow([
                r["qid"], r["label"], r["gold_paper_route"], r["top1_cos"],
                r["v1"]["dim1"], r["v1"]["dim2"], r["v1"]["dim3"], r["v1"]["total"],
                r["v2"]["dim1"], r["v2"]["dim2"], r["v2"]["dim3"], r["v2"]["total"],
                r["v3"]["dim1"], r["v3"]["dim2"], r["v3"]["dim3"], r["v3"]["total"],
            ])

    print("\n=== calibration report ===")
    for version, rep in reports.items():
        print(f"\n[{version}]")
        print(f"  dist_low : {rep['dist_low']}")
        print(f"  dist_high: {rep['dist_high']}")
        print(f"  threshold: {rep['selected_threshold']}  dev_f1={rep['dev_f1']}  test_f1={rep['test_f1']}")
        print(f"  direction_correct: {rep['direction_correct']}  AUC={rep['separation_auc']}  median_gap={rep['median_gap']}")
        print(f"  route_a_reachable: {rep['route_a_reachable']}  (max low score overall = {rep['max_low_score_all']})")
        if not rep["direction_correct"]:
            print("  ⚠️  방향 역전: low가 high보다 높게 채점된다. 임계값을 올릴수록 low가 먼저 걸린다.")
        elif not rep["route_a_reachable"]:
            print("  ⚠️  부분 겹침: 방향은 맞으나 두 분포의 꼬리가 겹쳐 완전 분리는 되지 않는다.")

    print(f"\nwrote {tables_dir / 'calibration_report.csv'}")
    print(f"wrote {tables_dir / 'table3_complexity.csv'}")


if __name__ == "__main__":
    main()
