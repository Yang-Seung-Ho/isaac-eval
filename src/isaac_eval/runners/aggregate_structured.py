"""results/runs/proposed_*.json 파일들을 모아 NCR/AA/실행성공률 표를 산출한다."""
from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_settings, get
from isaac_eval import metrics


def main():
    cfg = load_settings()
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(runs_dir.glob("proposed_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))

    ncr_exact = metrics.ncr(records, policy="exact")
    ncr_round = metrics.ncr(records, policy="round")
    aa = metrics.answer_accuracy(records, policy="round")
    exec_stats = metrics.code_exec_success(records)

    with open(tables_dir / "table_structured_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_queries", len(records)])
        writer.writerow(["ncr_exact", ncr_exact])
        writer.writerow(["ncr_round", ncr_round])
        writer.writerow(["answer_accuracy", aa])
        writer.writerow(["exec_first_attempt", exec_stats["first_attempt"]])
        writer.writerow(["exec_with_retry", exec_stats["with_retry"]])
        writer.writerow(["exec_mean_attempts", exec_stats["mean_attempts"]])

    with open(tables_dir / "table_structured_per_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "model", "final_ok", "attempts_used", "sandbox_value", "gold_value"])
        for r in records:
            writer.writerow([
                r["qid"],
                r["config_snapshot"]["model"],
                r["final_ok"],
                r["attempts_used"],
                json.dumps(r["sandbox_value"], ensure_ascii=False),
                json.dumps(r["gold_value"], ensure_ascii=False),
            ])

    print("=== summary ===")
    print(f"n_queries: {len(records)}")
    print(f"NCR (exact): {ncr_exact}")
    print(f"NCR (round): {ncr_round}")
    print(f"Answer Accuracy: {aa}")
    print(f"Exec success: {exec_stats}")
    print(f"\nwrote {tables_dir / 'table_structured_metrics.csv'}")
    print(f"wrote {tables_dir / 'table_structured_per_query.csv'}")


if __name__ == "__main__":
    main()
