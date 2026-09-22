from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.catalog import build_catalog_entry
from isaac_eval.llm import LLMClient
from isaac_eval.data_agent import answer
from isaac_eval.gold import GOLD_REGISTRY, load_dataset
from isaac_eval import metrics


def _sandbox_result_to_dict(r):
    return {"attempt": r.attempt, "ok": r.ok, "error": r.error, "elapsed_ms": r.elapsed_ms}


def main():
    load_env()
    cfg = load_settings()
    catalog = build_catalog_entry(cfg)
    llm = LLMClient(cfg)
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])

    queries_path = get(cfg, "paths.queries_structured")
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(queries_path, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    records = []
    for q in queries:
        gold_fn = GOLD_REGISTRY[q["gold_ref"]]
        gold_value = gold_fn(df, **q.get("gold_args", {}))

        result = answer(q["text"], catalog, cfg, llm)
        record = {
            "qid": q["qid"],
            "query": q["text"],
            "generated_code": result["generated_code"],
            "attempts": [_sandbox_result_to_dict(r) for r in result["sandbox_results"]],
            "attempts_used": len(result["sandbox_results"]),
            "final_ok": result["final_ok"],
            "sandbox_value": result["sandbox_value"],
            "final_response": result["final_response"],
            "gold_value": gold_value,
            "config_snapshot": {
                "model": cfg["llm"]["model"],
                "temperature": cfg["llm"]["temperature"],
                "seed": cfg["llm"]["seed"],
            },
        }
        records.append(record)
        (runs_dir / f"proposed_{q['qid']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"{q['qid']}: final_ok={record['final_ok']} sandbox={record['sandbox_value']} gold={gold_value}")

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

    print("\n=== summary ===")
    print(f"NCR (exact): {ncr_exact}")
    print(f"NCR (round): {ncr_round}")
    print(f"Answer Accuracy: {aa}")
    print(f"Exec success: {exec_stats}")


if __name__ == "__main__":
    main()
