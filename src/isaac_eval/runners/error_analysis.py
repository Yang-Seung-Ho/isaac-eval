"""RUNBOOK STEP 4 부속 — 5.6절 오류 유형별 발생 빈도 집계.

유형 1 (환각): 응답에 샌드박스 결과·질의에서 유래하지 않은 수치가 등장 (NCR 실패).
유형 2 (계산 오류): 샌드박스가 반환한 값이 gold와 불일치 (AA 실패).
두 유형은 독립이며, 한 질의가 양쪽에 모두 해당할 수 있다.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_settings, get
from isaac_eval.metrics import ncr_detail
from isaac_eval.numeric import flatten_facts, compare


def main():
    cfg = load_settings()
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    policy = get(cfg, "numeric.match_policy")
    rows = []
    counts = {"type1_hallucination": 0, "type2_calculation": 0, "exec_failure": 0, "clean": 0}

    for path in sorted(runs_dir.glob("proposed_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if not r["final_ok"]:
            counts["exec_failure"] += 1
            rows.append([r["qid"], r["config_snapshot"]["model"], "exec_failure", "", r["query"]])
            continue

        ncr_res = ncr_detail(r, policy=policy)
        sandbox_facts = flatten_facts(r["sandbox_value"])
        gold_facts = flatten_facts(r["gold_value"])
        aa_ok = (
            compare(sandbox_facts, gold_facts, policy=policy).pass_
            and compare(gold_facts, sandbox_facts, policy=policy).pass_
        )

        types = []
        if not ncr_res.pass_:
            types.append("type1_hallucination")
            counts["type1_hallucination"] += 1
        if not aa_ok:
            types.append("type2_calculation")
            counts["type2_calculation"] += 1
        if not types:
            counts["clean"] += 1
            continue

        rows.append([
            r["qid"],
            r["config_snapshot"]["model"],
            "+".join(types),
            " ".join(str(u) for u in ncr_res.unmatched),
            r["query"],
        ])

    with open(tables_dir / "error_analysis.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "model", "error_type", "unmatched_numbers", "query"])
        writer.writerows(rows)

    with open(tables_dir / "error_counts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "count", "policy"])
        for key, value in counts.items():
            writer.writerow([key, value, policy])

    print(f"policy={policy}")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"\nwrote {tables_dir / 'error_analysis.csv'} ({len(rows)} failing queries)")
    print(f"wrote {tables_dir / 'error_counts.csv'}")


if __name__ == "__main__":
    main()
