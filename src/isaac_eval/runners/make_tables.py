"""RUNBOOK STEP 9 — 논문 표 생성.

표 4는 제안 방식과 기준선을 6개 지표로 비교한다. 실행된 기준선만 행으로 만든다
(스펙 §11: 측정하지 않은 값은 표에 넣지 않는다).
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_settings, get
from isaac_eval import metrics
from isaac_eval.numeric import flatten_facts, compare, extract_numbers


def _load(runs_dir: Path, prefix: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob(f"{prefix}_*.json"))]


def _ci(successes: int, n: int) -> str:
    lo, hi = metrics.wilson_interval(successes, n)
    return f"[{lo:.3f}, {hi:.3f}]"


def _baseline_metrics(records: list[dict], policy: str) -> dict:
    """기준선은 샌드박스가 없으므로 컨텍스트 기반 NCR과 텍스트 기반 AA를 쓴다."""
    total = len(records)
    if total == 0:
        return {}
    ncr_pass = sum(1 for r in records if metrics.ncr_context_based(r, policy=policy).pass_)
    aa_rate = metrics.answer_accuracy_from_text(records, policy=policy)
    aa_pass = round(aa_rate * total)
    return {
        "n": total,
        "ncr": round(ncr_pass / total, 4),
        "ncr_ci": _ci(ncr_pass, total),
        "answer_accuracy": aa_rate,
        "aa_ci": _ci(aa_pass, total),
        "exec_first_attempt": "",
        "exec_with_retry": "",
        "mean_attempts": "",
    }


def _proposed_metrics(records: list[dict], policy: str) -> dict:
    exec_stats = metrics.code_exec_success(records)
    total = len(records)
    ncr_rate = metrics.ncr(records, policy=policy)
    aa_rate = metrics.answer_accuracy(records, policy=policy)
    return {
        "n": total,
        "ncr": ncr_rate,
        "ncr_ci": _ci(round(ncr_rate * total), total),
        "answer_accuracy": aa_rate,
        "aa_ci": _ci(round(aa_rate * total), total),
        "exec_first_attempt": exec_stats["first_attempt"],
        "exec_with_retry": exec_stats["with_retry"],
        "mean_attempts": exec_stats["mean_attempts"],
    }


def main():
    cfg = load_settings()
    policy = get(cfg, "numeric.match_policy")
    threshold = get(cfg, "complexity.threshold")
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    proposed = _load(runs_dir, "proposed")
    rows = []
    if proposed:
        rows.append(("proposed (masking + sandbox)", _proposed_metrics(proposed, policy)))
    for name, label in (("standard_rag", "Standard RAG (no masking)"),
                        ("self_rag", "Self-RAG"),
                        ("crag", "CRAG")):
        records = _load(runs_dir, name)
        if records:
            rows.append((label, _baseline_metrics(records, policy)))

    with open(tables_dir / "table4_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "n", "ncr", "ncr_ci95", "answer_accuracy", "aa_ci95",
                         "exec_first_attempt", "exec_with_retry", "mean_attempts"])
        for label, m in rows:
            writer.writerow([label, m["n"], m["ncr"], m["ncr_ci"], m["answer_accuracy"], m["aa_ci"],
                             m["exec_first_attempt"], m["exec_with_retry"], m["mean_attempts"]])

    # 하이브리드
    hybrid = _load(runs_dir, "hybrid")
    if hybrid:
        # 하이브리드 응답은 계산 결과와 함께 문헌 발췌를 인용한다. 인용된 발췌의
        # 수치는 지어낸 값이 아니므로 NCR 허용 집합에 포함한다.
        chunk_text = {}
        chunks_path = Path("data/corpus/chunks.jsonl")
        if chunks_path.exists():
            for line in chunks_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c = json.loads(line)
                    chunk_text[c["chunk_id"]] = c["text"]
        for r in hybrid:
            r["context_text"] = " ".join(chunk_text.get(cid, "") for cid in r.get("literature_sources", []))
        hy_ncr = sum(
            1 for r in hybrid
            if compare(
                extract_numbers(r["final_response"]),
                flatten_facts(r["sandbox_value"])
                + extract_numbers(r["query"])
                + extract_numbers(r["context_text"]),
                policy=policy,
            ).pass_
        )
        hy_aa = 0
        for r in hybrid:
            sf, gf = flatten_facts(r["sandbox_value"]), flatten_facts(r["gold_value"])
            if sf and gf and compare(sf, gf, policy=policy).pass_ and compare(gf, sf, policy=policy).pass_:
                hy_aa += 1
        decomposed = sum(1 for r in hybrid
                         if r["decomposition"]["data_subquery"] and r["decomposition"]["literature_subquery"])
        with open(tables_dir / "table_hybrid_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value", "n"])
            writer.writerow(["decomposition_completeness", round(decomposed / len(hybrid), 4), len(hybrid)])
            writer.writerow(["ncr", round(hy_ncr / len(hybrid), 4), len(hybrid)])
            writer.writerow(["answer_accuracy", round(hy_aa / len(hybrid), 4), len(hybrid)])
            writer.writerow(["exec_success", round(sum(1 for r in hybrid if r["final_ok"]) / len(hybrid), 4), len(hybrid)])

    # 충실성 평가 시트 (사람이 채움)
    faith_rows = []
    for r in _load(runs_dir, "unstructured"):
        faith_rows.append([r["qid"], "proposed", r["query"], r["final_response"],
                           " | ".join(r["sources"]), "", ""])
    for name in ("standard_rag", "self_rag", "crag"):
        for r in _load(runs_dir, name):
            faith_rows.append([r["qid"], name, r["query"], r["final_response"],
                               " | ".join(r.get("sources", [])), "", ""])
    if faith_rows:
        with open(results_dir / "faithfulness_sheet.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["qid", "system", "query", "response", "retrieved_chunks",
                             "rater_1_score", "rater_2_score"])
            writer.writerows(faith_rows)

    print("=== table 4 (proposed vs baselines) ===")
    for label, m in rows:
        print(f"  {label:<30} n={m['n']:<3} NCR={m['ncr']} {m['ncr_ci']}  AA={m['answer_accuracy']} {m['aa_ci']}")
    print(f"\nwrote {tables_dir / 'table4_metrics.csv'}")
    if hybrid:
        print(f"wrote {tables_dir / 'table_hybrid_metrics.csv'}")
    if faith_rows:
        print(f"wrote {results_dir / 'faithfulness_sheet.csv'} ({len(faith_rows)} rows for expert rating)")


if __name__ == "__main__":
    main()
