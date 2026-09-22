"""범위 판정 규칙 개정의 효과를 dev/test 분할로 평가한다.

규칙 개정은 관측된 오분류를 보고 이뤄졌으므로 과적합 위험이 있다. 캘리브레이션과 동일한
층화 분할(개발 20 / 검증 10, seed 고정)을 적용하여, 개정 규칙이 검증 집합에서도 유지되는지
확인한다. 검증 집합은 규칙 설계에 사용하지 않았다.

LLM 재판정(--with-llm)은 질의당 1회 호출한다. 호출 없이 실행하면 저장된 판정을 재사용한다.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.calibration import stratified_split
from isaac_eval.metrics import wilson_interval
from isaac_eval import scope


def route_of(record: dict, multi: bool, threshold: float) -> str:
    if multi:
        return "C"
    return "B" if record["complexity"]["total"] >= threshold else "A"


def score(records: list[dict], multi_fn, threshold: float) -> dict:
    hits = 0
    confusion: dict[str, dict[str, int]] = {}
    for r in records:
        route = route_of(r, multi_fn(r), threshold)
        gold = r["gold_paper_route"]
        confusion.setdefault(gold, {}).setdefault(route, 0)
        confusion[gold][route] += 1
        if route == gold:
            hits += 1
    lo, hi = wilson_interval(hits, len(records))
    return {"n": len(records), "hits": hits,
            "accuracy": round(hits / len(records), 4) if records else 0.0,
            "ci95": [lo, hi], "confusion": confusion}


def main():
    with_llm = "--with-llm" in sys.argv

    load_env()
    cfg = load_settings()
    threshold = get(cfg, "complexity.threshold")
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"

    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(runs_dir.glob("unstructured_*.json"))]

    if with_llm:
        from isaac_eval.llm import LLMClient
        cfg["llm"]["model"] = "gemini-3.5-flash"
        llm = LLMClient(cfg)
        for r in records:
            if "scope_llm_multi_v2" in r:
                continue
            try:
                r["scope_llm_multi_v2"] = scope.llm_multi_paper_v2(r["query"], llm)
            except genai_errors.ClientError as e:
                if e.code in (429, 404):
                    print(f"  ({r['qid']}: quota exhausted, stopping LLM rejudge)")
                    break
                raise
            (runs_dir / f"unstructured_{r['qid']}.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"  {r['qid']}: v2 multi={r['scope_llm_multi_v2']} (gold={r['gold_paper_route']})")

    # 캘리브레이션과 동일한 층화 분할
    items = [{"qid": r["qid"], "complexity_label": r["complexity_label"]} for r in records]
    dev_ids, test_ids = stratified_split(items, "complexity_label", 20, get(cfg, "llm.seed"))
    dev_ids = {i["qid"] for i in dev_ids}
    dev = [r for r in records if r["qid"] in dev_ids]
    test = [r for r in records if r["qid"] not in dev_ids]

    variants = {
        "kw_v1 (초기 규칙)": lambda r: scope.keyword_multi_paper_v1(r["query"]),
        "kw_v2 (개정 규칙)": lambda r: scope.keyword_multi_paper_v2(r["query"]),
        "llm_v1 (초기 프롬프트)": lambda r: r.get("scope_llm_multi_single", r["scope_llm_multi"]),
    }
    # 개정 프롬프트는 전건이 재판정된 경우에만 보고한다. 일부만 재판정된 상태에서
    # 나머지를 이전 판정으로 메우면 두 프롬프트가 섞인 값이 되어 해석할 수 없다
    # (스펙 §11: 측정하지 않은 값은 표에 넣지 않는다).
    n_v2 = sum(1 for r in records if "scope_llm_multi_v2" in r)
    if n_v2 == len(records):
        variants["llm_v2 (개정 프롬프트)"] = lambda r: r["scope_llm_multi_v2"]
    elif n_v2:
        print(f"\n⚠️  개정 프롬프트 재판정 {n_v2}/{len(records)}건 — 미완료이므로 표에서 제외한다.")
        print("    쿼타 확보 후 'make eval-scope-llm'으로 이어서 실행할 것.")

    report = {}
    print(f"\n=== 범위 판정 규칙 비교 (임계값 {threshold}) ===")
    print(f"{'방식':<24} {'dev(20)':<26} {'test(10)':<26} {'전체(30)'}")
    for name, fn in variants.items():
        d, t, a = score(dev, fn, threshold), score(test, fn, threshold), score(records, fn, threshold)
        report[name] = {"dev": d, "test": t, "all": a}
        print(f"{name:<24} {d['accuracy']:<6} {str(d['ci95']):<19} "
              f"{t['accuracy']:<6} {str(t['ci95']):<19} {a['accuracy']} {a['ci95']}")

    (results_dir / "scope_rules.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "table_scope_rules.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope_rule", "dev_n", "dev_accuracy", "test_n", "test_accuracy",
                         "all_n", "all_accuracy", "all_ci95_low", "all_ci95_high"])
        for name, r in report.items():
            writer.writerow([name, r["dev"]["n"], r["dev"]["accuracy"],
                             r["test"]["n"], r["test"]["accuracy"],
                             r["all"]["n"], r["all"]["accuracy"],
                             r["all"]["ci95"][0], r["all"]["ci95"][1]])

    print("\n혼동 행렬 (전체 30건, gold → pred):")
    for name, r in report.items():
        print(f"  {name:<24} {r['all']['confusion']}")
    print(f"\nwrote {tables_dir / 'table_scope_rules.csv'}")


if __name__ == "__main__":
    main()
