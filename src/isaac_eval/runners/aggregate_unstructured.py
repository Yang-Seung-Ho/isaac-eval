"""비정형 실행 결과 집계 — 경로 선택 정확도, 우회 전환율, 혼동 행렬.

모델별로도 쪼개어 보고한다. 쿼타 소진으로 모델을 갈아탄 구간이 있으면
지표가 오염되므로, 그 사실이 표에 드러나야 한다.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

from isaac_eval.config import load_settings, get
from isaac_eval import metrics


def main():
    cfg = load_settings()
    threshold = get(cfg, "complexity.threshold")
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(runs_dir.glob("unstructured_*.json"))]
    if not records:
        raise SystemExit("no unstructured run records found")

    # 스펙 §5.9는 질의 범위 판단을 'LLM 판정 + 키워드 규칙 병행, 둘 다 로그'로 규정한다.
    # 두 판정 방식이 경로 선택에 미치는 영향이 크므로 각각 산출해 함께 보고한다.
    def rejudge(record: dict, mode: str) -> str:
        multi = (record["scope_kw_multi"] or record["scope_llm_multi"]) if mode == "llm_or_kw" \
            else record["scope_kw_multi"]
        if multi:
            return "C"
        return "B" if record["complexity"]["total"] >= threshold else "A"

    scope_modes = {}
    for mode in ("llm_or_kw", "kw_only"):
        judged = [dict(r, paper_route=rejudge(r, mode)) for r in records]
        high_single = [r for r in judged if r["complexity_label"] == "high" and r["gold_paper_route"] != "C"]
        scope_modes[mode] = {
            "route_selection_accuracy": metrics.route_selection_accuracy(judged),
            "graceful_degradation_rate": round(
                sum(1 for r in high_single if r["paper_route"] == "B") / len(high_single), 4
            ) if high_single else 0.0,
            "confusion": metrics.confusion_matrix(judged, "paper_route", "gold_paper_route"),
        }

    overall = {
        "n": len(records),
        "route_selection_accuracy": metrics.route_selection_accuracy(records),
        "graceful_degradation_rate": scope_modes["llm_or_kw"]["graceful_degradation_rate"],
        "complexity_classification_accuracy": metrics.complexity_classification_accuracy(records, threshold),
    }

    by_model: dict[str, dict] = {}
    for model in sorted({r["config_snapshot"]["model"] for r in records}):
        subset = [r for r in records if r["config_snapshot"]["model"] == model]
        by_model[model] = {
            "n": len(subset),
            "route_selection_accuracy": metrics.route_selection_accuracy(subset),
            "complexity_classification_accuracy": metrics.complexity_classification_accuracy(subset, threshold),
        }

    cm = metrics.confusion_matrix(records, "paper_route", "gold_paper_route")

    with open(tables_dir / "table_unstructured_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "n", "route_selection_accuracy", "complexity_classification_accuracy",
                         "graceful_degradation_rate"])
        writer.writerow(["ALL", overall["n"], overall["route_selection_accuracy"],
                         overall["complexity_classification_accuracy"], overall["graceful_degradation_rate"]])
        for model, stats in by_model.items():
            writer.writerow([model, stats["n"], stats["route_selection_accuracy"],
                             stats["complexity_classification_accuracy"], ""])
        for mode, stats in scope_modes.items():
            writer.writerow([f"scope={mode}", len(records), stats["route_selection_accuracy"],
                             overall["complexity_classification_accuracy"], stats["graceful_degradation_rate"]])

    routes = ["A", "B", "C"]
    with open(tables_dir / "confusion_paper_route.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + routes)
        for gold in routes:
            writer.writerow([gold] + [cm.get(gold, {}).get(pred, 0) for pred in routes])

    with open(tables_dir / "table_unstructured_per_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "model", "complexity_label", "complexity_total", "threshold",
                         "gold_paper_route", "paper_route", "correct", "scope_kw_multi",
                         "scope_llm_multi", "degraded", "top1_cos"])
        for r in records:
            writer.writerow([
                r["qid"], r["config_snapshot"]["model"], r["complexity_label"],
                r["complexity"]["total"], threshold, r["gold_paper_route"], r["paper_route"],
                r["paper_route"] == r["gold_paper_route"], r["scope_kw_multi"],
                r["scope_llm_multi"], r["degraded"], r["top1_cos"],
            ])

    print("=== unstructured summary ===")
    print(f"n = {overall['n']}")
    print(f"route selection accuracy      : {overall['route_selection_accuracy']}")
    print(f"complexity classification acc : {overall['complexity_classification_accuracy']} (threshold={threshold})")
    print(f"graceful degradation rate     : {overall['graceful_degradation_rate']}")
    print("\nby model:")
    for model, stats in by_model.items():
        print(f"  {model:<28} n={stats['n']:<3} route_acc={stats['route_selection_accuracy']:<8} "
              f"complexity_acc={stats['complexity_classification_accuracy']}")
    print("\nscope judgement (LLM said multi-paper):")
    for model in by_model:
        subset = [r for r in records if r["config_snapshot"]["model"] == model]
        says_multi = sum(1 for r in subset if r["scope_llm_multi"])
        truly_multi = sum(1 for r in subset if r["gold_paper_route"] == "C")
        print(f"  {model:<28} llm_multi={says_multi}/{len(subset)}  actually_C={truly_multi}/{len(subset)}")
    print("\nby scope-judgement mode:")
    for mode, stats in scope_modes.items():
        print(f"  {mode:<12} route_acc={stats['route_selection_accuracy']:<8} "
              f"degradation={stats['graceful_degradation_rate']}")
        print(f"               confusion={dict(stats['confusion'])}")
    print("\nconfusion (gold -> pred):", dict(cm))
    print(f"\nwrote {tables_dir / 'table_unstructured_metrics.csv'}")


if __name__ == "__main__":
    main()
