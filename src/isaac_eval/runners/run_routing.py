"""RUNBOOK STEP 7 (라우팅 파트) — 70건 전체에 대한 마스터 라우터 분류 정확도.

라우팅만 측정하므로 질의당 LLM 호출 1회다. 이미 분류된 qid는 건너뛴다.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.llm import LLMClient
from isaac_eval.router import classify
from isaac_eval import metrics
from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION


def load_all_queries(cfg) -> list[dict]:
    queries = []
    with open(get(cfg, "paths.queries_structured"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries.append({"qid": q["qid"], "text": q["text"], "gold_route": "data"})
    for path, default_route in (("queries/unstructured.jsonl", "literature"), ("queries/hybrid.jsonl", "hybrid")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    queries.append({"qid": q["qid"], "text": q["text"],
                                    "gold_route": q.get("gold_route", default_route)})
    return queries


def main():
    load_env()
    cfg = load_settings()
    llm = LLMClient(cfg)

    results_dir = Path(get(cfg, "paths.results_dir"))
    out_path = results_dir / "routing.json"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    queries = load_all_queries(cfg)

    exhausted: set[str] = set()
    for q in queries:
        if q["qid"] in existing:
            continue
        routed = None
        used_model = None
        for model in DEFAULT_MODEL_ROTATION:
            if model in exhausted:
                continue
            cfg["llm"]["model"] = model
            try:
                routed = classify(q["text"], llm)
                used_model = model
                break
            except genai_errors.ClientError as e:
                if e.code in (429, 404):
                    print(f"  ({model} unavailable: {e.code})")
                    exhausted.add(model)
                    continue
                raise
        if routed is None:
            print(f"{q['qid']}: SKIPPED — all models exhausted")
            break
        existing[q["qid"]] = {"routed_to": routed, "gold_route": q["gold_route"], "model": used_model}
        out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        mark = "OK " if routed == q["gold_route"] else "MISS"
        print(f"{q['qid']} [{used_model}] {mark} routed={routed} gold={q['gold_route']}")

    records = [{"qid": k, **v} for k, v in existing.items()]
    accuracy = metrics.routing_accuracy(records)
    cm = metrics.confusion_matrix(records, "routed_to", "gold_route")

    with open(tables_dir / "table_routing.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "n", "routing_accuracy"])
        writer.writerow(["ALL", len(records), accuracy])
        for model in sorted({r["model"] for r in records}):
            subset = [r for r in records if r["model"] == model]
            writer.writerow([model, len(subset), metrics.routing_accuracy(subset)])

    labels = ["data", "literature", "hybrid", "chat"]
    with open(tables_dir / "confusion_routing.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + labels)
        for gold in labels:
            writer.writerow([gold] + [cm.get(gold, {}).get(pred, 0) for pred in labels])

    print(f"\nrouting accuracy (n={len(records)}): {accuracy}")
    print("confusion (gold -> pred):", dict(cm))


if __name__ == "__main__":
    main()
