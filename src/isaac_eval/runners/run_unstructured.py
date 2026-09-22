"""RUNBOOK STEP 5 — 비정형 문헌 질의 30건 실행.

이미 결과가 있는 qid는 건너뛴다 (쿼타 소진 시 이어달리기).
"""
from __future__ import annotations
import json
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.llm import LLMClient
from isaac_eval.retrieval import load_index
from isaac_eval import paper_agent
from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION


def main():
    load_env()
    cfg = load_settings()
    if get(cfg, "complexity.threshold") is None:
        raise SystemExit("complexity.threshold is null — run calibrate_threshold first (RUNBOOK STEP 3).")

    llm = LLMClient(cfg)
    chunk_index = load_index("data/corpus/chunk_index")
    paper_index = load_index("data/corpus/paper_index")

    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with open("queries/unstructured.jsonl", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    exhausted: set[str] = set()
    for q in queries:
        out_path = runs_dir / f"unstructured_{q['qid']}.json"
        if out_path.exists():
            print(f"{q['qid']}: skip (already have result)")
            continue

        result = None
        used_model = None
        for model in DEFAULT_MODEL_ROTATION:
            if model in exhausted:
                continue
            cfg["llm"]["model"] = model
            try:
                result = paper_agent.answer(q["text"], cfg, llm, chunk_index, paper_index)
                used_model = model
                break
            except genai_errors.ClientError as e:
                if e.code in (429, 404):
                    print(f"  ({model} unavailable: {e.code}, trying next)")
                    exhausted.add(model)
                    continue
                raise

        if result is None:
            print(f"{q['qid']}: SKIPPED — all candidate models exhausted")
            continue

        record = {
            "qid": q["qid"],
            "system": "proposed",
            "kind": "unstructured",
            "query": q["text"],
            "gold_route": q["gold_route"],
            "gold_paper_route": q["gold_paper_route"],
            "complexity_label": q["complexity_label"],
            "paper_route": result["paper_route"],
            "complexity": result["complexity"],
            "top1_cos": result["top1_cos"],
            "degraded": result["degraded"],
            "degraded_reason": result.get("degraded_reason"),
            "scope_kw_multi": result["scope_kw_multi"],
            "scope_llm_multi": result["scope_llm_multi"],
            "selected_papers": result["selected_papers"],
            "sources": result["sources"],
            "final_response": result["response"],
            "config_snapshot": {
                "model": used_model,
                "temperature": cfg["llm"]["temperature"],
                "seed": cfg["llm"]["seed"],
                "complexity_version": cfg["complexity"]["version"],
                "threshold": cfg["complexity"]["threshold"],
            },
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        mark = "OK " if result["paper_route"] == q["gold_paper_route"] else "MISS"
        print(f"{q['qid']} [{used_model}] {mark} route={result['paper_route']} "
              f"gold={q['gold_paper_route']} complexity={result['complexity']['total']}")

    if exhausted:
        print(f"\nExhausted models: {sorted(exhausted)}")


if __name__ == "__main__":
    main()
