"""남은 정형 질의를 실행한다 (일일 쿼타 소진 대응, 여러 모델로 자동 전환).

results/runs/ 에 이미 존재하는 qid는 건너뛴다. 각 질의마다 후보 모델을 순서대로
시도하다가 429(쿼타 소진)/404(모델 미가용)를 만나면 다음 모델로 넘어간다.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.catalog import build_catalog_entry
from isaac_eval.llm import LLMClient
from isaac_eval.data_agent import answer
from isaac_eval.gold import GOLD_REGISTRY, load_dataset

DEFAULT_MODEL_ROTATION = [
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
]


def _sandbox_result_to_dict(r):
    return {"attempt": r.attempt, "ok": r.ok, "error": r.error, "elapsed_ms": r.elapsed_ms}


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODEL_ROTATION

    load_env()
    cfg = load_settings()
    catalog = build_catalog_entry(cfg)
    llm = LLMClient(cfg)
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])

    queries_path = get(cfg, "paths.queries_structured")
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with open(queries_path, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    exhausted: set[str] = set()

    for q in queries:
        out_path = runs_dir / f"proposed_{q['qid']}.json"
        if out_path.exists():
            print(f"{q['qid']}: skip (already have result)")
            continue

        gold_fn = GOLD_REGISTRY[q["gold_ref"]]
        gold_value = gold_fn(df, **q.get("gold_args", {}))

        result = None
        used_model = None
        for model in models:
            if model in exhausted:
                continue
            cfg["llm"]["model"] = model
            try:
                result = answer(q["text"], catalog, cfg, llm)
                used_model = model
                break
            except genai_errors.ClientError as e:
                if e.code in (429, 404):
                    print(f"  ({model} unavailable: {e.code}, trying next model)")
                    exhausted.add(model)
                    continue
                raise

        if result is None:
            print(f"{q['qid']}: SKIPPED — all candidate models exhausted/unavailable")
            continue

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
                "model": used_model,
                "temperature": cfg["llm"]["temperature"],
                "seed": cfg["llm"]["seed"],
            },
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"{q['qid']} [{used_model}]: final_ok={record['final_ok']} sandbox={record['sandbox_value']} gold={gold_value}")

    if exhausted:
        print(f"\nExhausted/unavailable models this run: {sorted(exhausted)}")


if __name__ == "__main__":
    main()
