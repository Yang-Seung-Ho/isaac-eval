"""RUNBOOK STEP 7 — 기준선 3종을 정형 30건에 실행.

정형 질의를 쓰는 이유: 마스킹·샌드박스의 대조 효과(수치 환각)가 여기서 드러난다.
동일 LLM·동일 임베딩·동일 코퍼스를 쓰고, 기준선만 코드 생성·샌드박스를 쓰지 않는다.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.llm import LLMClient
from isaac_eval.retrieval import load_index
from isaac_eval.gold import GOLD_REGISTRY, load_dataset
from isaac_eval.baselines import standard_rag, self_rag, crag
from isaac_eval.baselines.standard_rag import build_data_context
from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION

SYSTEMS = {"standard_rag": standard_rag, "self_rag": self_rag, "crag": crag}


def main():
    names = sys.argv[1:] or list(SYSTEMS)
    _ROT = [os.environ["ISAAC_MODEL"]] if os.environ.get("ISAAC_MODEL") else DEFAULT_MODEL_ROTATION

    load_env()
    cfg = load_settings()
    llm = LLMClient(cfg)
    chunk_index = load_index("data/corpus/chunk_index")
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
    context_text = build_data_context(df)

    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with open(get(cfg, "paths.queries_structured"), encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    exhausted: set[str] = set()
    for name in names:
        module = SYSTEMS[name]
        for q in queries:
            out_path = runs_dir / f"{name}_{q['qid']}.json"
            if out_path.exists():
                continue

            gold_value = GOLD_REGISTRY[q["gold_ref"]](df, **q.get("gold_args", {}))

            result = None
            used_model = None
            for model in _ROT:
                if model in exhausted:
                    continue
                cfg["llm"]["model"] = model
                try:
                    result = module.answer(q["text"], cfg, llm, chunk_index=chunk_index, df=df)
                    used_model = model
                    break
                except genai_errors.ClientError as e:
                    if e.code in (429, 404):
                        print(f"  ({model} unavailable: {e.code})")
                        exhausted.add(model)
                        continue
                    raise

            if result is None:
                print(f"{name}/{q['qid']}: SKIPPED — all models exhausted")
                return

            record = {
                "qid": q["qid"],
                "system": name,
                "kind": "structured",
                "query": q["text"],
                "final_response": result["final_response"],
                "sandbox_value": None,
                "final_ok": True,
                "attempts_used": 0,
                "gold_value": gold_value,
                "context_text": context_text,
                "sources": result.get("sources", []),
                "extra": {k: v for k, v in result.items()
                          if k not in {"final_response", "sources", "sandbox_value", "final_ok", "attempts_used", "system"}},
                "config_snapshot": {"model": used_model, "temperature": cfg["llm"]["temperature"]},
            }
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"{name}/{q['qid']} [{used_model}] done")


if __name__ == "__main__":
    main()
