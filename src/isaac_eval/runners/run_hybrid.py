"""RUNBOOK STEP 6 — 복합 질의 10건 실행.

분해 → 데이터 경로(코드 생성·샌드박스) + 문헌 경로(검색) → 통합 응답.
NCR은 데이터 부분의 샌드박스 결과를 사실 집합으로 삼아 채점한다.
"""
from __future__ import annotations
import json
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.catalog import build_catalog_entry
from isaac_eval.llm import LLMClient
from isaac_eval.retrieval import load_index, search
from isaac_eval.router import decompose
from isaac_eval.data_agent import answer as data_answer
from isaac_eval.gold import GOLD_REGISTRY, load_dataset
from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION


def main():
    load_env()
    cfg = load_settings()
    llm = LLMClient(cfg)
    catalog = build_catalog_entry(cfg)
    chunk_index = load_index("data/corpus/chunk_index")
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
    embed_model = get(cfg, "embedding.model")
    top_k = get(cfg, "retrieval.top_k")

    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with open("queries/hybrid.jsonl", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    exhausted: set[str] = set()
    for q in queries:
        out_path = runs_dir / f"hybrid_{q['qid']}.json"
        if out_path.exists():
            print(f"{q['qid']}: skip (already have result)")
            continue

        gold_value = GOLD_REGISTRY[q["gold_ref"]](df, **q.get("gold_args", {}))

        record = None
        for model in DEFAULT_MODEL_ROTATION:
            if model in exhausted:
                continue
            cfg["llm"]["model"] = model
            try:
                parts = decompose(q["text"], llm)
                data_sub = parts["data_subquery"] or q["text"]
                lit_sub = parts["literature_subquery"] or q["text"]

                data_result = data_answer(data_sub, catalog, cfg, llm)

                hits = search(chunk_index, lit_sub, top_k, embed_model)
                lit_context = "\n\n".join(f"[{cid}]\n{pl['text']}" for cid, _, pl in hits)

                final = llm.generate(
                    system="너는 계산 결과와 문헌 근거를 통합해 답하는 연구 어시스턴트다. "
                           "계산 결과에 없는 수치를 지어내지 마라.",
                    user=f"질의: {q['text']}\n\n"
                         f"[데이터 계산 결과(JSON)]\n{data_result['sandbox_value']}\n\n"
                         f"[문헌 발췌]\n{lit_context}\n\n"
                         "계산 결과의 수치와 문헌 근거를 함께 제시해 답하라.",
                )
                record = {
                    "qid": q["qid"],
                    "system": "proposed",
                    "kind": "hybrid",
                    "query": q["text"],
                    "gold_route": q["gold_route"],
                    "decomposition": parts,
                    "expected_data_subquery": q["expected_data_subquery"],
                    "expected_literature_subquery": q["expected_literature_subquery"],
                    "generated_code": data_result["generated_code"],
                    "attempts_used": len(data_result["sandbox_results"]),
                    "final_ok": data_result["final_ok"],
                    "sandbox_value": data_result["sandbox_value"],
                    "gold_value": gold_value,
                    "literature_sources": [cid for cid, _, _ in hits],
                    "final_response": final,
                    "config_snapshot": {
                        "model": model,
                        "temperature": cfg["llm"]["temperature"],
                        "seed": cfg["llm"]["seed"],
                    },
                }
                break
            except genai_errors.ClientError as e:
                if e.code in (429, 404):
                    print(f"  ({model} unavailable: {e.code})")
                    exhausted.add(model)
                    continue
                raise

        if record is None:
            print(f"{q['qid']}: SKIPPED — all models exhausted")
            continue

        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"{record['qid']} [{record['config_snapshot']['model']}] ok={record['final_ok']} "
              f"sandbox={record['sandbox_value']} gold={gold_value}")


if __name__ == "__main__":
    main()
