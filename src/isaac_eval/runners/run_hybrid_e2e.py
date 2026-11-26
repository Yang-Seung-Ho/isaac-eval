"""R2-3 대응: 하이브리드 질의 10건을 '원 질의 → 최종 답'까지 전체 워크플로로 실행한다.

기존 run_hybrid.py는 문헌 발췌를 바로 최종 종합에 넣어 논문 에이전트의 경로 선택(A/B/C)을
거치지 않았다. 이 러너는 실제 서비스 순서를 그대로 따른다.

  1) 마스터 라우터 분류 (router.classify) — 오분류도 그대로 결과에 반영
  2) 하이브리드면 분해 (router.decompose)
  3) 데이터셋 선택 (등록 카탈로그 4종 중 임베딩 유사도 최댓값) — 틀리면 데이터 부분 실패로 처리
  4) 데이터 에이전트 (코드 생성 + 샌드박스)
  5) 논문 에이전트 (paper_agent.answer: 범위 판정 → 복잡도 → 경로 A/B/C)
  6) 경로별 최종 응답 구성 (III-C 서술 그대로: A=통합 보고, B=데이터 결과 + 추천 목록, C=병렬 제시)

단일 모델(ISAAC_MODEL)만 사용하고 자동 모델 전환은 하지 않는다. 429면 멈추고, 같은 명령을
다시 실행하면 이어서 진행한다(완료된 qid는 건너뜀).

실행:
  export ISAAC_MODEL=gemini-3.5-flash
  ./.venv/bin/python3 -m isaac_eval.runners.run_hybrid_e2e
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.llm import LLMClient
from isaac_eval.catalog import build_catalog_entry
from isaac_eval.retrieval import load_index, embed_texts
from isaac_eval.router import classify, decompose
from isaac_eval.data_agent import answer as data_answer
from isaac_eval import paper_agent
from isaac_eval.gold import GOLD_REGISTRY, load_dataset
from isaac_eval.runners.run_dataset_selection import DATASETS, _descriptor

PREFIX = "hybrid_e2e"
SYNTH_SYSTEM = ("너는 계산 결과와 문헌 근거를 통합해 답하는 연구 어시스턴트다. "
                "계산 결과와 제공된 문헌 응답에 없는 수치나 주장을 지어내지 마라.")


def compose(query: str, path: str, data_value, lit: dict, llm) -> str:
    if path == "A":
        return llm.generate(system=SYNTH_SYSTEM, user=(
            f"질의: {query}\n\n[데이터 계산 결과(JSON)]\n{data_value}\n\n"
            f"[문헌 요약(경로 A)]\n{lit['response']}\n\n"
            "계산 결과와 문헌 근거를 출처를 구분해 하나의 보고로 통합하라."))
    if path == "B":
        # 경로 B: 문헌 종합 생성 없이, 데이터 결과 서술 + 추천 목록을 덧붙인다.
        data_part = llm.generate(system=SYNTH_SYSTEM, user=(
            f"질의: {query}\n\n[데이터 계산 결과(JSON)]\n{data_value}\n\n"
            "질의의 데이터 부분에 대해서만 계산 결과로 답하라. 문헌 내용은 서술하지 마라."))
        return (data_part + "\n\n[문헌] 근거가 충분하지 않아 문헌 종합을 제공하지 않습니다. "
                "관련 등록 논문:\n" + "\n".join(f"- {r['paper_id']} (관련도 {r['score']})"
                                               for r in lit.get("recommendations", [])))
    # 경로 C: 데이터 결과와 문헌 리뷰를 나란히 제시
    data_part = llm.generate(system=SYNTH_SYSTEM, user=(
        f"질의: {query}\n\n[데이터 계산 결과(JSON)]\n{data_value}\n\n"
        "질의의 데이터 부분에 대해서만 계산 결과로 답하라."))
    return data_part + "\n\n[문헌 리뷰(경로 C)]\n" + lit["response"]


def main():
    if not os.environ.get("ISAAC_MODEL"):
        sys.exit("ISAAC_MODEL을 지정하세요 (예: export ISAAC_MODEL=gemini-3.5-flash)")
    load_env(); cfg = load_settings(); cfg["llm"]["model"] = os.environ["ISAAC_MODEL"]
    llm = LLMClient(cfg)
    catalog = build_catalog_entry(cfg)
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
    chunk_index = load_index("data/corpus/chunk_index")
    paper_index = load_index("data/corpus/paper_index")
    emb = cfg["embedding"]["model"]

    ds_names = list(DATASETS)
    ds_vecs = embed_texts([_descriptor(json.loads(Path(DATASETS[n][0]).read_text(encoding="utf-8")))
                           for n in ds_names], model=emb)

    runs = Path(get(cfg, "paths.results_dir")) / "runs"; runs.mkdir(parents=True, exist_ok=True)
    queries = [json.loads(l) for l in open("queries/hybrid.jsonl", encoding="utf-8") if l.strip()]
    for q in queries:
        out = runs / f"{PREFIX}_{q['qid']}.json"
        if out.exists():
            print(f"{q['qid']}: skip"); continue
        gold = GOLD_REGISTRY[q["gold_ref"]](df, **q.get("gold_args", {}))
        rec = {"qid": q["qid"], "system": "proposed_e2e", "query": q["text"], "gold_value": gold,
               "gold_route": q["gold_route"], "config_snapshot": {"model": cfg["llm"]["model"],
               "temperature": cfg["llm"]["temperature"]}}
        route = classify(q["text"], llm); rec["routed_to"] = route
        data_q, lit_q = q["text"], q["text"]
        if route == "hybrid":
            parts = decompose(q["text"], llm); rec["decomposition"] = parts
            data_q = parts["data_subquery"] or q["text"]; lit_q = parts["literature_subquery"] or q["text"]

        data_value, lit = None, None
        if route in ("hybrid", "data"):
            qv = embed_texts([data_q], model=emb)
            pred = ds_names[int(np.argmax(qv @ ds_vecs.T))]; rec["selected_dataset"] = pred
            if pred == "jeju":
                dres = data_answer(data_q, catalog, cfg, llm)
                rec.update({"generated_code": dres["generated_code"], "final_ok": dres["final_ok"],
                            "sandbox_value": dres["sandbox_value"]})
                data_value = dres["sandbox_value"]
            else:
                rec.update({"final_ok": False, "sandbox_value": None, "error": "wrong dataset selected"})
        if route in ("hybrid", "literature"):
            lit = paper_agent.answer(lit_q, cfg, llm, chunk_index, paper_index)
            rec.update({"paper_route": lit["paper_route"], "complexity": lit["complexity"],
                        "literature_response": lit["response"],
                        "literature_sources": lit.get("sources", []),
                        "recommendations": lit.get("recommendations")})

        if route == "hybrid":
            final = compose(q["text"], lit["paper_route"], data_value, lit, llm)
        elif route == "data":
            final = llm.generate(system=SYNTH_SYSTEM, user=f"질의: {q['text']}\n\n[데이터 계산 결과(JSON)]\n{data_value}\n\n계산 결과로 답하라.")
        elif route == "literature":
            final = lit["response"]
        else:
            final = "(chat: 에이전트 미호출)"
        rec["final_response"] = final
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"{q['qid']} route={route} ds={rec.get('selected_dataset')} path={rec.get('paper_route')} ok={rec.get('final_ok')}")
    print("[usage]", llm.usage)


if __name__ == "__main__":
    main()
