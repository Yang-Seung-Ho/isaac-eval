"""단일 다중논문 질의를 경로 C로 태워 실제 응답을 캡처한다 (그림 12 재현용).

사용법 (Mac 터미널, isaac-eval 폴더에서):
    ./.venv/bin/python -m isaac_eval.runners.ask_one "질의문"
인자를 생략하면 기본 마린 다중논문 질의를 사용한다.
결과는 화면에 출력되고 results/runs/adhoc_pathC.json 으로도 저장된다.
LLM 호출은 임베딩 1 + 범위판정 1 + 종합 1 ≈ 2~3건(무료 쿼타 내).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.llm import LLMClient
from isaac_eval.retrieval import load_index
from isaac_eval import paper_agent
from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION

DEFAULT_QUERY = "등록된 해양 조류(algae) 논문들의 연구 목적·방법·결과를 비교해줘."


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    load_env()
    cfg = load_settings()
    if get(cfg, "complexity.threshold") is None:
        raise SystemExit("complexity.threshold is null — 먼저 `make calibrate` 를 실행하세요.")

    llm = LLMClient(cfg)
    chunk_index = load_index("data/corpus/chunk_index")
    paper_index = load_index("data/corpus/paper_index")

    result = None
    used_model = None
    for model in DEFAULT_MODEL_ROTATION:
        cfg["llm"]["model"] = model
        try:
            result = paper_agent.answer(query, cfg, llm, chunk_index, paper_index)
            used_model = model
            break
        except genai_errors.ClientError as e:
            if e.code in (429, 404):
                print(f"  ({model} 사용 불가: {e.code}, 다음 모델 시도)")
                continue
            raise

    if result is None:
        raise SystemExit("후보 모델 모두 소진(일일 쿼타?). 태평양시 자정 이후 다시 시도하세요.")

    print("=" * 64)
    print("질의       :", query)
    print("모델       :", used_model)
    print("kw_multi   :", result["scope_kw_multi"], "| llm_multi:", result["scope_llm_multi"])
    print("paper_route:", result["paper_route"], "  (C 여야 정상)")
    print("복잡도합계 :", result["complexity"]["total"], "| top1_cos:", result["top1_cos"])
    print("선택 논문  :", result["selected_papers"])
    print("출처 청크  :", result["sources"])
    print("-" * 64)
    print(result["response"])
    print("=" * 64)
    if result["paper_route"] != "C":
        print("\n[주의] 경로 C로 가지 않았습니다. 질의에 '논문들/연구들/코퍼스/비교' 같은")
        print("       복수 문헌 표현을 넣어 다시 시도하세요.")

    out = Path(get(cfg, "paths.results_dir")) / "runs" / "adhoc_pathC.json"
    out.write_text(
        json.dumps({"query": query, "model": used_model, **result},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("저장됨:", out)


if __name__ == "__main__":
    main()
