"""제안 − 마스킹 조건 (어블레이션용).

제안 데이터 에이전트와 동일하게 HITL 카탈로그(데이터 사전 + 구조 온톨로지)를 받되,
마스킹을 끄고 '실제 관측 수치가 포함된 미리보기'까지 함께 제공한다. 그 뒤 pandas
코드를 생성해 샌드박스에서 실행한다.

- 제안(proposed)과의 유일한 차이 = 마스킹(실제 수치 미리보기 유무)  → 마스킹의 고유 기여 분리
- 순수 PoT/PAL(pot_pal)과의 유일한 차이 = HITL 카탈로그(사전+온톨로지) 유무 → 카탈로그의 고유 기여 분리
동일 LLM·샌드박스·데이터셋을 공유한다.
"""
from __future__ import annotations
import re
import json
from isaac_eval.prompt import CODE_CONTRACT
from isaac_eval.sandbox import run_with_retry

_FENCE = re.compile(r"^```(?:python)?\n|\n```$", re.MULTILINE)


def _strip(t: str) -> str:
    return _FENCE.sub("", (t or "").strip())


def build_prompt(catalog: dict, df, query: str, max_rows: int = 15) -> str:
    dd = json.dumps(catalog["data_dictionary"], ensure_ascii=False, indent=2)
    on = json.dumps(catalog["structural_ontology"], ensure_ascii=False, indent=2)
    try:
        preview = df.head(max_rows).to_markdown(index=False)
    except Exception:
        preview = df.head(max_rows).to_string(index=False)
    return (
        f"[Data Dictionary]\n{dd}\n\n"
        f"[Structural Ontology]\n{on}\n\n"
        f"[데이터 미리보기 — 실제 관측 수치 포함, 상위 {max_rows}행]\n{preview}\n\n"
        f"[User Query]\n{query}\n\n{CODE_CONTRACT}"
    )


def answer(query: str, catalog: dict, cfg, llm, df=None) -> dict:
    base = build_prompt(catalog, df, query)
    gen: list[str] = []

    def generator(prev_error):
        prompt = base if prev_error is None else (
            f"{base}\n\n[이전 시도 오류]\n{prev_error}\n위 오류를 수정해 코드를 처음부터 다시 출력하라.")
        code = _strip(llm.generate(system="너는 pandas 코드만 출력하는 데이터 분석 에이전트다.", user=prompt))
        gen.append(code)
        return code

    results = run_with_retry(generator, cfg["paths"]["dataset"], cfg["paths"]["sheet"], cfg)
    final = results[-1]
    sv = final.value if final.ok else None
    if final.ok:
        resp = llm.generate(
            system="너는 계산 결과를 한국어로 요약하는 어시스턴트다.",
            user=(f"사용자 질문: {query}\n계산된 결과(JSON): {sv}\n"
                  "위 계산 결과의 수치만 사용해 간결히 답하라. 결과에 없는 수치를 지어내지 마라."))
    else:
        resp = f"코드 실행에 {len(results)}회 시도 후 실패: {final.error}"
    return {
        "system": "proposed_unmasked", "final_response": resp, "sources": [],
        "sandbox_value": sv, "final_ok": final.ok, "attempts_used": len(results),
        "generated_code": gen[-1] if gen else None,
    }
