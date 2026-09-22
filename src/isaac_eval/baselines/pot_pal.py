"""PoT/PAL 스타일 기준선 = 어블레이션 '마스킹 OFF + 샌드박스 ON'.

표준 RAG(마스킹 없음·샌드박스 없음)와 제안(마스킹 ON·샌드박스 ON) 사이의 빠진 칸.
마스킹 없이 실제 관측 수치가 포함된 데이터 미리보기를 주고, LLM이 pandas 코드를
생성해 샌드박스에서 전수 실행한다(PoT[3]/PAL[10] 계열). 제안과의 유일한 차이는
'수치 마스킹 유무'이므로, 이 조건의 성능은 마스킹의 고유 기여를 분리해 준다.

동일 LLM·임베딩·코퍼스·샌드박스를 제안과 공유한다. 응답 정확도(AA)가 제안과 대등하면
마스킹은 정확도 요인이 아니라 프라이버시·프롬프트 크기 절감 요인으로 재포지셔닝된다.
"""
from __future__ import annotations
import re
from isaac_eval.prompt import CODE_CONTRACT
from isaac_eval.sandbox import run_with_retry

_CODE_FENCE_RE = re.compile(r"^```(?:python)?\n|\n```$", re.MULTILINE)


def _strip(t: str) -> str:
    return _CODE_FENCE_RE.sub("", (t or "").strip())


def build_unmasked_context(df, max_rows: int = 15) -> str:
    # 마스킹과 정반대: 실제 수치가 그대로 보이는 미리보기(열 의미 + 상위 행).
    cols = ", ".join(str(c) for c in df.columns)
    try:
        preview = df.head(max_rows).to_markdown(index=False)
    except Exception:
        preview = df.head(max_rows).to_string(index=False)
    return f"[열 목록]\n{cols}\n\n[데이터 미리보기 — 실제 관측 수치 포함, 상위 {max_rows}행]\n{preview}"


def answer(query: str, cfg, llm, chunk_index=None, df=None) -> dict:
    context = build_unmasked_context(df) if df is not None else ""
    base_prompt = (
        f"{context}\n\n[사용자 질의]\n{query}\n\n"
        "위 데이터셋 전체(미리보기는 일부일 뿐, 전체 행이 df로 로드되어 있음)에 대해 "
        "질의에 답하는 pandas 코드를 작성하라.\n\n" + CODE_CONTRACT
    )
    generated: list[str] = []

    def generator(prev_error):
        prompt = base_prompt if prev_error is None else (
            f"{base_prompt}\n\n[이전 시도 오류]\n{prev_error}\n위 오류를 수정해 코드를 처음부터 다시 출력하라.")
        code = _strip(llm.generate(system="너는 pandas 코드만 출력하는 데이터 분석 에이전트다.", user=prompt))
        generated.append(code)
        return code

    results = run_with_retry(generator, cfg["paths"]["dataset"], cfg["paths"]["sheet"], cfg)
    final = results[-1]
    sandbox_value = final.value if final.ok else None
    if final.ok:
        resp = llm.generate(
            system="너는 계산 결과를 한국어로 요약하는 어시스턴트다.",
            user=(f"사용자 질문: {query}\n계산된 결과(JSON): {sandbox_value}\n"
                  "위 계산 결과의 수치만 사용해 간결히 답하라. 결과에 없는 수치를 지어내지 마라."))
    else:
        resp = f"코드 실행에 {len(results)}회 시도 후 실패: {final.error}"
    return {
        "system": "pot_pal",
        "final_response": resp,
        "sources": [],
        "sandbox_value": sandbox_value,
        "final_ok": final.ok,
        "attempts_used": len(results),
        "generated_code": generated[-1] if generated else None,
    }
