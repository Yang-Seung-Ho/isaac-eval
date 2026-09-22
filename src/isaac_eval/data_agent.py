from __future__ import annotations
import re
from isaac_eval.prompt import build_data_prompt
from isaac_eval.sandbox import run_with_retry

_CODE_FENCE_RE = re.compile(r"^```(?:python)?\n|\n```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text.strip())


def answer(query: str, catalog: dict, cfg: dict, llm) -> dict:
    base_prompt = build_data_prompt(catalog, query)
    generated_codes: list[str] = []

    def generator(prev_error: str | None) -> str:
        if prev_error is None:
            prompt = base_prompt
        else:
            prompt = (
                f"{base_prompt}\n\n[이전 시도 오류]\n{prev_error}\n"
                "위 오류를 수정한 새 코드를 처음부터 다시 출력하라."
            )
        raw = llm.generate(system="너는 pandas 코드만 출력하는 데이터 분석 에이전트다.", user=prompt)
        code = _strip_code_fence(raw)
        generated_codes.append(code)
        return code

    dataset_path = cfg["paths"]["dataset"]
    sheet = cfg["paths"]["sheet"]
    results = run_with_retry(generator, dataset_path, sheet, cfg)
    final = results[-1]
    sandbox_value = final.value if final.ok else None
    if final.ok:
        response_prompt = (
            f"사용자 질문: {query}\n"
            f"계산된 결과(JSON): {sandbox_value}\n"
            "위 계산 결과에 등장하는 수치만 사용해서 한국어로 간결하게 답하라. "
            "결과에 없는 수치를 새로 만들어 내지 마라."
        )
        final_response = llm.generate(
            system="너는 계산 결과를 한국어로 요약하는 어시스턴트다.",
            user=response_prompt,
        )
    else:
        final_response = f"코드 실행에 {len(results)}회 시도 후 실패했습니다: {final.error}"
    return {
        "generated_code": generated_codes[-1] if generated_codes else None,
        "generated_codes": generated_codes,
        "sandbox_results": results,
        "final_ok": final.ok,
        "sandbox_value": sandbox_value,
        "final_response": final_response,
    }
