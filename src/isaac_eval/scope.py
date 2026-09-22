"""질의 범위 판정 — 단일 논문 질의와 다중 논문 질의의 구분.

7장 측정에서 이 단계가 경로 선택의 병목으로 드러났다. 고복잡도 단일 논문 질의(경로 B)가
다중 논문 질의(경로 C)로 분류되는 오류가 지배적이었으며, 원인은 판정기의 성능이 아니라
두 범주의 조작적 경계에 있었다.

핵심 구분: '질의가 복잡한가'와 '여러 편의 논문이 필요한가'는 서로 다른 축이다.
초기 규칙은 이를 섞어, 단일 논문 안에서의 심층 종합을 요구하는 표현("종합하라",
"교차하여 분석")까지 다중 논문 신호로 취급했다.

개정된 정의:
  다중 논문 질의 = 질의가 **서로 다른 문헌 간의** 비교·대조·공통점 도출을 요구한다.
  단일 논문 질의 = 하나의 문헌 안에서 답을 찾는다. 그 안에서 여러 요인을 종합하도록
                   요구하더라도 단일 논문 질의다.

따라서 신호어는 '복수의 문헌을 명시적으로 가리키는 표현'으로 한정한다. 종합·정리·분석은
단일 문헌에서도 성립하므로 그 자체로는 신호가 아니다.
"""
from __future__ import annotations
import json
import re

# 복수의 문헌 대상을 명시하는 표현만 신호로 삼는다.
PLURAL_SOURCE_TERMS = [
    "여러 논문", "논문들", "문헌들", "연구들", "등록된 논문", "여러 연구", "여러 편",
    "코퍼스", "전반에서", "전반적으로", "각 논문", "논문 간", "연구 간",
    "across papers", "multiple papers", "these papers", "the papers", "corpus",
    "each paper", "between studies", "across studies",
]

# 비교·대조 요구는 비교 대상이 복수 문헌일 때만 신호가 된다. 따라서 단독으로 쓰지 않고
# PLURAL_SOURCE_TERMS와 함께 나타날 때만 가중한다.
COMPARISON_TERMS = ["비교", "대조", "공통", "차이", "동향", "compare", "contrast", "trend"]

# 단일 문헌 안에서도 성립하므로 신호에서 제외한다. 초기 규칙의 오분류 원인이었다.
NON_SIGNALS = ["종합하", "정리해", "분석하", "설명하", "논하", "평가하", "synthesize", "analyze"]

SCOPE_SYSTEM = (
    "너는 질의가 한 편의 논문으로 답할 수 있는지, 서로 다른 여러 편의 논문을 비교해야 "
    "답할 수 있는지 판정한다. JSON만 출력한다."
)

SCOPE_PROMPT = """\
질의: {query}

판정 기준:
- multi_paper=true: 질의가 **서로 다른 문헌들 사이의** 비교, 대조, 공통점·차이점 도출,
  또는 여러 연구에 걸친 동향 파악을 요구한다. 즉 답하려면 둘 이상의 논문을 나란히
  놓고 견주어야 한다.
- multi_paper=false: 하나의 논문 안에서 답을 찾을 수 있다. 그 논문 안의 여러 요인이나
  단계를 종합·분석하도록 요구하더라도, 비교 대상이 복수의 문헌이 아니라면 false다.

주의: "종합하라", "교차하여 분석하라", "단계별로 설명하라"는 표현은 한 편의 논문
안에서도 성립하므로 그 자체로는 multi_paper의 근거가 아니다.

{{"multi_paper": true|false}} 형식의 JSON만 출력하라."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def keyword_multi_paper_v1(query: str) -> bool:
    """초기 규칙 — 논문 7.3절이 보고한 오분류의 원인. 비교를 위해 보존한다."""
    legacy = ["여러 논문", "논문들", "코퍼스", "전반", "공통", "비교하", "대조하", "종합하",
              "등록된 논문", "across papers", "multiple papers", "compare", "corpus"]
    lowered = query.lower()
    return any(k.lower() in lowered for k in legacy)


def keyword_multi_paper_v2(query: str) -> bool:
    """개정 규칙 — 복수 문헌을 명시하는 표현만 신호로 삼는다."""
    lowered = query.lower()
    return any(term.lower() in lowered for term in PLURAL_SOURCE_TERMS)


def llm_multi_paper_v2(query: str, llm) -> bool:
    raw = llm.generate(system=SCOPE_SYSTEM, user=SCOPE_PROMPT.format(query=query))
    match = _JSON_RE.search(raw or "")
    if not match:
        return False
    try:
        return bool(json.loads(match.group(0)).get("multi_paper", False))
    except json.JSONDecodeError:
        return False
