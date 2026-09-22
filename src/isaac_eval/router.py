"""마스터 라우터 — 질의를 data / literature / hybrid / chat 으로 분류한다.

few-shot 예시는 3개로 고정하며 질의별로 바꾸지 않는다 (스펙 §5.8).
"""
from __future__ import annotations
import json
import re

ROUTER_SYSTEM = "너는 연구 질의를 처리 경로로 분류하는 라우터다. 지정된 JSON만 출력한다."

FEW_SHOT = """\
분류 기준:
- data: 등록된 정형 데이터셋(수치 표)에 대한 계산·집계·조회로 답할 수 있는 질의
- literature: 등록된 논문 본문의 내용으로 답해야 하는 질의
- hybrid: 데이터 계산과 논문 해석이 모두 필요한 질의
- chat: 위 어디에도 해당하지 않는 일반 대화

예시 1
질의: 4월 전체 정점의 개체 수 합계는 얼마인가?
{"route": "data"}

예시 2
질의: What is Heterosigma akashiwo and why is it studied?
{"route": "literature"}

예시 3
질의: 정점 J9의 평균 개체 수를 계산하고, 그 수준이 문헌의 대발생 기준과 어떻게 비교되는지 논하라.
{"route": "hybrid"}
"""

DECOMPOSE_SYSTEM = "너는 복합 질의를 데이터 부분과 문헌 부분으로 분해한다. 지정된 JSON만 출력한다."

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    match = _JSON_RE.search(text or "")
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def classify(query: str, llm) -> str:
    raw = llm.generate(
        system=ROUTER_SYSTEM,
        user=f"{FEW_SHOT}\n이제 아래 질의를 분류하라.\n질의: {query}\n"
             '{"route": "data"|"literature"|"hybrid"|"chat"} 형식의 JSON만 출력하라.',
    )
    route = _parse_json(raw).get("route", "")
    return route if route in {"data", "literature", "hybrid", "chat"} else "chat"


def decompose(query: str, llm) -> dict:
    raw = llm.generate(
        system=DECOMPOSE_SYSTEM,
        user=f"복합 질의: {query}\n"
             '이 질의를 데이터 계산 부분과 문헌 조회 부분으로 나눠라.\n'
             '{"data_subquery": "...", "literature_subquery": "..."} 형식의 JSON만 출력하라.',
    )
    parsed = _parse_json(raw)
    return {
        "data_subquery": parsed.get("data_subquery", ""),
        "literature_subquery": parsed.get("literature_subquery", ""),
    }
