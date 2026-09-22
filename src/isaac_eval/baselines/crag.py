"""CRAG 기준선 — 검색 품질 평가기 → {correct, ambiguous, incorrect} → 재검색/분해 전환."""
from __future__ import annotations
import json
import re

from isaac_eval.retrieval import search
from isaac_eval.baselines.standard_rag import build_data_context

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    match = _JSON_RE.search(text or "")
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _retrieve(query: str, cfg, chunk_index, top_k: int):
    if chunk_index is None:
        return [], ""
    hits = search(chunk_index, query, top_k, cfg["embedding"]["model"])
    context = "\n\n".join(f"[{cid}]\n{pl['text']}" for cid, _, pl in hits)
    return [cid for cid, _, _ in hits], context


def answer(query: str, cfg, llm, chunk_index=None, df=None) -> dict:
    top_k = cfg["retrieval"]["top_k"]
    sources, lit_context = _retrieve(query, cfg, chunk_index, top_k)

    grade_raw = llm.generate(
        system="너는 검색 결과가 질의에 답하기에 충분한지 평가한다. JSON만 출력한다.",
        user=f"질의: {query}\n\n[검색 결과]\n{lit_context or '(없음)'}\n\n"
             '{"grade": "correct"|"ambiguous"|"incorrect"} 형식의 JSON만 출력하라.',
    )
    grade = _parse_json(grade_raw).get("grade", "ambiguous")
    if grade not in {"correct", "ambiguous", "incorrect"}:
        grade = "ambiguous"

    action = "use_as_is"
    if grade in {"ambiguous", "incorrect"}:
        # 질의를 재작성해 한 번 더 검색한다 (CRAG의 corrective 단계).
        rewritten = llm.generate(
            system="너는 검색이 잘 되도록 질의를 다시 쓴다. 재작성된 질의 한 줄만 출력한다.",
            user=f"원 질의: {query}\n검색이 불충분했다. 핵심 용어 중심으로 재작성하라.",
        ).strip()
        extra_sources, extra_context = _retrieve(rewritten, cfg, chunk_index, top_k * 2)
        sources = list(dict.fromkeys(sources + extra_sources))
        lit_context = f"{lit_context}\n\n[재검색]\n{extra_context}"
        action = "re_retrieve"

    parts = []
    if df is not None:
        parts.append(f"[데이터셋 미리보기]\n{build_data_context(df)}")
    if lit_context.strip():
        parts.append(f"[문헌 발췌]\n{lit_context}")
    context = "\n\n".join(parts) if parts else "(컨텍스트 없음)"

    response = llm.generate(
        system="너는 평가된 컨텍스트를 근거로 답하는 어시스턴트다. 컨텍스트에 없는 수치를 지어내지 마라.",
        user=f"[컨텍스트]\n{context}\n\n[질문]\n{query}\n\n답하라.",
    )
    return {"system": "crag", "final_response": response, "sources": sources,
            "retrieval_grade": grade, "corrective_action": action,
            "sandbox_value": None, "final_ok": True, "attempts_used": 0}
