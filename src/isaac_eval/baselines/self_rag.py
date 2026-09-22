"""Self-RAG 기준선 — 검색 필요 여부 자체 판단 → 검색 → 생성 → 자기 비평 후 1회 재생성."""
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


def answer(query: str, cfg, llm, chunk_index=None, df=None) -> dict:
    need_raw = llm.generate(
        system="너는 질의에 답하기 위해 외부 근거 검색이 필요한지 판정한다. JSON만 출력한다.",
        user=f"질의: {query}\n{{\"need_retrieval\": true|false}} 형식의 JSON만 출력하라.",
    )
    need_retrieval = bool(_parse_json(need_raw).get("need_retrieval", True))

    parts = []
    sources: list[str] = []
    if need_retrieval:
        if df is not None:
            parts.append(f"[데이터셋 미리보기]\n{build_data_context(df)}")
        if chunk_index is not None:
            hits = search(chunk_index, query, cfg["retrieval"]["top_k"], cfg["embedding"]["model"])
            sources = [cid for cid, _, _ in hits]
            parts.append("[문헌 발췌]\n" + "\n\n".join(f"[{cid}]\n{pl['text']}" for cid, _, pl in hits))
    context = "\n\n".join(parts) if parts else "(검색 없음)"

    draft = llm.generate(
        system="너는 주어진 컨텍스트를 근거로 답하는 어시스턴트다.",
        user=f"[컨텍스트]\n{context}\n\n[질문]\n{query}\n\n답하라.",
    )

    critique = llm.generate(
        system="너는 초안 응답을 비평한다. 근거가 없는 주장이나 컨텍스트에 없는 수치를 지적하라.",
        user=f"[질문]\n{query}\n\n[컨텍스트]\n{context}\n\n[초안]\n{draft}\n\n"
             "이 초안의 문제점을 짧게 지적하라. 문제가 없으면 '문제 없음'이라고만 답하라.",
    )

    revised = llm.generate(
        system="너는 비평을 반영해 응답을 한 번 수정한다. 컨텍스트에 없는 수치를 지어내지 마라.",
        user=f"[질문]\n{query}\n\n[컨텍스트]\n{context}\n\n[초안]\n{draft}\n\n[비평]\n{critique}\n\n"
             "수정된 최종 응답만 출력하라.",
    )

    return {"system": "self_rag", "final_response": revised, "sources": sources,
            "need_retrieval": need_retrieval, "critique": critique,
            "sandbox_value": None, "final_ok": True, "attempts_used": 0}
