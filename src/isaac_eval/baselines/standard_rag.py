"""Standard RAG 기준선.

⚠️ 마스킹 없음 — 실제 관측 수치가 그대로 프롬프트에 들어간다. 이것이 제안 방식과의
대조 조건이며, 코드 생성·샌드박스를 쓰지 않고 컨텍스트를 LLM에 직접 준다.
"""
from __future__ import annotations

from isaac_eval.retrieval import search

SYSTEM = "너는 주어진 컨텍스트만 근거로 답하는 어시스턴트다."


def build_data_context(df, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def answer(query: str, cfg, llm, chunk_index=None, df=None) -> dict:
    parts = []
    sources: list[str] = []

    if df is not None:
        parts.append(f"[데이터셋 미리보기]\n{build_data_context(df)}")

    if chunk_index is not None:
        hits = search(chunk_index, query, cfg["retrieval"]["top_k"], cfg["embedding"]["model"])
        sources = [cid for cid, _, _ in hits]
        parts.append("[문헌 발췌]\n" + "\n\n".join(f"[{cid}]\n{pl['text']}" for cid, _, pl in hits))

    context = "\n\n".join(parts)
    response = llm.generate(
        system=SYSTEM,
        user=f"[컨텍스트]\n{context}\n\n[질문]\n{query}\n\n컨텍스트를 근거로 답하라.",
    )
    return {"system": "standard_rag", "final_response": response, "sources": sources,
            "sandbox_value": None, "final_ok": True, "attempts_used": 0}
