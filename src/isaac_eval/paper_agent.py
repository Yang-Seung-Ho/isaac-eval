"""문헌 에이전트 — 경로 A / B / C 판정 및 응답 생성 (논문 3.5.2절 정의).

경로 A: 단일 논문 + 복잡도 < 임계값 → 검색 청크에 근거한 직접 요약
경로 B: 단일 논문 + 복잡도 >= 임계값 → 강제 합성을 포기하고 논문 추천 카드 반환
경로 C: 다중 논문 질의 → 논문 단위 인덱스로 대상 논문을 고른 뒤 메타데이터 비교

경로 B가 곧 우회(graceful degradation)다. 고복잡도 질의에서 요약을 생성하지 않고
추천으로 전환한 비율이 논문 5.4절의 '우회 전환율'이다.
"""
from __future__ import annotations
import json
import re

from isaac_eval.complexity import score_v1, score_v2, score_v3
from isaac_eval.retrieval import search

from isaac_eval import scope

# 범위 판정 규칙과 프롬프트는 scope.py에 있다. 초기 규칙은 단일 논문 안에서의 심층
# 종합을 요구하는 표현("종합하라")까지 다중 논문 신호로 취급해 경로 B를 C로 오분류했다
# (7.3절). 개정 규칙은 복수 문헌을 명시하는 표현만 신호로 삼는다.
MULTI_PAPER_KEYWORDS = scope.PLURAL_SOURCE_TERMS

SCOPE_SYSTEM = scope.SCOPE_SYSTEM

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    match = _JSON_RE.search(text or "")
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def keyword_multi_paper(query: str) -> bool:
    return scope.keyword_multi_paper_v2(query)


def llm_multi_paper(query: str, llm) -> bool:
    return scope.llm_multi_paper_v2(query, llm)


def compute_complexity(chunks: list[str], query: str, top1_cos: float, version: str):
    if version == "v1":
        return score_v1(chunks, query)
    if version == "v2":
        return score_v2(chunks, query, top1_cos)
    return score_v3(chunks, query, top1_cos)


def answer(query: str, cfg, llm, chunk_index, paper_index) -> dict:
    embed_model = cfg["embedding"]["model"]
    top_k = cfg["retrieval"]["top_k"]
    version = cfg["complexity"]["version"]
    threshold = cfg["complexity"]["threshold"]
    if threshold is None:
        raise RuntimeError(
            "complexity.threshold is null — run calibrate_threshold first (RUNBOOK STEP 3). "
            "본 측정을 임계값 없이 돌리면 안 된다."
        )

    kw_multi = keyword_multi_paper(query)
    llm_multi = llm_multi_paper(query, llm)
    is_multi = kw_multi or llm_multi

    hits = search(chunk_index, query, top_k, embed_model)
    chunks = [payload["text"] for _, _, payload in hits]
    top1_cos = hits[0][1] if hits else 0.0
    breakdown = compute_complexity(chunks, query, top1_cos, version)

    if is_multi:
        paper_hits = search(paper_index, query, min(3, len(paper_index.ids)), embed_model)
        selected_papers = [pid for pid, _, _ in paper_hits]
        per_paper_chunks = []
        for pid in selected_papers:
            subset = [
                (cid, sc, pl) for cid, sc, pl in
                search(chunk_index, query, top_k * 4, embed_model)
                if pl["paper_id"] == pid
            ][:2]
            per_paper_chunks.extend(subset)
        context = "\n\n".join(
            f"[{pl['paper_id']} / {cid}]\n{pl['text']}" for cid, _, pl in per_paper_chunks
        )
        response = llm.generate(
            system="너는 여러 논문의 근거를 종합하는 연구 어시스턴트다. 제공된 발췌 밖의 내용을 지어내지 마라.",
            user=f"질의: {query}\n\n[여러 논문에서 검색된 발췌]\n{context}\n\n"
                 "논문별 근거를 구분해 제시하고 종합하라.",
        )
        return {
            "paper_route": "C",
            "complexity": breakdown.as_dict(),
            "response": response,
            "sources": [cid for cid, _, _ in per_paper_chunks],
            "selected_papers": selected_papers,
            "degraded": False,
            "scope_kw_multi": kw_multi,
            "scope_llm_multi": llm_multi,
            "top1_cos": round(top1_cos, 4),
        }

    is_complex = breakdown.total >= threshold
    if is_complex:
        # 경로 B — 강제 합성을 포기하고 추천 카드만 반환한다. 요약 생성 호출 자체가
        # 일어나지 않는 것이 이 경로의 요점이므로, 여기서 LLM을 부르지 않는다.
        paper_hits = search(paper_index, query, min(3, len(paper_index.ids)), embed_model)
        recommendations = [
            {"paper_id": pid, "score": round(score, 4), "n_words": pl.get("n_words")}
            for pid, score, pl in paper_hits
        ]
        titles = "\n".join(f"- {r['paper_id']} (관련도 {r['score']})" for r in recommendations)
        response = (
            "질의의 복잡도가 임계값을 초과하여 직접 요약을 생성하지 않았습니다. "
            "대신 관련성이 높은 등록 논문을 추천합니다.\n" + titles
        )
        return {
            "paper_route": "B",
            "complexity": breakdown.as_dict(),
            "response": response,
            "sources": [],
            "recommendations": recommendations,
            "selected_papers": [r["paper_id"] for r in recommendations],
            "degraded": True,
            "scope_kw_multi": kw_multi,
            "scope_llm_multi": llm_multi,
            "top1_cos": round(top1_cos, 4),
        }

    # 경로 A — 검색된 청크에 근거한 직접 요약
    context = "\n\n".join(f"[{cid}]\n{pl['text']}" for cid, _, pl in hits)
    response = llm.generate(
        system="너는 논문 발췌를 근거로 답하는 연구 어시스턴트다. 발췌 밖의 내용을 지어내지 마라.",
        user=f"질의: {query}\n\n[검색된 발췌]\n{context}\n\n"
             "출처를 표기하여 연구 목적·방법·주요 발견 중심으로 요약하라.",
    )
    return {
        "paper_route": "A",
        "complexity": breakdown.as_dict(),
        "response": response,
        "sources": [cid for cid, _, _ in hits],
        "selected_papers": sorted({pl["paper_id"] for _, _, pl in hits}),
        "degraded": False,
        "scope_kw_multi": kw_multi,
        "scope_llm_multi": llm_multi,
        "top1_cos": round(top1_cos, 4),
    }
