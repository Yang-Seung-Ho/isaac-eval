"""모듈 C — 질의 복잡도 산출 (v1 / v2 동시 제공).

v1은 논문 4.3절에 게재된 초기 공식이다. 실패(경로 A 도달 불가)가 논문의 실측
결과이므로 **수정하지 않고 그대로 보존한다.**
v2는 재정규화·방향 재정렬 버전이다.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

_STOPWORDS_KO = {
    "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로", "으로", "에서",
    "무엇", "무엇인가", "어떤", "어떻게", "왜", "그리고", "또한", "대해", "대한", "관한",
    "하는", "한다", "인가", "입니까", "설명해줘", "알려줘", "무엇입니까",
}
_STOPWORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "for", "to", "and",
    "or", "what", "why", "how", "which", "does", "do", "did", "with", "as", "by", "that",
    "this", "these", "those", "it", "its", "be", "been", "from", "at", "can",
}
STOPWORDS = _STOPWORDS_KO | _STOPWORDS_EN

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+|[가-힣]+|\d+")


def load_domain_terms(path: str | Path = "config/domain_terms.json") -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


DOMAIN_TERMS: list[str] = load_domain_terms()


@dataclass
class ScoreBreakdown:
    dim1: float
    dim2: float
    dim3: float
    total: float

    def as_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS]


def _term_hits(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def score_v1(chunks: list[str], query: str, terms: list[str] | None = None) -> ScoreBreakdown:
    """논문 4.3절에 게재된 초기 공식. 그대로 보존한다.

    dim1 = min(term_hits * 1.5, 10)                        # 검색된 본문의 도메인 용어 밀도
    dim2 = (1 - mean_jaccard(adjacent chunks)) * 10        # 청크 간 이질성
    dim3 = |query_words - text_words| / |query_words| * 10 # 길이 격차
    """
    terms = terms if terms is not None else DOMAIN_TERMS
    text = " ".join(chunks)

    dim1 = min(_term_hits(text, terms) * 1.5, 10.0)

    if len(chunks) < 2:
        dim2 = 0.0
    else:
        sims = []
        for first, second in zip(chunks, chunks[1:]):
            sims.append(_jaccard(set(_tokens(first)), set(_tokens(second))))
        dim2 = (1 - sum(sims) / len(sims)) * 10

    query_words = len(_tokens(query))
    text_words = len(_tokens(text))
    dim3 = abs(query_words - text_words) / query_words * 10 if query_words else 0.0

    return ScoreBreakdown(round(dim1, 4), round(dim2, 4), round(dim3, 4), round(dim1 + dim2 + dim3, 4))


def score_v2(chunks: list[str], query: str, top1_cos: float, terms: list[str] | None = None) -> ScoreBreakdown:
    """재정규화·방향 재정렬 버전. 각 차원이 [0, 10]으로 유계이고 방향이 일치한다.

    dim1 = min(query_domain_term_hits / 3, 1) * 10   # 질의 특이도
    dim2 = max(0, 1 - top1_cos / 0.35) * 10          # 검색 불충분도
    dim3 = |query_content_words - text_words| / |query_content_words| * 10  (10에서 절단)
    """
    terms = terms if terms is not None else DOMAIN_TERMS
    text = " ".join(chunks)

    dim1 = min(_term_hits(query, terms) / 3, 1.0) * 10

    dim2 = max(0.0, 1 - top1_cos / 0.35) * 10

    query_content = len(_content_tokens(query))
    text_words = len(_tokens(text))
    if query_content:
        raw = abs(query_content - text_words) / query_content * 10
        dim3 = min(raw, 10.0)
    else:
        dim3 = 10.0

    return ScoreBreakdown(round(dim1, 4), round(dim2, 4), round(dim3, 4), round(dim1 + dim2 + dim3, 4))


# 임베딩 검색에서 관측된 top-1 코사인 범위. v2의 0.35 상수는 TF-IDF 기준이라
# 임베딩 코사인(무관한 텍스트도 0.5 이상)에서는 dim2가 항상 0이 된다.
# 이 두 상수는 코퍼스 관측 분포에서 유도했으며, 검색 방식이 바뀌면 재보정해야 한다.
COS_STRONG = 0.85
COS_WEAK = 0.55
QUERY_LEN_SATURATION = 25


def score_v3(chunks: list[str], query: str, top1_cos: float, terms: list[str] | None = None) -> ScoreBreakdown:
    """임베딩 검색 코퍼스에 맞춰 재보정한 버전.

    v2가 이 코퍼스에서 실패한 세 원인을 각각 교정한다:
      dim1 — 텍스트 길이 격차(항상 포화) 대신 질의 자체의 내용어 수로 다면성을 잰다.
      dim2 — 코사인 임계를 TF-IDF 기준 0.35에서 임베딩 관측 범위로 재보정한다.
      dim3 — 도메인 용어 매칭을 코퍼스 전체 용어집 기준으로 되돌린다.
    """
    terms = terms if terms is not None else DOMAIN_TERMS

    query_content = len(_content_tokens(query))
    dim1 = min(query_content / QUERY_LEN_SATURATION, 1.0) * 10

    span = COS_STRONG - COS_WEAK
    dim2 = min(max((COS_STRONG - top1_cos) / span, 0.0), 1.0) * 10

    dim3 = min(_term_hits(query, terms) / 3, 1.0) * 10

    return ScoreBreakdown(round(dim1, 4), round(dim2, 4), round(dim3, 4), round(dim1 + dim2 + dim3, 4))
