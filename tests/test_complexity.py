from isaac_eval.complexity import score_v1, score_v2, ScoreBreakdown


CHUNKS = [
    "Heterosigma akashiwo is a harmful algal bloom species forming red tides in coastal waters.",
    "Vertical migration and population dynamics were monitored across stations during the bloom.",
]


def test_v1_returns_breakdown_with_total_equal_to_sum():
    s = score_v1(CHUNKS, "적조란 무엇인가?")
    assert isinstance(s, ScoreBreakdown)
    assert round(s.dim1 + s.dim2 + s.dim3, 4) == s.total


def test_v1_dim1_capped_at_ten():
    many_terms = " ".join(["phytoplankton", "적조", "harmful algal bloom", "chlorophyll",
                           "station", "cells/L", "bloom", "monitoring", "제주", "연안"])
    s = score_v1([many_terms], "질의")
    assert s.dim1 == 10.0


def test_v2_dimensions_are_bounded_zero_to_ten():
    s = score_v2(CHUNKS, "Heterosigma akashiwo 적조 phytoplankton 개체 수", top1_cos=0.8)
    for value in (s.dim1, s.dim2, s.dim3):
        assert 0.0 <= value <= 10.0


def test_v2_dim2_is_zero_when_retrieval_is_strong():
    s = score_v2(CHUNKS, "적조", top1_cos=0.9)
    assert s.dim2 == 0.0


def test_v2_dim2_is_max_when_retrieval_fails():
    s = score_v2(CHUNKS, "적조", top1_cos=0.0)
    assert s.dim2 == 10.0


def test_v1_dim3_is_unbounded_by_design():
    """회귀 방지: v1의 dim3은 정규화되지 않아 10을 크게 넘어설 수 있다.
    이것이 경로 A 도달 불가의 원인이며, 논문의 실측 결과이므로 보존한다."""
    long_text = " ".join(["word"] * 500)
    s = score_v1([long_text], "짧은 질의")
    assert s.dim3 > 10.0


def test_scope_rule_ignores_single_paper_synthesis_verbs():
    """회귀: '종합하라', '교차하여 분석'은 한 편의 논문 안에서도 성립하므로
    다중 논문 신호가 아니다. 초기 규칙이 이를 신호로 취급해 경로 B를 C로 오분류했다."""
    from isaac_eval.scope import keyword_multi_paper_v1, keyword_multi_paper_v2
    q = "양이온성 코로나 공학의 물리화학적 기작을 표면 전하·입자 크기 관점에서 종합하라."
    assert keyword_multi_paper_v1(q) is True    # 초기 규칙: 오탐
    assert keyword_multi_paper_v2(q) is False   # 개정 규칙: 정상


def test_scope_rule_still_detects_real_multi_paper_queries():
    from isaac_eval.scope import keyword_multi_paper_v2
    for q in ["이 코퍼스에 포함된 나노제형 연구들이 공통적으로 채택한 전략은?",
              "등록된 논문들 전반에서 사용된 정량 평가 방법론을 비교하라.",
              "Compare the methods across papers in this corpus."]:
        assert keyword_multi_paper_v2(q) is True, q
