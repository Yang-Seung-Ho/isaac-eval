from decimal import Decimal
from isaac_eval.numeric import extract_numbers, flatten_facts, compare


def test_station_id_not_parsed_as_number():
    assert extract_numbers("정점 J9에서 관측") == []


def test_section_and_table_refs_excluded():
    assert extract_numbers("4.2절, 표 3, 그림 A 참고") == []


def test_multi_digit_station_id_fully_excluded():
    nums = extract_numbers("정점 J12의 전체 기간 평균 개체 수는 168136.54입니다")
    assert Decimal("2") not in nums
    assert Decimal("12") not in nums
    assert Decimal("168136.54") in nums


def test_month_label_not_counted_as_fact_number():
    nums = extract_numbers("4월 전체 정점의 개체 수 합계는 9074685.76입니다")
    assert Decimal("4") not in nums
    assert Decimal("9074685.76") in nums


def test_extracts_comma_and_unit_numbers():
    nums = extract_numbers("최고 개체 수는 1,675,955 cells/L 로 2.22배 증가했다")
    assert Decimal("1675955") in nums
    assert Decimal("2.22") in nums


def test_extracts_percent():
    nums = extract_numbers("증가율은 100% 였다")
    assert Decimal("100") in nums


def test_flatten_facts_recurses_dict():
    facts = flatten_facts({"max_month": "Apr.", "max_site": "J9", "max_value": 1675955})
    assert Decimal("1675955") in facts


def test_ncr_detects_invented_number():
    facts = [Decimal("1675955")]
    reported = [Decimal("1675955"), Decimal("2000000")]
    r = compare(reported, facts, policy="round")
    assert r.pass_ is False
    assert Decimal("2000000") in r.unmatched


def test_ncr_round_policy_matches_rounded_float():
    facts = [Decimal("1675954.77")]
    reported = [Decimal("1675955")]
    r = compare(reported, facts, policy="round")
    assert r.pass_ is True
    assert Decimal("1675955") in r.matched_rounded


def test_ncr_exact_policy_rejects_rounded_float():
    facts = [Decimal("1675954.77")]
    reported = [Decimal("1675955")]
    r = compare(reported, facts, policy="exact")
    assert r.pass_ is False


def test_markdown_list_markers_not_counted_as_facts():
    text = "**1. 발생 월**\n* **2. 정점:** J9\n3) 값: 1675954.77"
    nums = extract_numbers(text)
    assert Decimal("1") not in nums
    assert Decimal("2") not in nums
    assert Decimal("3") not in nums
    assert Decimal("1675954.77") in nums


def test_cited_chunk_id_not_counted_as_fact():
    text = "저층 영양염을 보충합니다 (`kim2024_hakashiwo::0017`). 합계는 9074685.76입니다."
    nums = extract_numbers(text)
    assert Decimal("17") not in nums
    assert Decimal("2024") not in nums
    assert Decimal("9074685.76") in nums


def test_wilson_interval_handles_perfect_score():
    """회귀: 30/30처럼 비율이 1.0일 때 정규근사는 폭 0인 구간을 주지만,
    Wilson 구간은 유효한 상한·하한을 준다."""
    from isaac_eval.metrics import wilson_interval
    lo, hi = wilson_interval(30, 30)
    assert 0.0 < lo < 1.0
    assert hi == 1.0


def test_wilson_interval_matches_known_value():
    from isaac_eval.metrics import wilson_interval
    lo, hi = wilson_interval(17, 30)   # p = 0.5667
    assert abs(lo - 0.3920) < 0.001
    assert abs(hi - 0.7262) < 0.001
