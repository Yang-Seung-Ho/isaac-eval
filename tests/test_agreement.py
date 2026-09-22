from isaac_eval.runners.score_agreement import cohen_kappa


def test_perfect_agreement_gives_kappa_one():
    pairs = [("data", "data"), ("literature", "literature"), ("hybrid", "hybrid")] * 3
    assert cohen_kappa(pairs)["kappa"] == 1.0


def test_known_kappa_value():
    """Wikipedia의 표준 예제: 두 평가자가 각 20/30건에 yes.
    a=20(둘 다 yes), b=5, c=10, d=15 -> 관측 0.70, 기대 0.50, kappa 0.40"""
    pairs = ([("yes", "yes")] * 20 + [("yes", "no")] * 5
             + [("no", "yes")] * 10 + [("no", "no")] * 15)
    r = cohen_kappa(pairs)
    assert r["observed_agreement"] == 0.70
    assert abs(r["kappa"] - 0.40) < 0.001


def test_no_label_variance_returns_none_not_crash():
    """회귀: 두 주석자가 모두 한 라벨만 쓰면 기대 일치도가 1이 되어
    kappa의 분모가 0이 된다. 0으로 나누지 않고 None을 반환해야 한다."""
    pairs = [("data", "data")] * 10
    r = cohen_kappa(pairs)
    assert r["kappa"] is None
    assert r["observed_agreement"] == 1.0


def test_empty_input():
    assert cohen_kappa([])["kappa"] is None
