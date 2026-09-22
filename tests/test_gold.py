from isaac_eval import gold


def _df():
    return gold.load_dataset("data/datasets/jeju_phytoplankton.xlsx", "식.플 정량분석")


def test_round_not_truncate_spring_peak():
    result = gold.spring_peak(_df())
    assert result == {"max_month": "Apr.", "max_site": "J9", "max_value": 1675955}


def test_overall_max_station():
    result = gold.overall_max_station(_df())
    assert result["station"] == "J9"
    assert result["value"] == 1675955


def test_month_total_is_sum_of_stations_rounded():
    df = _df()
    row = df[df["Month"] == "Apr."]
    station_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("J") and c[1:].isdigit()]
    expected = round(row[station_cols].sum(axis=1).iloc[0])
    result = gold.month_total(df, "Apr.")
    assert result == {"month": "Apr.", "total": expected}


def test_registry_covers_all_functions():
    assert set(gold.GOLD_REGISTRY) == {
        "gold.spring_peak", "gold.station_mean", "gold.month_total", "gold.overall_max_station",
        "gold.overall_min_station", "gold.station_extreme_month", "gold.chl_extreme_month",
        "gold.month_field", "gold.count_above_threshold",
    }


def test_overall_min_station():
    result = gold.overall_min_station(_df())
    assert result["station"] == "J8"
    assert result["value"] == round(7425.742574257426)


def test_station_extreme_month_max():
    result = gold.station_extreme_month(_df(), "J1", mode="max")
    assert result == {"station": "J1", "mode": "max", "month": "Apr.", "value": round(372691.7475728155)}


def test_station_extreme_month_min():
    result = gold.station_extreme_month(_df(), "J12", mode="min")
    assert result == {"station": "J12", "mode": "min", "month": "Mar.", "value": round(15441.176470588236)}


def test_chl_extreme_month_max():
    result = gold.chl_extreme_month(_df(), mode="max")
    assert result == {"mode": "max", "month": "Aug.", "chl_mean": 2.055}


def test_month_field():
    result = gold.month_field(_df(), "Apr.", "평균")
    assert result == {"month": "Apr.", "field": "평균", "value": round(756223.8131580086, 4)}


def test_count_above_threshold():
    result = gold.count_above_threshold(_df(), 500000)
    assert result == {"count": 8}
