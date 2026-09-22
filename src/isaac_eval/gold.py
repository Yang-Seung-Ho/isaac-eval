from __future__ import annotations
from typing import Callable
import pandas as pd


def load_dataset(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.rename(columns={df.columns[0]: "Month"})
    return df


def _station_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith("J") and c[1:].isdigit()]


def spring_peak(df: pd.DataFrame) -> dict:
    spring = df[df["Month"].isin(["Mar.", "Apr.", "May"])]
    station_cols = _station_cols(df)
    row_means = spring[station_cols].mean(axis=1)
    max_idx = row_means.idxmax()
    max_month = spring.loc[max_idx, "Month"]
    row_values = spring.loc[max_idx, station_cols]
    max_site = row_values.idxmax()
    max_value = round(float(row_values.max()))
    return {"max_month": max_month, "max_site": max_site, "max_value": max_value}


def station_mean(df: pd.DataFrame, station: str) -> dict:
    return {"station": station, "mean": round(float(df[station].mean()), 2)}


def month_total(df: pd.DataFrame, month: str) -> dict:
    station_cols = _station_cols(df)
    row = df[df["Month"] == month]
    total = round(float(row[station_cols].sum(axis=1).iloc[0]))
    return {"month": month, "total": total}


def overall_max_station(df: pd.DataFrame) -> dict:
    station_cols = _station_cols(df)
    stacked = df[station_cols].stack()
    idx = stacked.idxmax()
    return {"station": idx[1], "value": round(float(stacked.max()))}


def overall_min_station(df: pd.DataFrame) -> dict:
    station_cols = _station_cols(df)
    stacked = df[station_cols].stack()
    idx = stacked.idxmin()
    return {"station": idx[1], "value": round(float(stacked.min()))}


def station_extreme_month(df: pd.DataFrame, station: str, mode: str = "max") -> dict:
    series = df.set_index("Month")[station]
    month = series.idxmax() if mode == "max" else series.idxmin()
    value = series.max() if mode == "max" else series.min()
    return {"station": station, "mode": mode, "month": month, "value": round(float(value))}


def chl_extreme_month(df: pd.DataFrame, mode: str = "max") -> dict:
    series = df.set_index("Month")["Chl. 평균"]
    month = series.idxmax() if mode == "max" else series.idxmin()
    value = series.max() if mode == "max" else series.min()
    return {"mode": mode, "month": month, "chl_mean": round(float(value), 3)}


def month_field(df: pd.DataFrame, month: str, field: str) -> dict:
    row = df[df["Month"] == month]
    value = round(float(row[field].iloc[0]), 4)
    return {"month": month, "field": field, "value": value}


def count_above_threshold(df: pd.DataFrame, threshold: float) -> dict:
    # The threshold is an input echoed from the query, not a computed value, so
    # it stays out of the returned dict — AA compares computed numbers only.
    station_cols = _station_cols(df)
    count = int((df[station_cols] > threshold).sum().sum())
    return {"count": count}


GOLD_REGISTRY: dict[str, Callable] = {
    "gold.spring_peak": spring_peak,
    "gold.station_mean": station_mean,
    "gold.month_total": month_total,
    "gold.overall_max_station": overall_max_station,
    "gold.overall_min_station": overall_min_station,
    "gold.station_extreme_month": station_extreme_month,
    "gold.chl_extreme_month": chl_extreme_month,
    "gold.month_field": month_field,
    "gold.count_above_threshold": count_above_threshold,
}
