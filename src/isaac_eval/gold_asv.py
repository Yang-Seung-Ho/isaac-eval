"""미생물 군집(ASV) 데이터셋용 정답(gold) 함수.

제주 식물플랑크톤(gold.py)과 동일한 계약을 따른다:
- 각 함수는 (df, **args) 를 받아 계산된 수치를 담은 dict 를 반환한다.
- 반환 dict 의 '수치 내용'만 채점된다(metrics.answer_accuracy). 분류군 이름 등
  문자열 값은 참고용이며 AA 채점에서는 무시된다.
- 모든 값은 코드로 결정론적으로 계산되며 LLM 을 호출하지 않는다.

데이터 스키마: 분류 7열(ASV 서열, Kingdom, Phylum, Class, Order, Family, Genus)
+ 샘플 12열 {배지}_{반복1..3}_D{0,4,8,12} (정수 read count). 배지 접두사에
의존하지 않도록, 샘플 열은 '수치형 열'로 자동 식별한다.
"""
from __future__ import annotations
from typing import Callable
import pandas as pd


def load_dataset(path: str, sheet: str) -> pd.DataFrame:
    # 제주 로더와 달리 첫 열 이름을 바꾸지 않는다(ASV 서열 열을 보존).
    return pd.read_excel(path, sheet_name=sheet)


def _sample_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _day_cols(df: pd.DataFrame, day: str) -> list[str]:
    suffix = f"_{day}"
    return [c for c in _sample_cols(df) if str(c).endswith(suffix)]


def sample_total(df: pd.DataFrame, sample: str) -> dict:
    return {"sample": sample, "total": int(df[sample].sum())}


def sample_max(df: pd.DataFrame, sample: str) -> dict:
    return {"sample": sample, "max": int(df[sample].max())}


def grand_total(df: pd.DataFrame) -> dict:
    sc = _sample_cols(df)
    return {"total": int(df[sc].sum().sum())}


def day_total(df: pd.DataFrame, day: str) -> dict:
    cols = _day_cols(df, day)
    return {"day": day, "total": int(df[cols].sum().sum())}


def day_diff(df: pd.DataFrame, day_minuend: str, day_subtrahend: str) -> dict:
    a = int(df[_day_cols(df, day_minuend)].sum().sum())
    b = int(df[_day_cols(df, day_subtrahend)].sum().sum())
    return {"day_minuend": day_minuend, "day_subtrahend": day_subtrahend, "diff": a - b}


def nunique_taxon(df: pd.DataFrame, rank: str) -> dict:
    # nunique()는 결측(미분류)을 제외한 '명명된' 분류군 수를 센다.
    return {"rank": rank, "n": int(df[rank].nunique())}


def top_taxon_by_total(df: pd.DataFrame, rank: str) -> dict:
    sc = _sample_cols(df)
    # groupby는 결측 분류군 행을 자동 제외한다.
    totals = df.groupby(rank)[sc].sum().sum(axis=1)
    taxon = totals.idxmax()
    return {"rank": rank, "taxon": str(taxon), "total": int(totals.max())}


def taxon_total(df: pd.DataFrame, rank: str, value: str) -> dict:
    sc = _sample_cols(df)
    total = int(df.loc[df[rank] == value, sc].sum().sum())
    return {"rank": rank, "value": value, "total": total}


def count_rows_in_taxon(df: pd.DataFrame, rank: str, value: str) -> dict:
    # 각 행이 하나의 ASV이므로 조건을 만족하는 행 수 = 해당 분류군의 ASV 수.
    return {"rank": rank, "value": value, "count": int((df[rank] == value).sum())}


GOLD_REGISTRY: dict[str, Callable] = {
    "gold.sample_total": sample_total,
    "gold.sample_max": sample_max,
    "gold.grand_total": grand_total,
    "gold.day_total": day_total,
    "gold.day_diff": day_diff,
    "gold.nunique_taxon": nunique_taxon,
    "gold.top_taxon_by_total": top_taxon_by_total,
    "gold.taxon_total": taxon_total,
    "gold.count_rows_in_taxon": count_rows_in_taxon,
}
