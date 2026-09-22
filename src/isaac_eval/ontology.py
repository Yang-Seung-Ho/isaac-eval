from __future__ import annotations
import json
import re
from decimal import Decimal
import pandas as pd
from pandas.api.types import is_numeric_dtype

MASK = "[Numeric values for calculation]"


def extract_structural_ontology(df: pd.DataFrame, max_enum: int = 20) -> dict:
    schema = {}
    for col in df.columns:
        series = df[col]
        if is_numeric_dtype(series):
            schema[col] = {"type": str(series.dtype), "samples": MASK}
            continue
        uniques = series.dropna().unique().tolist()
        samples = uniques if len(uniques) <= max_enum else uniques[:5]
        schema[col] = {"type": str(series.dtype), "samples": samples}
    return {"schema": schema}


def _numeric_value_strings(df: pd.DataFrame) -> set[str]:
    variants: set[str] = set()
    for col in df.columns:
        series = df[col]
        if not is_numeric_dtype(series):
            continue
        for v in series.dropna().tolist():
            dec = Decimal(str(v))
            as_float = float(dec)
            variants.add(str(as_float))
            variants.add(f"{as_float:,.2f}")
            variants.add(f"{as_float:,.0f}")
            if as_float == int(as_float):
                variants.add(str(int(as_float)))
                variants.add(f"{int(as_float):,}")
    return variants


def assert_no_leakage(df: pd.DataFrame, payload: str, min_digits: int = 4) -> list[str]:
    candidates = _numeric_value_strings(df)
    leaked = []
    for candidate in candidates:
        digits_only = re.sub(r"[^\d]", "", candidate)
        if len(digits_only) < min_digits:
            continue
        if candidate in payload:
            leaked.append(candidate)
    return sorted(set(leaked))


def masking_report(df: pd.DataFrame, ontology: dict) -> dict:
    schema = ontology["schema"]
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    enumerated_values = {
        c: schema[c]["samples"] for c in categorical_cols if isinstance(schema[c]["samples"], list)
    }
    payload = json.dumps(ontology, ensure_ascii=False)
    leaked = assert_no_leakage(df, payload)
    numeric_masked = sum(1 for c in numeric_cols if schema[c]["samples"] == MASK)
    nan_cells_in_stations = int(df[numeric_cols].isna().sum().sum())
    return {
        "columns_total": len(df.columns),
        "numeric_masked": numeric_masked,
        "numeric_masked_pct": round(numeric_masked / len(numeric_cols) * 100, 2) if numeric_cols else 0.0,
        "categorical_enumerated": len(categorical_cols),
        "enumerated_values": enumerated_values,
        "payload_chars": len(payload),
        "leaked_values": leaked,
        "nan_cells_in_stations": nan_cells_in_stations,
    }
