import json
import pandas as pd
from isaac_eval.ontology import (
    extract_structural_ontology,
    masking_report,
    assert_no_leakage,
)


def test_string_dtype_column_is_enumerated_not_masked():
    df = pd.DataFrame({
        "Month": pd.Series(["Mar.", "Apr."], dtype="str"),
        "J1": [1.0, 2.0],
    })
    ont = extract_structural_ontology(df)
    assert ont["schema"]["Month"]["samples"] == ["Mar.", "Apr."]
    assert ont["schema"]["J1"]["samples"] == "[Numeric values for calculation]"


def test_numeric_column_enum_over_max_enum_falls_back_to_five():
    df = pd.DataFrame({"Cat": [f"c{i}" for i in range(30)]})
    ont = extract_structural_ontology(df, max_enum=20)
    assert len(ont["schema"]["Cat"]["samples"]) == 5


def test_real_dataset_no_leakage_and_full_masking():
    df = pd.read_excel("data/datasets/jeju_phytoplankton.xlsx", sheet_name="식.플 정량분석")
    df = df.rename(columns={df.columns[0]: "Month"})
    ont = extract_structural_ontology(df)
    payload = json.dumps(ont, ensure_ascii=False)
    leaks = assert_no_leakage(df, payload)
    assert leaks == []
    report = masking_report(df, ont)
    assert report["leaked_values"] == []
    assert report["numeric_masked_pct"] == 100.0
    assert set(report["enumerated_values"]["Month"]) == {
        "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Oct.", "Nov.",
    }
