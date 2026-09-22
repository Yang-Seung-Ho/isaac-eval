from __future__ import annotations
import json
from pathlib import Path
from isaac_eval.gold import load_dataset
from isaac_eval.ontology import extract_structural_ontology

DATA_DICTIONARY = {
    "Month": "관측 월 (범주형)",
    "J1": "조사 정점 1의 식물플랑크톤 개체 수 (cells/L)",
    "J2": "조사 정점 2의 식물플랑크톤 개체 수 (cells/L)",
    "J3": "조사 정점 3의 식물플랑크톤 개체 수 (cells/L)",
    "J4": "조사 정점 4의 식물플랑크톤 개체 수 (cells/L)",
    "J5": "조사 정점 5의 식물플랑크톤 개체 수 (cells/L)",
    "J6": "조사 정점 6의 식물플랑크톤 개체 수 (cells/L)",
    "J7": "조사 정점 7의 식물플랑크톤 개체 수 (cells/L)",
    "J8": "조사 정점 8의 식물플랑크톤 개체 수 (cells/L)",
    "J9": "조사 정점 9의 식물플랑크톤 개체 수 (cells/L)",
    "J10": "조사 정점 10의 식물플랑크톤 개체 수 (cells/L)",
    "J11": "조사 정점 11의 식물플랑크톤 개체 수 (cells/L)",
    "J12": "조사 정점 12의 식물플랑크톤 개체 수 (cells/L)",
    "평균": "해당 월 전 정점 평균 개체 수",
    "표준편차": "해당 월 전 정점 표준편차",
    "Chl. 평균": "해당 월 클로로필-a 평균 농도",
    "Chl. 표준편차": "해당 월 클로로필-a 표준편차",
}


def build_catalog_entry(cfg: dict) -> dict:
    paths = cfg["paths"]
    df = load_dataset(paths["dataset"], paths["sheet"])
    ontology = extract_structural_ontology(df)
    return {
        "data_name": "제주 연안 식물플랑크톤 정량분석",
        "file_path": paths["dataset"],
        "sheet": paths["sheet"],
        "data_dictionary": {k: v for k, v in DATA_DICTIONARY.items() if k in df.columns},
        "structural_ontology": ontology,
    }


def save_catalog(entry: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
