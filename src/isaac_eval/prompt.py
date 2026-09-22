from __future__ import annotations
import json
from isaac_eval.ontology import assert_no_leakage

CODE_CONTRACT = """\
코드 생성 규칙 (반드시 준수):
- pandas as pd, json 만 사용한다 (다른 라이브러리 import 금지)
- 데이터프레임은 이미 변수 df 로 로드되어 있다 (다시 읽지 않는다)
- 마지막 줄은 반드시: print(json.dumps(result, ensure_ascii=False))
- result 는 dict 여야 한다
- 반올림이 필요하면 round() 를 사용한다. int() 절삭은 금지한다
- 수치 연산을 네가 직접 암산하지 말고 반드시 코드로 계산한다
- 설명·주석·마크다운 코드펜스 없이 순수 Python 코드만 출력한다
"""


def build_data_prompt(catalog: dict, query: str) -> str:
    data_dict_json = json.dumps(catalog["data_dictionary"], ensure_ascii=False, indent=2)
    ontology_json = json.dumps(catalog["structural_ontology"], ensure_ascii=False, indent=2)
    return f"""\
[Data Dictionary]
{data_dict_json}

[Structural Ontology]
{ontology_json}

[User Query]
{query}

{CODE_CONTRACT}
"""


def assert_prompt_clean(prompt: str, df) -> list[str]:
    return assert_no_leakage(df, prompt)
