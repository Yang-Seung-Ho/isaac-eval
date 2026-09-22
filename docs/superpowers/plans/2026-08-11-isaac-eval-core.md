# Isaac-Eval Core (NCR/AA 실측 하네스) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 정형 질의(structured query)에 대해 수치 일치율(NCR)과 응답 정확도(AA)를 실측하는 최소 하네스를 만든다. 논문 §9 우선순위 1~3 범위: 온톨로지 마스킹, gold 참조 구현, 샌드박스 실행, 프롬프트 조립, 데이터 에이전트, 정형 질의 러너.

**Architecture:** LLM(Gemini)이 마스킹된 스키마만 보고 pandas 코드를 생성 → subprocess 샌드박스에서 실제 데이터로 실행 → 자연어 응답 생성 → 응답 속 수치를 샌드박스 결과값과 대조(NCR), 샌드박스 결과값을 gold 값과 대조(AA). 두 지표는 반드시 별도 컬럼으로 산출한다.

**Tech Stack:** Python 3.13, pandas, openpyxl, google-genai(Gemini API), pytest. Docker 미설치 → sandbox는 subprocess 모드만 구현.

## Global Constraints

- 수치형 컬럼은 무조건 마스킹, 유형 판별은 `pandas.api.types.is_numeric_dtype()`만 사용 (`str(dtype)=='object'` 비교 금지).
- 반올림은 `round()`만 사용, `int()` 절삭 금지.
- NCR과 AA는 반드시 분리된 필드/컬럼으로 산출한다 (섞어서 하나의 지표로 합치지 않는다).
- LLM 호출은 `temperature=0.0`, 코드/설정은 `config/settings.yaml`에서 로드 (하드코딩 금지).
- 자식 프로세스가 `dataset_path`에서 직접 데이터를 로드한다 (부모가 df를 직렬화해 넘기지 않는다).
- 샌드박스는 네트워크 차단 + 타임아웃 강제 (subprocess 모드: `python -I`, `resource.setrlimit`, 임시 cwd, 환경변수 제거).
- API 키는 `.env`에서 로드 (커밋 금지, 이미 `.gitignore` 처리됨).
- 실측하지 않은 값은 표/결과에 넣지 않는다.

---

## File Structure

```
isaac-eval/
├── pyproject.toml
├── Makefile
├── .env                          # 이미 생성됨 (GEMINI_API_KEY)
├── .env.example
├── config/
│   ├── settings.yaml
│   └── domain_terms.json
├── data/
│   ├── datasets/jeju_phytoplankton.xlsx   # 이미 배치됨
│   ├── papers/kim2024_hakashiwo.pdf       # 이미 배치됨 (본 범위에서는 미사용)
│   └── catalog/                            # 생성됨
├── queries/
│   └── structured.jeju.jsonl
├── src/isaac_eval/
│   ├── __init__.py
│   ├── config.py
│   ├── llm.py                # Gemini 클라이언트 래퍼
│   ├── ontology.py           # 모듈 A
│   ├── catalog.py
│   ├── prompt.py
│   ├── sandbox.py            # 모듈 B (subprocess)
│   ├── numeric.py            # NCR 핵심
│   ├── gold.py                # AA 참조 구현
│   ├── data_agent.py
│   ├── metrics.py
│   └── runners/
│       ├── __init__.py
│       ├── build_catalog.py
│       └── run_structured.py
├── tests/
│   ├── test_ontology.py
│   ├── test_numeric.py
│   ├── test_sandbox.py
│   └── test_gold.py
└── results/                   # 생성됨
```

---

### Task 1: 프로젝트 스캐폴딩 + 설정

**Files:**
- Create: `pyproject.toml`
- Create: `config/settings.yaml`
- Create: `config/domain_terms.json`
- Create: `.env.example`
- Create: `src/isaac_eval/__init__.py`
- Create: `src/isaac_eval/config.py`
- Create: `Makefile`

**Interfaces:**
- Produces: `load_settings(path: str = "config/settings.yaml") -> dict` (dict는 `dotted.key` 접근을 지원하는 얕은 래퍼 `AttrDict` 또는 단순 `dict` + 헬퍼 `get(cfg, "sandbox.timeout_s")`). 이후 모든 모듈은 이 dict를 인자로 받는다.
- Produces: `.env`에서 `GEMINI_API_KEY`를 읽어 `os.environ`에 있으면 그대로 쓰고 없으면 `.env`를 파싱해 주입하는 `load_env()`.

- [ ] **Step 1: `pyproject.toml` 작성**

```toml
[project]
name = "isaac-eval"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "openpyxl>=3.1",
    "pyyaml>=6.0",
    "google-genai>=0.3.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: `config/settings.yaml` 작성**

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash
  temperature: 0.0
  top_p: 1.0
  max_tokens: 2048
  seed: 20260811
sandbox:
  mode: subprocess   # docker 미설치 환경 — subprocess만 구현
  timeout_s: 10
  memory_mb: 512
  max_attempts: 3
  network: false
numeric:
  match_policy: round   # exact | round
  sigfig: 6
  min_digits_for_leak_scan: 4
paths:
  dataset: data/datasets/jeju_phytoplankton.xlsx
  sheet: "식.플 정량분석"
  papers_dir: data/papers
  results_dir: results
  catalog_dir: data/catalog
  queries_structured: queries/structured.jeju.jsonl
```

- [ ] **Step 3: `config/domain_terms.json` 작성** (실제 데이터셋/논문 도메인 용어, complexity 모듈 대비용이지만 이번 범위엔 미사용 — 존재만 시켜 §9 우선순위 4 확장 시 재작업 없게 함)

```json
["phytoplankton", "식물플랑크톤", "Heterosigma akashiwo", "적조", "harmful algal bloom",
 "chlorophyll", "엽록소", "정점", "station", "개체 수", "cells/L", "정량분석",
 "제주", "연안", "계절", "봄철", "표준편차", "평균", "vertical migration", "population dynamics",
 "sexual reproduction", "bloom", "densities", "seasonal variation", "monitoring"]
```

- [ ] **Step 4: `.env.example` 작성**

```
GEMINI_API_KEY=
```

- [ ] **Step 5: `src/isaac_eval/config.py` 작성**

```python
from __future__ import annotations
import os
from pathlib import Path
import yaml


def load_env(dotenv_path: str | Path = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings(path: str | Path = "config/settings.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(cfg: dict, dotted_key: str):
    node = cfg
    for part in dotted_key.split("."):
        node = node[part]
    return node
```

- [ ] **Step 6: `Makefile` 작성**

```makefile
setup:
	pip install -e ".[dev]"
catalog:
	python -m isaac_eval.runners.build_catalog
test:
	pytest -q
structured:
	python -m isaac_eval.runners.run_structured
all: test catalog structured
```

- [ ] **Step 7: 설치 및 확인**

Run: `cd isaac-eval && pip install -e ".[dev]"`
Expected: 의존성 설치 성공 (google-genai, pandas, pyyaml, python-dotenv, pytest)

- [ ] **Step 8: Commit**

```bash
git init -q 2>/dev/null; git add pyproject.toml config Makefile .env.example src/isaac_eval/__init__.py src/isaac_eval/config.py .gitignore
git commit -m "chore: scaffold isaac-eval project and settings"
```

---

### Task 2: `ontology.py` — 구조적 온톨로지 추출 + 마스킹 (모듈 A) + 회귀 테스트

**Files:**
- Create: `src/isaac_eval/ontology.py`
- Test: `tests/test_ontology.py`

**Interfaces:**
- Consumes: `pandas.DataFrame`
- Produces:
  - `extract_structural_ontology(df: pd.DataFrame, max_enum: int = 20) -> dict` → `{"schema": {col: {"type": str, "samples": list|str}}}`
  - `masking_report(df: pd.DataFrame, ontology: dict) -> dict` → 표 2용 dict (스키마는 스펙 §5.1 그대로)
  - `assert_no_leakage(df: pd.DataFrame, payload: str, min_digits: int = 4) -> list[str]`

- [ ] **Step 1: 실패하는 테스트 작성 (dtype 회귀 + 유출 회귀)**

```python
# tests/test_ontology.py
import json
import pandas as pd
import pytest
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `pytest tests/test_ontology.py -v`
Expected: FAIL (`ModuleNotFoundError: isaac_eval.ontology`)

- [ ] **Step 3: `ontology.py` 구현**

```python
from __future__ import annotations
import re
from decimal import Decimal
import pandas as pd
from pandas.api.types import is_numeric_dtype

MASK = "[Numeric values for calculation]"

_NUMBER_VARIANT_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_structural_ontology(df: pd.DataFrame, max_enum: int = 20) -> dict:
    schema = {}
    for col in df.columns:
        series = df[col]
        if is_numeric_dtype(series):
            schema[col] = {"type": str(series.dtype), "samples": MASK}
            continue
        uniques = [v for v in series.dropna().unique().tolist()]
        if len(uniques) <= max_enum:
            samples = uniques
        else:
            samples = uniques[:5]
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
    import json

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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `pytest tests/test_ontology.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/isaac_eval/ontology.py tests/test_ontology.py
git commit -m "feat: add structural ontology extraction and leakage regression tests"
```

---

### Task 3: `numeric.py` — NCR 핵심 로직 + 회귀 테스트

**Files:**
- Create: `src/isaac_eval/numeric.py`
- Test: `tests/test_numeric.py`

**Interfaces:**
- Produces:
  - `extract_numbers(text: str) -> list[Decimal]`
  - `flatten_facts(value) -> list[Decimal]`
  - `ComparisonResult` dataclass: `matched_exact: list[Decimal]`, `matched_rounded: list[Decimal]`, `unmatched: list[Decimal]`, `pass_: bool`
  - `compare(reported: list[Decimal], facts: list[Decimal], policy: str, sigfig: int = 6) -> ComparisonResult`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_numeric.py
from decimal import Decimal
from isaac_eval.numeric import extract_numbers, flatten_facts, compare


def test_station_id_not_parsed_as_number():
    assert extract_numbers("정점 J9에서 관측") == []


def test_section_and_table_refs_excluded():
    assert extract_numbers("4.2절, 표 3, 그림 A 참고") == []


def test_extracts_comma_and_unit_numbers():
    nums = extract_numbers("최고 개체 수는 1,675,955 cells/L 로 2.22배 증가했다")
    assert Decimal("1675955") in nums
    assert Decimal("2.22") in nums


def test_extracts_scientific_and_percent():
    nums = extract_numbers("농도는 1.68e6 이었고 증가율은 100% 였다")
    assert Decimal("1680000") in nums or Decimal("1.68E+6") in nums
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `pytest tests/test_numeric.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: `numeric.py` 구현**

```python
from __future__ import annotations
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

_STATION_PREFIXED_RE = re.compile(r"[A-Za-z가-힣]\s*\d+(\.\d+)?")
_SECTION_REF_RE = re.compile(r"\d+(\.\d+)*\s*(절|장|조)|(표|그림)\s*\d+|그림\s*[A-Za-z]")
_NUMBER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z가-힣])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![A-Za-z가-힣])"
    r"|(?<![A-Za-z가-힣])[-+]?\d+\.\d+[eE][-+]?\d+(?![A-Za-z가-힣])"
    r"|(?<![A-Za-z가-힣])[-+]?\d+(?:\.\d+)?%?(?![A-Za-z가-힣])"
)
_YEAR_RE = re.compile(r"^(19|20|21)\d{2}$")


def _is_excluded_span(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 2):start]
    if prefix and re.match(r"[A-Za-z가-힣]\s*$", prefix):
        return True
    window = text[max(0, start - 6):min(len(text), end + 6)]
    if _SECTION_REF_RE.search(window):
        span_text = text[start:end]
        if re.match(r"^\d+(\.\d+)*$", span_text):
            return True
    return False


def extract_numbers(text: str) -> list[Decimal]:
    results: list[Decimal] = []
    for m in _NUMBER_TOKEN_RE.finditer(text):
        raw = m.group(0)
        if _is_excluded_span(text, m.start(), m.end()):
            continue
        cleaned = raw.rstrip("%").replace(",", "")
        try:
            value = Decimal(cleaned)
        except InvalidOperation:
            continue
        results.append(value)
    return results


def flatten_facts(value) -> list[Decimal]:
    out: list[Decimal] = []
    if isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_facts(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(flatten_facts(v))
    elif isinstance(value, bool):
        return out
    elif isinstance(value, (int, float)):
        out.append(Decimal(str(value)))
    elif isinstance(value, str):
        try:
            out.append(Decimal(value))
        except InvalidOperation:
            pass
    return out


@dataclass
class ComparisonResult:
    matched_exact: list = field(default_factory=list)
    matched_rounded: list = field(default_factory=list)
    unmatched: list = field(default_factory=list)
    pass_: bool = False


def compare(reported: list[Decimal], facts: list[Decimal], policy: str, sigfig: int = 6) -> ComparisonResult:
    result = ComparisonResult()
    fact_exact = set(facts)
    fact_rounded = {f.quantize(Decimal("1")) for f in facts}
    for num in reported:
        if num in fact_exact:
            result.matched_exact.append(num)
            continue
        if policy in ("round",) and num.quantize(Decimal("1")) in fact_rounded:
            result.matched_rounded.append(num)
            continue
        if policy == "sigfig":
            matched = False
            for f in facts:
                if _same_sigfig(num, f, sigfig):
                    result.matched_rounded.append(num)
                    matched = True
                    break
            if matched:
                continue
        result.unmatched.append(num)
    result.pass_ = len(result.unmatched) == 0
    return result


def _same_sigfig(a: Decimal, b: Decimal, sigfig: int) -> bool:
    if a == 0 or b == 0:
        return a == b
    from decimal import Context
    ctx = Context(prec=sigfig)
    return ctx.create_decimal(a) == ctx.create_decimal(b)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인, 필요시 정규식 보정**

Run: `pytest tests/test_numeric.py -v`
Expected: PASS (8 passed). 실패 시 `_NUMBER_TOKEN_RE`/`_is_excluded_span` 경계 조건을 각 실패 케이스에 맞춰 조정한다 (예: "2.22배"에서 "배" 앞 숫자는 배제 대상이 아니므로 한글 접미사는 lookahead에서 제외하지 않도록 주의).

- [ ] **Step 5: Commit**

```bash
git add src/isaac_eval/numeric.py tests/test_numeric.py
git commit -m "feat: add NCR numeric extraction and comparison with regression tests"
```

---

### Task 4: `gold.py` — 참조 구현 (AA 기준값) + 실제 데이터 검증 테스트

**Files:**
- Create: `src/isaac_eval/gold.py`
- Test: `tests/test_gold.py`

**Interfaces:**
- Consumes: `pd.DataFrame` (컬럼: `Month`, `J1..J12`, `평균`, `표준편차`, `Chl. 평균`, `Chl. 표준편차`)
- Produces:
  - `load_dataset(path: str, sheet: str) -> pd.DataFrame` (첫 컬럼을 `Month`로 rename)
  - `spring_peak(df) -> dict` → `{"max_month": str, "max_site": str, "max_value": int}`
  - `station_mean(df, station: str) -> dict` → `{"station": str, "mean": float}` (round 2)
  - `month_total(df, month: str) -> dict` → `{"month": str, "total": int}` (12개 정점 합, round)
  - `overall_max_station(df) -> dict` → `{"station": str, "value": int}` (전체 기간 통틀어 단일 관측 최댓값)
  - `GOLD_REGISTRY: dict[str, Callable]` — 위 4개를 `"gold.spring_peak"` 등의 키로 등록

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_gold.py
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
    station_cols = [c for c in df.columns if c.startswith("J")]
    expected = round(row[station_cols].sum(axis=1).iloc[0])
    result = gold.month_total(df, "Apr.")
    assert result == {"month": "Apr.", "total": expected}


def test_registry_covers_all_functions():
    assert set(gold.GOLD_REGISTRY) == {
        "gold.spring_peak", "gold.station_mean", "gold.month_total", "gold.overall_max_station",
    }
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `pytest tests/test_gold.py -v`
Expected: FAIL

- [ ] **Step 3: `gold.py` 구현**

```python
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


GOLD_REGISTRY: dict[str, Callable] = {
    "gold.spring_peak": spring_peak,
    "gold.station_mean": station_mean,
    "gold.month_total": month_total,
    "gold.overall_max_station": overall_max_station,
}
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `pytest tests/test_gold.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/isaac_eval/gold.py tests/test_gold.py
git commit -m "feat: add gold reference implementations validated against real dataset"
```

---

### Task 5: `sandbox.py` — subprocess 격리 실행 (모듈 B) + 테스트

**Files:**
- Create: `src/isaac_eval/sandbox.py`
- Test: `tests/test_sandbox.py`

**Interfaces:**
- Produces:
  - `SandboxResult` dataclass: `ok: bool`, `stdout: str`, `value`, `error: str|None`, `traceback: str|None`, `elapsed_ms: int`, `attempt: int`
  - `run_code(code: str, dataset_path: str, sheet: str, cfg: dict, attempt: int = 1) -> SandboxResult`
  - `run_with_retry(code_generator: Callable[[str|None], str], dataset_path: str, sheet: str, cfg: dict) -> list[SandboxResult]`

**설계 메모:** `code_generator(prev_error: str | None) -> str` — 첫 호출은 `None`, 실패 시 이전 traceback을 전달해 재생성. 자식 프로세스는 `_runner_template.py`를 통해 `dataset_path`/`sheet`에서 직접 `pd.read_excel`로 df를 로드한 뒤 사용자 코드를 exec한다 (부모가 df를 직렬화해 넘기지 않음).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_sandbox.py
import json
from isaac_eval.sandbox import run_code, run_with_retry

CFG = {"sandbox": {"mode": "subprocess", "timeout_s": 5, "memory_mb": 512, "max_attempts": 3, "network": False}}
DATASET = "data/datasets/jeju_phytoplankton.xlsx"
SHEET = "식.플 정량분석"


def test_last_line_json_parsed():
    code = (
        "import json\n"
        "result = {'ok': 1}\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )
    r = run_code(code, DATASET, SHEET, CFG)
    assert r.ok is True
    assert r.value == {"ok": 1}


def test_dataset_loaded_directly_by_child():
    code = (
        "import json\n"
        "result = {'n_rows': len(df)}\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )
    r = run_code(code, DATASET, SHEET, CFG)
    assert r.ok is True
    assert r.value["n_rows"] == 8


def test_timeout_enforced():
    code = "import time\ntime.sleep(30)\n"
    cfg = {"sandbox": {**CFG["sandbox"], "timeout_s": 1}}
    r = run_code(code, DATASET, SHEET, cfg)
    assert r.ok is False
    assert "timeout" in (r.error or "").lower()


def test_network_blocked():
    code = (
        "import socket, json\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=2)\n"
        "    result = {'network': True}\n"
        "except Exception as e:\n"
        "    result = {'network': False}\n"
        "print(json.dumps(result))\n"
    )
    r = run_code(code, DATASET, SHEET, CFG)
    assert r.ok is True
    assert r.value["network"] is False


def test_retry_feeds_traceback():
    calls = []

    def gen(prev_error):
        calls.append(prev_error)
        if prev_error is None:
            return "raise KeyError('boom')\n"
        return "import json\nprint(json.dumps({'fixed': True}))\n"

    results = run_with_retry(gen, DATASET, SHEET, CFG)
    assert results[-1].ok is True
    assert results[0].ok is False
    assert "boom" in results[0].traceback
    assert calls[1] is not None and "boom" in calls[1]
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL

- [ ] **Step 3: `sandbox.py` 구현**

```python
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_RUNNER_TEMPLATE = '''
import sys, json, resource
resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
import pandas as pd

df = pd.read_excel({dataset_path!r}, sheet_name={sheet!r})
df = df.rename(columns={{df.columns[0]: "Month"}})

USER_CODE = {code!r}
exec(compile(USER_CODE, "<user_code>", "exec"), {{"pd": pd, "json": json, "df": df}})
'''


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    value: object
    error: str | None
    traceback: str | None
    elapsed_ms: int
    attempt: int


def run_code(code: str, dataset_path: str, sheet: str, cfg: dict, attempt: int = 1) -> SandboxResult:
    sb_cfg = cfg["sandbox"]
    mem_bytes = sb_cfg["memory_mb"] * 1024 * 1024
    script = _RUNNER_TEMPLATE.format(
        mem_bytes=mem_bytes,
        dataset_path=os.path.abspath(dataset_path),
        sheet=sheet,
        code=code,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text(script, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", "")}
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=sb_cfg["timeout_s"],
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return SandboxResult(False, "", None, "timeout exceeded", None, elapsed_ms, attempt)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if proc.returncode != 0:
            return SandboxResult(False, proc.stdout, None, "execution failed", proc.stderr, elapsed_ms, attempt)
        last_line = next((l for l in reversed(proc.stdout.splitlines()) if l.strip()), "")
        try:
            value = json.loads(last_line)
        except json.JSONDecodeError as e:
            return SandboxResult(False, proc.stdout, None, f"invalid json output: {e}", proc.stdout, elapsed_ms, attempt)
        return SandboxResult(True, proc.stdout, value, None, None, elapsed_ms, attempt)


def run_with_retry(code_generator: Callable[[str | None], str], dataset_path: str, sheet: str, cfg: dict) -> list[SandboxResult]:
    max_attempts = cfg["sandbox"]["max_attempts"]
    results: list[SandboxResult] = []
    prev_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        code = code_generator(prev_error)
        result = run_code(code, dataset_path, sheet, cfg, attempt=attempt)
        results.append(result)
        if result.ok:
            break
        prev_error = result.traceback or result.error
    return results
```

- [ ] **Step 4: 네트워크 격리 보정 (subprocess 모드는 커널 레벨 차단이 불가하므로 코드 자체에 소켓 차단 훅 주입)**

`_RUNNER_TEMPLATE` 최상단에 `network: false`인 경우 소켓 생성을 막는 훅을 추가한다:

```python
_NETWORK_BLOCK_SNIPPET = '''
import socket as _socket
def _blocked(*a, **k):
    raise OSError("network disabled in sandbox")
_socket.socket = _blocked
_socket.create_connection = _blocked
'''
```

`run_code`에서 `sb_cfg.get("network", False)`가 False면 `_RUNNER_TEMPLATE` 조립 시 `_NETWORK_BLOCK_SNIPPET`을 `import pandas as pd` 앞에 삽입한다.

- [ ] **Step 5: 테스트 실행 → 통과 확인**

Run: `pytest tests/test_sandbox.py -v`
Expected: PASS (5 passed). `test_dataset_loaded_directly_by_child`는 실제 8행 데이터 기준.

- [ ] **Step 6: Commit**

```bash
git add src/isaac_eval/sandbox.py tests/test_sandbox.py
git commit -m "feat: add subprocess sandbox with timeout, network block, and retry"
```

---

### Task 6: `catalog.py` + `runners/build_catalog.py` — 카탈로그 생성

**Files:**
- Create: `src/isaac_eval/catalog.py`
- Create: `src/isaac_eval/runners/__init__.py`
- Create: `src/isaac_eval/runners/build_catalog.py`

**Interfaces:**
- Consumes: `isaac_eval.ontology.extract_structural_ontology`, `isaac_eval.gold.load_dataset`
- Produces: `build_catalog_entry(cfg: dict) -> dict` (스펙 §4.1 스키마), `save_catalog(entry: dict, out_path: str) -> None`. 결과 파일: `data/catalog/jeju_phytoplankton.json`

- [ ] **Step 1: `catalog.py` 작성**

```python
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
```

- [ ] **Step 2: `runners/build_catalog.py` 작성**

```python
from isaac_eval.config import load_env, load_settings, get
from isaac_eval.catalog import build_catalog_entry, save_catalog


def main():
    load_env()
    cfg = load_settings()
    entry = build_catalog_entry(cfg)
    out_path = f"{get(cfg, 'paths.catalog_dir')}/jeju_phytoplankton.json"
    save_catalog(entry, out_path)
    print(f"catalog written to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 실행 확인**

Run: `cd isaac-eval && python -m isaac_eval.runners.build_catalog`
Expected: `catalog written to data/catalog/jeju_phytoplankton.json` 출력, 파일에 실제 관측값(예: `1675954`, `1,675,955` 등)이 없는지 `grep -E "167595|1,675"` 로 직접 확인.

- [ ] **Step 4: Commit**

```bash
git add src/isaac_eval/catalog.py src/isaac_eval/runners/__init__.py src/isaac_eval/runners/build_catalog.py
git commit -m "feat: add catalog builder producing masked dataset ontology"
```

---

### Task 7: `llm.py` — Gemini 클라이언트 래퍼

**Files:**
- Create: `src/isaac_eval/llm.py`

**Interfaces:**
- Produces: `class LLMClient` with `generate(system: str, user: str) -> str` — `cfg["llm"]`에서 `model`, `temperature`, `top_p`, `max_tokens` 사용. `GEMINI_API_KEY` 없으면 생성자에서 명확한 `RuntimeError`.

- [ ] **Step 1: `llm.py` 작성**

```python
from __future__ import annotations
import os
from google import genai
from google.genai import types


class LLMClient:
    def __init__(self, cfg: dict):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Put it in isaac-eval/.env as GEMINI_API_KEY=... "
                "and call config.load_env() before constructing LLMClient."
            )
        self._client = genai.Client(api_key=api_key)
        self._cfg = cfg["llm"]

    def generate(self, system: str, user: str) -> str:
        response = self._client.models.generate_content(
            model=self._cfg["model"],
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self._cfg["temperature"],
                top_p=self._cfg["top_p"],
                max_output_tokens=self._cfg["max_tokens"],
            ),
        )
        return response.text
```

- [ ] **Step 2: 키가 없을 때 에러 메시지 확인 (단위 테스트 대신 수동 확인, API 키가 이미 있으므로 스킵 가능)**

Run: `cd isaac-eval && python -c "
import os
os.environ.pop('GEMINI_API_KEY', None)
from isaac_eval.llm import LLMClient
try:
    LLMClient({'llm': {'model': 'x', 'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 10}})
except RuntimeError as e:
    print('OK:', e)
"`
Expected: `OK: GEMINI_API_KEY not set...`

- [ ] **Step 3: 실제 키로 최소 호출 스모크 테스트**

Run: `cd isaac-eval && python -c "
from isaac_eval.config import load_env, load_settings
from isaac_eval.llm import LLMClient
load_env()
cfg = load_settings()
client = LLMClient(cfg)
print(client.generate('You are terse.', 'Say OK.'))
"`
Expected: 실제 Gemini 응답 텍스트 출력 (예: `OK`). 실패 시 `google-genai` 패키지 버전/모델명(`gemini-2.5-flash`)을 현재 사용 가능한 모델로 조정한다.

- [ ] **Step 4: Commit**

```bash
git add src/isaac_eval/llm.py
git commit -m "feat: add Gemini LLM client wrapper"
```

---

### Task 8: `prompt.py` — 하이브리드 프롬프트 조립 + 유출 재검사

**Files:**
- Create: `src/isaac_eval/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Produces:
  - `build_data_prompt(catalog: dict, query: str) -> str`
  - `assert_prompt_clean(prompt: str, df) -> list[str]` (내부적으로 `ontology.assert_no_leakage` 재사용)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_prompt.py
import json
from isaac_eval.catalog import build_catalog_entry
from isaac_eval.gold import load_dataset
from isaac_eval.prompt import build_data_prompt, assert_prompt_clean
from isaac_eval.config import load_settings


def test_prompt_has_no_leaked_observation_values():
    cfg = load_settings()
    catalog = build_catalog_entry(cfg)
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
    prompt = build_data_prompt(catalog, "봄철 최고 발생 월과 정점은?")
    assert assert_prompt_clean(prompt, df) == []
    assert "df" in prompt
    assert "json.dumps" in prompt
    assert "int()" in prompt  # 절삭 금지 문구 포함 확인
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL

- [ ] **Step 3: `prompt.py` 구현**

```python
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

Run: `pytest tests/test_prompt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/isaac_eval/prompt.py tests/test_prompt.py
git commit -m "feat: add hybrid prompt assembly with leakage re-check"
```

---

### Task 9: `data_agent.py` — 코드 생성 + 샌드박스 + 응답 생성 파이프라인

**Files:**
- Create: `src/isaac_eval/data_agent.py`

**Interfaces:**
- Consumes: `LLMClient.generate`, `sandbox.run_with_retry`, `prompt.build_data_prompt`
- Produces: `answer(query: str, catalog: dict, cfg: dict, llm: LLMClient) -> dict` →
  ```
  {
    "generated_code": str,          # 최종 채택된 코드(마지막 시도)
    "sandbox_results": list[SandboxResult],
    "final_ok": bool,
    "sandbox_value": dict | None,
    "final_response": str,
  }
  ```

- [ ] **Step 1: `data_agent.py` 구현**

```python
from __future__ import annotations
import re
from isaac_eval.prompt import build_data_prompt
from isaac_eval.sandbox import run_with_retry

_CODE_FENCE_RE = re.compile(r"^```(?:python)?\n|\n```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text.strip())


def _make_code_generator(llm, base_prompt: str):
    def generate(prev_error: str | None) -> str:
        if prev_error is None:
            prompt = base_prompt
        else:
            prompt = (
                f"{base_prompt}\n\n[이전 시도 오류]\n{prev_error}\n"
                "위 오류를 수정한 새 코드를 처음부터 다시 출력하라."
            )
        raw = llm.generate(
            system="너는 pandas 코드만 출력하는 데이터 분석 에이전트다.",
            user=prompt,
        )
        return _strip_code_fence(raw)

    return generate


def answer(query: str, catalog: dict, cfg: dict, llm) -> dict:
    base_prompt = build_data_prompt(catalog, query)
    generator = _make_code_generator(llm, base_prompt)
    dataset_path = cfg["paths"]["dataset"]
    sheet = cfg["paths"]["sheet"]
    results = run_with_retry(generator, dataset_path, sheet, cfg)
    final = results[-1]
    sandbox_value = final.value if final.ok else None
    if final.ok:
        response_prompt = (
            f"사용자 질문: {query}\n"
            f"계산된 결과(JSON): {sandbox_value}\n"
            "위 계산 결과에 등장하는 수치만 사용해서 한국어로 간결하게 답하라. "
            "결과에 없는 수치를 새로 만들어 내지 마라."
        )
        final_response = llm.generate(
            system="너는 계산 결과를 한국어로 요약하는 어시스턴트다.",
            user=response_prompt,
        )
    else:
        final_response = f"코드 실행에 {len(results)}회 시도 후 실패했습니다: {final.error}"
    return {
        "generated_code": generator(None) if False else None,  # placeholder overwritten below
        "sandbox_results": results,
        "final_ok": final.ok,
        "sandbox_value": sandbox_value,
        "final_response": final_response,
    }
```

**주의:** 위 초안의 `"generated_code"` 필드는 잘못된 placeholder다 — `run_with_retry`가 각 시도에 사용한 코드를 반환하지 않으므로, `run_with_retry`를 감싸는 대신 시도별 코드를 직접 추적해야 한다. 다음과 같이 수정한다:

```python
def answer(query: str, catalog: dict, cfg: dict, llm) -> dict:
    base_prompt = build_data_prompt(catalog, query)
    generated_codes: list[str] = []

    def generator(prev_error: str | None) -> str:
        if prev_error is None:
            prompt = base_prompt
        else:
            prompt = (
                f"{base_prompt}\n\n[이전 시도 오류]\n{prev_error}\n"
                "위 오류를 수정한 새 코드를 처음부터 다시 출력하라."
            )
        raw = llm.generate(system="너는 pandas 코드만 출력하는 데이터 분석 에이전트다.", user=prompt)
        code = _strip_code_fence(raw)
        generated_codes.append(code)
        return code

    dataset_path = cfg["paths"]["dataset"]
    sheet = cfg["paths"]["sheet"]
    results = run_with_retry(generator, dataset_path, sheet, cfg)
    final = results[-1]
    sandbox_value = final.value if final.ok else None
    if final.ok:
        response_prompt = (
            f"사용자 질문: {query}\n"
            f"계산된 결과(JSON): {sandbox_value}\n"
            "위 계산 결과에 등장하는 수치만 사용해서 한국어로 간결하게 답하라. "
            "결과에 없는 수치를 새로 만들어 내지 마라."
        )
        final_response = llm.generate(
            system="너는 계산 결과를 한국어로 요약하는 어시스턴트다.",
            user=response_prompt,
        )
    else:
        final_response = f"코드 실행에 {len(results)}회 시도 후 실패했습니다: {final.error}"
    return {
        "generated_code": generated_codes[-1] if generated_codes else None,
        "generated_codes": generated_codes,
        "sandbox_results": results,
        "final_ok": final.ok,
        "sandbox_value": sandbox_value,
        "final_response": final_response,
    }
```

(구현 시 첫 초안 없이 바로 이 수정본만 파일에 작성한다.)

- [ ] **Step 2: 수동 통합 확인 (실제 API 호출 1회)**

Run:
```bash
cd isaac-eval && python -c "
from isaac_eval.config import load_env, load_settings
from isaac_eval.catalog import build_catalog_entry
from isaac_eval.llm import LLMClient
from isaac_eval.data_agent import answer

load_env()
cfg = load_settings()
catalog = build_catalog_entry(cfg)
llm = LLMClient(cfg)
r = answer('봄철(3~5월) 중 식물플랑크톤이 가장 많이 발생한 달과 정점은?', catalog, cfg, llm)
print('final_ok:', r['final_ok'])
print('sandbox_value:', r['sandbox_value'])
print('response:', r['final_response'])
"
```
Expected: `sandbox_value`가 `{"max_month": "Apr.", "max_site": "J9", "max_value": 1675955}`와 일치(또는 근접). 불일치 시 이는 논문의 AA 실측값 자체이므로 코드를 고치지 말고 그대로 기록한다 (프롬프트 버그가 명백한 경우만 `prompt.py` 수정).

- [ ] **Step 3: Commit**

```bash
git add src/isaac_eval/data_agent.py
git commit -m "feat: add data agent pipeline (code gen -> sandbox -> response)"
```

---

### Task 10: `metrics.py` + 질의셋 + `runners/run_structured.py` — 실측 실행

**Files:**
- Create: `queries/structured.jeju.jsonl`
- Create: `src/isaac_eval/metrics.py`
- Create: `src/isaac_eval/runners/run_structured.py`

**Interfaces:**
- Consumes: `numeric.extract_numbers`, `numeric.flatten_facts`, `numeric.compare`, `gold.GOLD_REGISTRY`, `data_agent.answer`
- Produces:
  - `metrics.ncr(records: list[dict], policy: str) -> float`
  - `metrics.answer_accuracy(records: list[dict]) -> float`
  - `metrics.code_exec_success(records: list[dict]) -> dict` → `{"first_attempt": float, "with_retry": float, "mean_attempts": float}`
  - `runners/run_structured.py`가 `results/runs/proposed_<qid>.json` 및 `results/tables/table_structured_metrics.csv`를 생성

- [ ] **Step 1: 실제 데이터 스키마 기반 질의셋 작성** (`queries/structured.jeju.jsonl`, 한 줄에 한 건, gold_ref는 Task 4의 4개 함수를 조합해 커버)

```jsonl
{"qid": "S001", "text": "봄철(3~5월) 데이터 중 식물플랑크톤이 가장 많이 발생한 달과 가장 심각했던 정점은?", "gold_ref": "gold.spring_peak", "gold_args": {}}
{"qid": "S002", "text": "정점 J9의 전체 기간 평균 개체 수는 얼마인가?", "gold_ref": "gold.station_mean", "gold_args": {"station": "J9"}}
{"qid": "S003", "text": "4월(Apr.) 전체 정점의 개체 수 합계는 얼마인가?", "gold_ref": "gold.month_total", "gold_args": {"month": "Apr."}}
{"qid": "S004", "text": "전체 관측 기간을 통틀어 단일 정점에서 관측된 최댓값은 어느 정점에서 얼마인가?", "gold_ref": "gold.overall_max_station", "gold_args": {}}
{"qid": "S005", "text": "정점 J1의 전체 기간 평균 개체 수는 얼마인가?", "gold_ref": "gold.station_mean", "gold_args": {"station": "J1"}}
{"qid": "S006", "text": "8월(Aug.) 전체 정점의 개체 수 합계는 얼마인가?", "gold_ref": "gold.month_total", "gold_args": {"month": "Aug."}}
{"qid": "S007", "text": "정점 J12의 전체 기간 평균 개체 수는 얼마인가?", "gold_ref": "gold.station_mean", "gold_args": {"station": "J12"}}
{"qid": "S008", "text": "3월(Mar.) 전체 정점의 개체 수 합계는 얼마인가?", "gold_ref": "gold.month_total", "gold_args": {"month": "Mar."}}
{"qid": "S009", "text": "정점 J5의 전체 기간 평균 개체 수는 얼마인가?", "gold_ref": "gold.station_mean", "gold_args": {"station": "J5"}}
{"qid": "S010", "text": "11월(Nov.) 전체 정점의 개체 수 합계는 얼마인가? (결측 정점은 제외하고 계산)", "gold_ref": "gold.month_total", "gold_args": {"month": "Nov."}}
```

**주의:** `gold.month_total`은 결측치(`NaN`)를 `sum(skipna=True)` 기본 동작으로 처리하므로 S010은 그대로 `df[station_cols].sum(axis=1)` 결과와 비교하면 일치한다. 별도 분기 불필요.

- [ ] **Step 2: `metrics.py` 구현**

```python
from __future__ import annotations
from isaac_eval.numeric import extract_numbers, flatten_facts, compare


def ncr(records: list[dict], policy: str = "round") -> float:
    total = len(records)
    if total == 0:
        return 0.0
    passed = 0
    for r in records:
        if not r["final_ok"]:
            continue
        reported = extract_numbers(r["final_response"])
        facts = flatten_facts(r["sandbox_value"])
        cmp = compare(reported, facts, policy=policy)
        if cmp.pass_:
            passed += 1
    return round(passed / total, 4)


def answer_accuracy(records: list[dict]) -> float:
    total = len(records)
    if total == 0:
        return 0.0
    correct = sum(1 for r in records if r["final_ok"] and r["sandbox_value"] == r["gold_value"])
    return round(correct / total, 4)


def code_exec_success(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"first_attempt": 0.0, "with_retry": 0.0, "mean_attempts": 0.0}
    first_ok = sum(1 for r in records if r["attempts_used"] == 1 and r["final_ok"])
    with_retry_ok = sum(1 for r in records if r["final_ok"])
    mean_attempts = sum(r["attempts_used"] for r in records) / total
    return {
        "first_attempt": round(first_ok / total, 4),
        "with_retry": round(with_retry_ok / total, 4),
        "mean_attempts": round(mean_attempts, 2),
    }
```

- [ ] **Step 3: `runners/run_structured.py` 구현**

```python
from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.catalog import build_catalog_entry
from isaac_eval.llm import LLMClient
from isaac_eval.data_agent import answer
from isaac_eval.gold import GOLD_REGISTRY
from isaac_eval import metrics


def _sandbox_result_to_dict(r):
    return {"attempt": r.attempt, "ok": r.ok, "error": r.error, "elapsed_ms": r.elapsed_ms}


def main():
    load_env()
    cfg = load_settings()
    catalog = build_catalog_entry(cfg)
    llm = LLMClient(cfg)

    queries_path = get(cfg, "paths.queries_structured")
    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(queries_path, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    for q in queries:
        gold_fn = GOLD_REGISTRY[q["gold_ref"]]
        from isaac_eval.gold import load_dataset
        df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
        gold_value = gold_fn(df, **q.get("gold_args", {}))

        result = answer(q["text"], catalog, cfg, llm)
        record = {
            "qid": q["qid"],
            "query": q["text"],
            "generated_code": result["generated_code"],
            "attempts": [_sandbox_result_to_dict(r) for r in result["sandbox_results"]],
            "attempts_used": len(result["sandbox_results"]),
            "final_ok": result["final_ok"],
            "sandbox_value": result["sandbox_value"],
            "final_response": result["final_response"],
            "gold_value": gold_value,
        }
        records.append(record)
        (runs_dir / f"proposed_{q['qid']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"{q['qid']}: final_ok={record['final_ok']} sandbox={record['sandbox_value']} gold={gold_value}")

    ncr_exact = metrics.ncr(records, policy="exact")
    ncr_round = metrics.ncr(records, policy="round")
    aa = metrics.answer_accuracy(records)
    exec_stats = metrics.code_exec_success(records)

    with open(tables_dir / "table_structured_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_queries", len(records)])
        writer.writerow(["ncr_exact", ncr_exact])
        writer.writerow(["ncr_round", ncr_round])
        writer.writerow(["answer_accuracy", aa])
        writer.writerow(["exec_first_attempt", exec_stats["first_attempt"]])
        writer.writerow(["exec_with_retry", exec_stats["with_retry"]])
        writer.writerow(["exec_mean_attempts", exec_stats["mean_attempts"]])

    print("\n=== summary ===")
    print(f"NCR (exact): {ncr_exact}")
    print(f"NCR (round): {ncr_round}")
    print(f"Answer Accuracy: {aa}")
    print(f"Exec success: {exec_stats}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 실제 실행**

Run: `cd isaac-eval && python -m isaac_eval.runners.run_structured`
Expected: 10건 각각의 `final_ok`/`sandbox_value`/`gold` 로그 출력 후 `results/tables/table_structured_metrics.csv` 생성. NCR/AA 수치가 콘솔에 출력됨 — 이것이 이번 범위의 핵심 산출물이다.

- [ ] **Step 5: Commit**

```bash
git add queries/structured.jeju.jsonl src/isaac_eval/metrics.py src/isaac_eval/runners/run_structured.py
git commit -m "feat: add structured query runner producing NCR/AA/exec-success metrics"
```

---

## Explicitly Out of Scope (§9 우선순위 4 이상 — 이번 계획에서 다루지 않음)

- `complexity.py` (v1/v2), `calibrate_threshold.py`, `router.py`, `paper_agent.py`, `chunking.py`/`retrieval.py`, `baselines/*`, `run_unstructured.py`, `run_hybrid.py`, `make_tables.py`(표 3/4), 충실성 시트, Docker 샌드박스 모드.
- 이 항목들은 핵심(NCR/AA) 실측이 끝난 뒤 별도 계획으로 이어간다.
