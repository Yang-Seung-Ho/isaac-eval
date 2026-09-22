"""저장된 비정형 실행 레코드의 경로를 논문 3.5.2절 정의로 재판정한다.

논문 정의: 다중 논문 → C, 단일 논문에서 복잡도 < 임계값 → A(직접 요약),
          복잡도 >= 임계값 → B(논문 추천 = 우회).

복잡도 점수와 범위 판정은 이미 레코드에 저장돼 있으므로 LLM을 재호출하지 않는다.
생성된 응답 텍스트는 이전 정의로 만들어진 것이므로 건드리지 않고,
경로 판정 필드만 갱신한다 (응답 본문이 필요한 지표는 재실행 후 산출할 것).
"""
from __future__ import annotations
import json
from pathlib import Path

from isaac_eval.config import load_settings, get


def main():
    cfg = load_settings()
    threshold = get(cfg, "complexity.threshold")
    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"

    changed = 0
    for path in sorted(runs_dir.glob("unstructured_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        is_multi = r["scope_kw_multi"] or r["scope_llm_multi"]
        if is_multi:
            route = "C"
        elif r["complexity"]["total"] >= threshold:
            route = "B"
        else:
            route = "A"

        if route != r["paper_route"]:
            changed += 1
        r["paper_route"] = route
        r["degraded"] = route == "B"
        r["route_definition"] = "paper-3.5.2: low<threshold->A, high>=threshold->B, multi->C"
        path.write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"rejudged {len(list(runs_dir.glob('unstructured_*.json')))} records "
          f"({changed} routes changed), threshold={threshold}")


if __name__ == "__main__":
    main()
