"""저장된 실행 레코드의 gold_value를 현재 gold.py 기준으로 다시 계산해 갱신한다.

참조 구현이 수정되었을 때 LLM을 재호출하지 않고 AA만 재채점하기 위한 도구다.
샌드박스 결과·생성 코드·응답은 건드리지 않는다.
"""
from __future__ import annotations
import json
from pathlib import Path

from isaac_eval.config import load_settings, get
from isaac_eval.gold import GOLD_REGISTRY, load_dataset


def main():
    cfg = load_settings()
    df = load_dataset(cfg["paths"]["dataset"], cfg["paths"]["sheet"])
    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"

    with open(get(cfg, "paths.queries_structured"), encoding="utf-8") as f:
        queries = {json.loads(line)["qid"]: json.loads(line) for line in f if line.strip()}

    for path in sorted(runs_dir.glob("proposed_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        q = queries[record["qid"]]
        new_gold = GOLD_REGISTRY[q["gold_ref"]](df, **q.get("gold_args", {}))
        if new_gold != record["gold_value"]:
            print(f"{record['qid']}: {record['gold_value']} -> {new_gold}")
            record["gold_value"] = new_gold
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("gold values refreshed")


if __name__ == "__main__":
    main()
