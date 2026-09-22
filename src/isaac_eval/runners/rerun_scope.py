"""질의 범위 판정만 단일 모델로 재측정한다 (경로 선택 정확도 확정값 산출).

기존 실행 레코드의 복잡도 점수와 응답은 그대로 두고, LLM 범위 판정만 지정된 단일
모델로 다시 수행한다. 질의당 LLM 호출이 1회이므로 모델 혼용 없이 30건을 측정할 수
있다. 결과는 scope_llm_multi_single 필드에 기록된다.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.llm import LLMClient
from isaac_eval.paper_agent import llm_multi_paper


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash"

    load_env()
    cfg = load_settings()
    cfg["llm"]["model"] = model
    llm = LLMClient(cfg)

    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"
    paths = sorted(runs_dir.glob("unstructured_*.json"))

    done = skipped = 0
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("scope_judge_model") == model:
            skipped += 1
            continue
        try:
            verdict = llm_multi_paper(record["query"], llm)
        except genai_errors.ClientError as e:
            if e.code in (429, 404):
                print(f"{record['qid']}: quota/model unavailable ({e.code}) — stopping at {done} done")
                break
            raise
        record["scope_llm_multi_single"] = verdict
        record["scope_judge_model"] = model
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        done += 1
        print(f"{record['qid']}: multi_paper={verdict} (gold={record['gold_paper_route']})")

    print(f"\nrescored {done} records with {model} ({skipped} already done, "
          f"{len(paths) - done - skipped} remaining)")


if __name__ == "__main__":
    main()
