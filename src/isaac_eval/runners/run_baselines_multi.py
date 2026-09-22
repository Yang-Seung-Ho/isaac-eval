"""도메인 확장용 범용 기준선 러너 — 제안 vs 기준선 격차를 새 도메인에서 재현.

기존 run_baselines.py 는 제주에 고정. 이 러너는 카탈로그·질의·gold 모듈을 인자로 받아
임의 도메인(예: 미생물 ASV)에 기준선(standard_rag/self_rag/crag)을 실행한다.
기준선은 마스킹·샌드박스를 쓰지 않고 '데이터 미리보기(head N행) + 검색 발췌'를 LLM에
직접 준다 — 전수 계산이 안 되므로 집계 질의에서 수치가 무너지는 것이 핵심 대조점.

- resume: 기존 로그 건너뜀. 429/404 시 대체 모델 자동 전환.
- --drop-cols: 미리보기에서 제외할 열(ASV 서열처럼 거대한 식별자 열) — 기준선에 불리하게
  만들지 않기 위한 공정성 조치.
- 종료 시 기준선 NCR(문맥기반)·AA(텍스트기반)를 즉시 출력.

실행 예:
  ./.venv/bin/python3 -m isaac_eval.runners.run_baselines_multi \
      --catalog data/catalog/microbiome_asv_bg11.json \
      --queries queries/structured.asv.jsonl \
      --gold-module gold_asv --out-prefix asv_bg11 \
      --systems standard_rag --drop-cols ASV
"""
from __future__ import annotations
import os
import argparse
import importlib
import json
from pathlib import Path

from google.genai import errors as genai_errors

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.retrieval import load_index
from isaac_eval.baselines import standard_rag, self_rag, crag, pot_pal
from isaac_eval.baselines.standard_rag import build_data_context
from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION
from isaac_eval import metrics

SYSTEMS = {"standard_rag": standard_rag, "self_rag": self_rag, "crag": crag, "pot_pal": pot_pal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold-module", default="gold_asv")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--systems", default="standard_rag",
                    help="쉼표구분: standard_rag,self_rag,crag (기본 standard_rag)")
    ap.add_argument("--drop-cols", default="",
                    help="미리보기에서 제외할 열(쉼표구분). 예: ASV")
    ap.add_argument("--limit", type=int, default=0, help="시스템별 새로 시도할 질의 수 상한(0=무제한)")
    args = ap.parse_args()

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    load_env()
    cfg = load_settings()
    meta = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    cfg["paths"]["dataset"] = meta["file_path"]
    cfg["paths"]["sheet"] = meta["sheet"]

    gold_mod = importlib.import_module(f"isaac_eval.{args.gold_module}")
    GOLD = gold_mod.GOLD_REGISTRY
    df = gold_mod.load_dataset(meta["file_path"], meta["sheet"])
    df_preview = df.drop(columns=[c for c in drop_cols if c in df.columns]) if drop_cols else df

    chunk_index = load_index("data/corpus/chunk_index")
    context_text = build_data_context(df_preview)

    from isaac_eval.llm import LLMClient
    llm = LLMClient(cfg)

    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"
    tables_dir = Path(get(cfg, "paths.results_dir")) / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(args.queries, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    _ROT = [os.environ["ISAAC_MODEL"]] if os.environ.get("ISAAC_MODEL") else DEFAULT_MODEL_ROTATION
    exhausted: set[str] = set()
    for name in names:
        module = SYSTEMS[name]
        def out_path(qid):
            return runs_dir / f"{name}_{args.out_prefix}_{qid}.json"
        new_attempts = 0
        stopped = False
        for q in queries:
            if out_path(q["qid"]).exists():
                continue
            if args.limit and new_attempts >= args.limit:
                print(f"[limit] {name}: 상한 {args.limit}건 도달.")
                break
            gold_value = GOLD[q["gold_ref"]](df, **q.get("gold_args", {}))
            result = None
            used_model = None
            for model in _ROT:
                if model in exhausted:
                    continue
                cfg["llm"]["model"] = model
                try:
                    result = module.answer(q["text"], cfg, llm, chunk_index=chunk_index, df=df_preview)
                    used_model = model
                    break
                except genai_errors.ClientError as e:
                    if e.code in (429, 404):
                        print(f"  ({model} unavailable: {e.code} → 다음 모델)")
                        exhausted.add(model)
                        continue
                    raise
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    if "503" in msg or "unavailable" in msg or "overloaded" in msg:
                        print(f"  ({model} 과부하 503 → 다음 모델)")
                        continue
                    raise
            if result is None:
                print(f"{name}/{q['qid']}: 후보 모델 전부 소진 — 중단(resume 가능).")
                stopped = True
                break
            new_attempts += 1
            record = {
                "qid": q["qid"], "system": name, "kind": "structured", "query": q["text"],
                "final_response": result["final_response"], "sandbox_value": None,
                "final_ok": True, "attempts_used": 0, "gold_value": gold_value,
                "context_text": context_text, "sources": result.get("sources", []),
                "config_snapshot": {"model": used_model, "temperature": cfg["llm"]["temperature"]},
            }
            out_path(q["qid"]).write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"{name}/{q['qid']} [{used_model}] done")

        # 집계(디스크의 완료분 전체)
        recs = []
        for q in queries:
            p = out_path(q["qid"])
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                r["gold_value"] = GOLD[q["gold_ref"]](df, **q.get("gold_args", {}))
                recs.append(r)
        if recs:
            n = len(recs)
            ncr_ctx = round(sum(metrics.ncr_context_based(r, policy="round").pass_ for r in recs) / n, 4)
            aa_txt = metrics.answer_accuracy_from_text(recs, policy="round")
            print(f"\n=== {name} ({args.out_prefix}) | 완료 {n}/{len(queries)} ===")
            print(f"NCR(context-based): {ncr_ctx}")
            print(f"AA(text-based):     {aa_txt}")
            (tables_dir / f"table_baseline_{name}_{args.out_prefix}.csv").write_text(
                "metric,value\n" f"n,{n}\nncr_context_based,{ncr_ctx}\naa_text_based,{aa_txt}\n", encoding="utf-8")
        if stopped:
            print(f"[안내] {name} 미완료 — 쿼타 회복 후 같은 명령으로 이어서.")


if __name__ == "__main__":
    main()
