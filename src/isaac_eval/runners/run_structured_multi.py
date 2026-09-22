"""도메인 확장용 범용 정형(structured) 러너 — 이어달리기(resume) 지원.

기존 run_structured.py 는 제주 한 데이터셋에 고정. 이 러너는 동일 파이프라인
(data_agent.answer + 샌드박스 + NCR/AA)을 재사용하되 데이터셋·카탈로그·질의·gold
모듈을 인자로 받는다. 제주 코드/결과는 건드리지 않는다.

무료 티어(하루 20 요청) 대응:
- 이미 완료된 질의(proposed_{prefix}_{qid}.json 존재)는 건너뛴다 → 여러 날 나눠 실행 가능.
- 실행 중 쿼타(429/RESOURCE_EXHAUSTED)에 걸리면 크래시 없이 그 질의만 건너뛰고 정상 종료.
- --limit N: 이번 실행에서 '새로' 시도할 질의 수 상한(하루 20회 한도 안에 여유 두기).
- 집계는 디스크의 완료 파일 전체로 하므로 부분 실행에서도 지금까지의 지표가 나온다.

사용 예 (맥 .venv):
  ./.venv/bin/python3 -m isaac_eval.runners.run_structured_multi \
      --catalog data/catalog/microbiome_asv.json \
      --queries queries/structured.asv.jsonl \
      --gold-module gold_asv --out-prefix asv_bg11 --limit 8
"""
from __future__ import annotations
import os
import argparse
import csv
import importlib
import json
from pathlib import Path

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.ontology import extract_structural_ontology
from isaac_eval.data_agent import answer
from isaac_eval import metrics

MAX_SAMPLE_STR = 40  # 긴 문자열 표본(예: ASV DNA 서열)은 프롬프트에서 잘라낸다


def _trim_long_samples(ontology: dict, limit: int = MAX_SAMPLE_STR) -> dict:
    for spec in ontology["schema"].values():
        s = spec.get("samples")
        if isinstance(s, list):
            spec["samples"] = [
                (x[:limit] + "…") if isinstance(x, str) and len(x) > limit else x
                for x in s
            ]
    return ontology


def build_catalog(data_name: str, data_dictionary: dict, df) -> dict:
    ontology = _trim_long_samples(extract_structural_ontology(df))
    return {
        "data_name": data_name,
        "data_dictionary": {k: v for k, v in data_dictionary.items() if k in df.columns},
        "structural_ontology": ontology,
    }


def _sandbox_result_to_dict(r):
    return {"attempt": r.attempt, "ok": r.ok, "error": r.error, "elapsed_ms": r.elapsed_ms}


def _classify_error(exc: Exception) -> str:
    """쿼타(429) vs 일시적 서버 과부하(503/500/UNAVAILABLE) vs 기타 구분."""
    msg = str(exc).lower()
    if "resource_exhausted" in msg or "429" in msg or "quota" in msg:
        return "quota"
    if ("503" in msg or "500" in msg or "unavailable" in msg
            or "high demand" in msg or "overloaded" in msg or "try again" in msg):
        return "transient"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="data_name/file_path/sheet/data_dictionary JSON")
    ap.add_argument("--queries", required=True, help="structured 질의 jsonl")
    ap.add_argument("--gold-module", default="gold_asv", help="isaac_eval 하위 gold 모듈명")
    ap.add_argument("--out-prefix", required=True, help="산출물 파일 접두사 (예: asv_bg11)")
    ap.add_argument("--limit", type=int, default=0, help="이번 실행에서 새로 시도할 질의 수 상한 (0=무제한)")
    ap.add_argument("--keep-first-col", action="store_true", help="제주식 첫 열 rename 유지")
    args = ap.parse_args()

    load_env()
    cfg = load_settings()

    meta = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    dataset = meta["file_path"]
    sheet = meta["sheet"]
    cfg["paths"]["dataset"] = dataset
    cfg["paths"]["sheet"] = sheet
    if not args.keep_first_col:
        cfg["paths"]["rename_first_col_to"] = None

    gold_mod = importlib.import_module(f"isaac_eval.{args.gold_module}")
    GOLD = gold_mod.GOLD_REGISTRY
    df = gold_mod.load_dataset(dataset, sheet)
    catalog = build_catalog(meta["data_name"], meta["data_dictionary"], df)

    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"
    tables_dir = results_dir / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(args.queries, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    def out_path(qid):
        return runs_dir / f"proposed_{args.out_prefix}_{qid}.json"

    # gold 는 항상 갱신(코드/데이터 변경 반영). LLM 은 미완료 질의만 호출.
    done = [q for q in queries if out_path(q["qid"]).exists()]
    todo = [q for q in queries if not out_path(q["qid"]).exists()]
    print(f"[resume] 완료 {len(done)}/{len(queries)}, 남음 {len(todo)}")

    from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION
    _ROT = [os.environ["ISAAC_MODEL"]] if os.environ.get("ISAAC_MODEL") else DEFAULT_MODEL_ROTATION
    llm = None
    new_attempts = 0
    stop_reason = None
    exhausted: set[str] = set()   # 일일 쿼타 소진 모델(이 실행 동안 제외)
    for q in todo:
        if args.limit and new_attempts >= args.limit:
            print(f"[limit] 이번 실행 상한 {args.limit}건 도달 — 나머지는 다음 실행에서.")
            break
        if llm is None:
            from isaac_eval.llm import LLMClient  # 지연 임포트(호출 전까지 키 불필요)
            llm = LLMClient(cfg)
        # 503(과부하)·429(쿼타) 시 대체 모델로 자동 전환. NCR/AA는 샌드박스 산출이라
        # 모델 혼용에 불변(정형 데이터도 flash+flash-lite 혼용). 사용 모델은 로그에 기록.
        result = None
        used_model = None
        for model in _ROT:
            if model in exhausted:
                continue
            cfg["llm"]["model"] = model
            try:
                result = answer(q["text"], catalog, cfg, llm)
                used_model = model
                break
            except Exception as e:  # noqa: BLE001
                kind = _classify_error(e)
                if kind == "quota":
                    print(f"  ({model} 일일 한도 도달 → 다음 모델 시도)")
                    exhausted.add(model)
                    continue
                if kind == "transient":
                    print(f"  ({model} 과부하 503 → 다음 모델 시도)")
                    continue
                raise
        if result is None:
            stop_reason = "exhausted"
            print(f"[중단] {q['qid']}: 후보 모델이 모두 쿼타/과부하 — 여기서 중단. "
                  f"쿼타 회복(한국시간 16:00) 후 같은 명령으로 이어서 실행.")
            break
        new_attempts += 1
        gold_value = GOLD[q["gold_ref"]](df, **q.get("gold_args", {}))
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
            "config_snapshot": {
                "model": used_model,
                "temperature": cfg["llm"]["temperature"],
                "seed": cfg["llm"]["seed"],
                "dataset": dataset,
                "sheet": sheet,
            },
        }
        out_path(q["qid"]).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"{q['qid']}: final_ok={record['final_ok']} sandbox={record['sandbox_value']} gold={gold_value}")

    # --- 집계: 디스크의 완료 파일 전체(이전 날 포함) + gold 최신 재계산 ---
    records = []
    for q in queries:
        p = out_path(q["qid"])
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["gold_value"] = GOLD[q["gold_ref"]](df, **q.get("gold_args", {}))
        records.append(rec)

    n_done = len(records)
    ncr_exact = metrics.ncr(records, policy="exact")
    ncr_round = metrics.ncr(records, policy="round")
    aa = metrics.answer_accuracy(records, policy="round")
    exec_stats = metrics.code_exec_success(records)

    out_csv = tables_dir / f"table_structured_metrics_{args.out_prefix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_queries_total", len(queries)])
        writer.writerow(["n_queries_done", n_done])
        writer.writerow(["ncr_exact", ncr_exact])
        writer.writerow(["ncr_round", ncr_round])
        writer.writerow(["answer_accuracy", aa])
        writer.writerow(["exec_first_attempt", exec_stats["first_attempt"]])
        writer.writerow(["exec_with_retry", exec_stats["with_retry"]])
        writer.writerow(["exec_mean_attempts", exec_stats["mean_attempts"]])

    print(f"\n=== summary ({args.out_prefix}) | 완료 {n_done}/{len(queries)} ===")
    print(f"NCR (exact): {ncr_exact}")
    print(f"NCR (round): {ncr_round}")
    print(f"Answer Accuracy: {aa}")
    print(f"Exec success: {exec_stats}")
    print(f"table -> {out_csv}")
    if llm is not None:
        u = llm.usage
        per = (lambda k: round(u[k]/u["calls"],1) if u["calls"] else 0)
        print(f"[usage] LLM calls={u['calls']} total_tokens={u['total_tokens']} "
              f"latency_s={round(u['latency_s'],1)} | per-call tokens={per('total_tokens')} latency_s≈{per('latency_s')}")
        with open(tables_dir / f"table_usage_{args.out_prefix}.csv","w",encoding="utf-8") as f:
            f.write("metric,value\n")
            for k,v in u.items(): f.write(f"{k},{round(v,3) if isinstance(v,float) else v}\n")

    if stop_reason or n_done < len(queries):
        print("\n[안내] 아직 미완료 질의가 있습니다. 다음 무료 쿼타 회복 후 같은 명령을 다시 실행하면 이어서 채웁니다.")


if __name__ == "__main__":
    main()
