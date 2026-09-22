"""4조건 카탈로그·마스킹 어블레이션 러너 (단일 모델).

조건 (모두 동일 LLM·샌드박스·데이터셋):
  standard_rag       카탈로그 ✗ · 실제수치 미리보기 ✓ · 샌드박스 ✗
  pot_pal            카탈로그 ✗ · 실제수치 미리보기 ✓ · 샌드박스 ✓   (순수 PoT/PAL)
  proposed_unmasked  카탈로그 ✓ · 실제수치 미리보기 ✓ · 샌드박스 ✓   (제안 − 마스킹)
  proposed           카탈로그 ✓ · 실제수치 마스킹      · 샌드박스 ✓   (제안)

리뷰 NEW-2 대응: pot_pal↔proposed_unmasked 차이 = 카탈로그 고유 기여,
proposed_unmasked↔proposed 차이 = 마스킹 고유 기여. resume 지원. ISAAC_MODEL 단일.

사용:
  export ISAAC_MODEL=gemini-3.5-flash
  ./.venv/bin/python3 -m isaac_eval.runners.run_catalog_ablation \
    --catalog data/catalog/jeju_phytoplankton.json --queries queries/structured.jeju.jsonl \
    --gold-module gold --out-prefix jeju_abl [--limit N] [--keep-first-col] \
    [--systems standard_rag,pot_pal,proposed_unmasked,proposed]
"""
from __future__ import annotations
import argparse, importlib, json, os, time
from pathlib import Path

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.baselines import standard_rag, pot_pal, proposed_unmasked
from isaac_eval.data_agent import answer as proposed_answer
from isaac_eval import metrics


def build_catalog(data_name, data_dictionary, df):
    from isaac_eval.ontology import extract_structural_ontology
    return {"data_name": data_name, "data_dictionary": data_dictionary,
            "structural_ontology": extract_structural_ontology(df)}


def _classify(e):
    m = str(e).lower()
    if "429" in m or "resource_exhausted" in m or "quota" in m or "rate limit" in m or "credits are depleted" in m:
        return "quota"
    if ("503" in m or "unavailable" in m or "overloaded" in m or "high demand" in m
            or "connection reset" in m or "readerror" in m or "read error" in m
            or "connecterror" in m or "connect error" in m or "timeout" in m or "timed out" in m
            or "peer" in m or "remoteprotocol" in m or "protocolerror" in m or "incompleteread" in m):
        return "transient"
    return "other"


def run_system(name, q, catalog, cfg, llm, df, chunk_index):
    if name == "standard_rag":
        r = standard_rag.answer(q["text"], cfg, llm, chunk_index=None, df=df)
    elif name == "pot_pal":
        r = pot_pal.answer(q["text"], cfg, llm, chunk_index=None, df=df)
    elif name == "proposed_unmasked":
        r = proposed_unmasked.answer(q["text"], catalog, cfg, llm, df=df)
    elif name == "proposed":
        r = proposed_answer(q["text"], catalog, cfg, llm)
    else:
        raise ValueError(name)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold-module", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-first-col", action="store_true")
    ap.add_argument("--systems", default="standard_rag,pot_pal,proposed_unmasked,proposed")
    args = ap.parse_args()

    load_env(); cfg = load_settings()
    meta = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    cfg["paths"]["dataset"] = meta["file_path"]; cfg["paths"]["sheet"] = meta["sheet"]
    if not args.keep_first_col:
        cfg["paths"]["rename_first_col_to"] = None

    gold_mod = importlib.import_module(f"isaac_eval.{args.gold_module}")
    GOLD = gold_mod.GOLD_REGISTRY
    df = gold_mod.load_dataset(meta["file_path"], meta["sheet"])
    catalog = build_catalog(meta["data_name"], meta["data_dictionary"], df)

    results_dir = Path(get(cfg, "paths.results_dir"))
    runs_dir = results_dir / "runs"; tables_dir = results_dir / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True); tables_dir.mkdir(parents=True, exist_ok=True)

    with open(args.queries, encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]

    from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION
    ROT = [os.environ["ISAAC_MODEL"]] if os.environ.get("ISAAC_MODEL") else DEFAULT_MODEL_ROTATION
    from isaac_eval.llm import LLMClient
    llm = LLMClient(cfg)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    attempts = 0
    for name in systems:
        exhausted = set(); stopped = False
        for q in queries:
            out = runs_dir / f"{name}_{args.out_prefix}_{q['qid']}.json"
            if out.exists():
                continue
            if args.limit and attempts >= args.limit:
                print(f"[limit] {args.limit}건 도달 — 중단(resume 가능)"); stopped = True; break
            gold_value = GOLD[q["gold_ref"]](df, **q.get("gold_args", {}))
            result = None; used = None
            for model in ROT:
                if model in exhausted:
                    continue
                cfg["llm"]["model"] = model
                got = False
                for attempt in range(1, 4):
                    try:
                        result = run_system(name, q, catalog, cfg, llm, df, None); used = model; got = True; break
                    except Exception as e:  # noqa
                        k = _classify(e)
                        detail = " ".join(str(e).split())[:300]
                        if k == "quota":
                            print(f"  ({name}/{q['qid']} {model} 한도 → 다음 모델) :: {detail}"); exhausted.add(model); break
                        if k == "transient":
                            print(f"  ({name}/{q['qid']} {model} 일시오류 재시도 {attempt}/3 :: {detail}"); time.sleep(6); continue
                        raise
                if got:
                    break
            if result is None:
                print(f"[중단] {name}/{q['qid']} 모델 소진 — 쿼타 회복 후 같은 명령으로 이어서."); stopped = True; break
            attempts += 1
            rec = {"qid": q["qid"], "system": name, "query": q["text"],
                   "final_response": result["final_response"], "sandbox_value": result.get("sandbox_value"),
                   "final_ok": result.get("final_ok", True), "gold_value": gold_value,
                   "context_text": "", "config_snapshot": {"model": used}}
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"{name}/{q['qid']} [{used}] ok")

        # 집계
        recs = []
        for q in queries:
            p = runs_dir / f"{name}_{args.out_prefix}_{q['qid']}.json"
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                r["gold_value"] = GOLD[q["gold_ref"]](df, **q.get("gold_args", {}))
                recs.append(r)
        if recs:
            n = len(recs)
            aa_text = metrics.answer_accuracy_from_text(recs, policy="round")
            aa_dict = metrics.answer_accuracy(recs, policy="round")
            print(f"\n=== {name} ({args.out_prefix}) | {n}/{len(queries)} ===")
            print(f"AA(text): {aa_text}   AA(dict/sandbox): {aa_dict}")
            (tables_dir / f"table_ablation_{name}_{args.out_prefix}.csv").write_text(
                f"metric,value\nn,{n}\naa_text,{aa_text}\naa_dict,{aa_dict}\n", encoding="utf-8")
        if stopped:
            print(f"[안내] {name} 미완료 — 쿼타 회복 후 같은 명령으로 이어서.")


if __name__ == "__main__":
    main()
