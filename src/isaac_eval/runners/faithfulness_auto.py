"""경로 A 직접 요약의 자동 충실성(LLM 판정 기반 주장 지지율) 측정.

재생성 없이 기존 proposed 경로 A 실행 로그(final_response + sources)를 사용한다.
각 응답을 원자적 사실 주장으로 분해하고, 검색된 발췌만을 근거로 각 주장이
supported/unsupported/generic 인지 LLM 심판이 판정한다.
충실성 = supported / (supported + unsupported).

주의(논문에 반드시 명시): (1) LLM이 LLM 출력을 판정하므로 같은 모델 계열의
자기평가 편향이 있다. (2) n이 작다. → 전문가 5점 평가의 대체가 아니라 자동 프록시.

사용법 (Mac 터미널):
    ./.venv/bin/python3 -m isaac_eval.runners.faithfulness_auto
결과: results/tables/faithfulness_auto.csv, results/runs/faithfulness_auto.json
LLM 호출 = 경로 A 응답 수(약 10건). 무료 쿼타 내.
"""
from __future__ import annotations
import csv, json, glob, re
from pathlib import Path

from google.genai import errors as genai_errors
from isaac_eval.config import load_env, load_settings, get
from isaac_eval.llm import LLMClient
from isaac_eval.runners.resume_structured import DEFAULT_MODEL_ROTATION

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_SYS = (
    "너는 RAG 응답의 충실성(faithfulness)을 평가하는 엄격한 심판이다. "
    "오직 제공된 발췌(excerpts)만을 근거로, 응답의 각 사실 주장이 발췌로 뒷받침되는지 판정한다. "
    "발췌 밖의 배경지식은 절대 사용하지 마라. 발췌가 영어이고 응답이 한국어여도 의미로 대조하라."
)

def judge_prompt(excerpts: str, response: str) -> str:
    return (
        f"[발췌 (근거로 삼을 유일한 출처)]\n{excerpts}\n\n"
        f"[평가 대상 응답]\n{response}\n\n"
        "지시: 응답을 원자적 '사실 주장'들로 분해하라. 각 주장에 대해 다음 중 하나로 판정한다.\n"
        "- supported: 발췌 내용으로 검증 가능한 사실 주장\n"
        "- unsupported: 발췌로 검증되지 않거나 발췌와 모순되는 사실 주장\n"
        "- generic: 사실 정보가 없는 서술(도입/연결/의견 문장)\n"
        "출처를 확인할 수 없다고 정직하게 밝히는 문장은 generic으로 처리한다.\n"
        '반드시 JSON만 출력: {"claims":[{"claim":"...","verdict":"supported|unsupported|generic"}]}'
    )

def parse(text: str) -> dict:
    m = _JSON_RE.search(text or "")
    if not m: return {"claims": []}
    try: return json.loads(m.group(0))
    except json.JSONDecodeError: return {"claims": []}

def main():
    load_env()
    cfg = load_settings()
    llm = LLMClient(cfg)

    chunks = {}
    for line in open("data/corpus/chunks.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            c = json.loads(line); chunks[c.get("chunk_id")] = c.get("text", "")

    items = []
    for f in sorted(glob.glob("results/runs/unstructured_U*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("system") == "proposed" and d.get("paper_route") == "A":
            items.append(d)
    print(f"경로 A proposed 응답 {len(items)}건 판정 시작\n")

    rows = []; raw = []
    for d in items:
        excerpts = "\n\n".join(f"[{s}]\n{chunks.get(s,'(청크 없음)')}" for s in d.get("sources", []))
        resp = d.get("final_response", "")
        result = None
        for model in DEFAULT_MODEL_ROTATION:
            cfg["llm"]["model"] = model
            try:
                out = llm.generate(system=JUDGE_SYS, user=judge_prompt(excerpts, resp))
                result = parse(out); result["_judge_model"] = model; break
            except genai_errors.ClientError as e:
                if e.code in (429, 404):
                    print(f"  ({model} 불가 {e.code}, 다음)"); continue
                raise
        if result is None:
            print(f"{d['qid']}: 판정 모델 소진 — 건너뜀"); continue
        claims = result.get("claims", [])
        sup = sum(1 for c in claims if c.get("verdict") == "supported")
        uns = sum(1 for c in claims if c.get("verdict") == "unsupported")
        gen = sum(1 for c in claims if c.get("verdict") == "generic")
        denom = sup + uns
        faith = (sup / denom) if denom else 1.0
        rows.append({"qid": d["qid"], "n_supported": sup, "n_unsupported": uns,
                     "n_generic": gen, "faithfulness": round(faith, 4),
                     "judge_model": result.get("_judge_model")})
        raw.append({"qid": d["qid"], "response": resp, "sources": d.get("sources"),
                    "judge": claims, "judge_model": result.get("_judge_model")})
        print(f"{d['qid']}: supported {sup} / unsupported {uns} / generic {gen} -> 충실성 {faith:.3f}")

    tables = Path(get(cfg, "paths.results_dir")) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "faithfulness_auto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["qid","n_supported","n_unsupported","n_generic","faithfulness","judge_model"])
        w.writeheader(); w.writerows(rows)
    (Path(get(cfg, "paths.results_dir")) / "runs" / "faithfulness_auto.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        n = len(rows)
        mean_faith = sum(r["faithfulness"] for r in rows) / n
        tot_uns = sum(r["n_unsupported"] for r in rows)
        tot_claims = sum(r["n_supported"] + r["n_unsupported"] for r in rows)
        micro = (tot_claims - tot_uns) / tot_claims if tot_claims else 1.0
        print("\n" + "=" * 56)
        print(f"경로 A 자동 충실성 (n={n})")
        print(f"  질의 평균(macro): {mean_faith:.4f}")
        print(f"  주장 단위(micro): {micro:.4f}  (전체 사실 주장 {tot_claims}개 중 미지지 {tot_uns}개)")
        print(f"  저장: results/tables/faithfulness_auto.csv, results/runs/faithfulness_auto.json")

if __name__ == "__main__":
    main()
