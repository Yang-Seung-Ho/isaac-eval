"""사람 검증용 충실성 판정 시트 만들기 (R1-6, R3-3, R6 대응).

results/runs/faithfulness_auto.json 의 원자 주장 중 40개를 뽑아, 각 주장과 그 응답이 근거로 쓴
발췌 원문을 함께 엑셀 시트로 만든다. LLM 판정 결과는 시트에 넣지 않고(맹검) 별도 키 파일에 둔다.

표본: LLM이 supported가 아니라고 본 주장(드문 범주)은 전부 넣고, 나머지는 supported에서
무작위로 채운다(seed 고정). 드문 범주를 다 넣어야 일치도(kappa)가 의미를 가진다.

실행: ./.venv/bin/python3 -m isaac_eval.runners.make_faithfulness_sheet
산출: results/annotation/faithfulness_check_sheet.xlsx  (주석자에게 전달)
      results/annotation/faithfulness_check_key.json    (주석자에게 주지 말 것)
"""
from __future__ import annotations
import json, random
from pathlib import Path
import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from isaac_eval.retrieval import load_index

N_TOTAL, SEED = 40, 20261118

def main():
    data = json.loads(Path("results/runs/faithfulness_auto.json").read_text(encoding="utf-8"))
    idx = load_index("data/corpus/chunk_index")
    text = {i: ILLEGAL_CHARACTERS_RE.sub("", p.get("text", "")) for i, p in zip(idx.ids, idx.payloads)}
    claims = []
    for r in data:
        for j, c in enumerate(r["judge"]):
            claims.append({"qid": r["qid"], "n": j, "claim": c["claim"], "verdict": c["verdict"],
                           "sources": r.get("sources", [])})
    rare = [c for c in claims if c["verdict"] != "supported"]
    common = [c for c in claims if c["verdict"] == "supported"]
    rng = random.Random(SEED)
    sample = rare + rng.sample(common, max(0, N_TOTAL - len(rare)))
    rng.shuffle(sample)
    rows, key = [], {}
    for k, c in enumerate(sample, 1):
        cid = f"C{k:02d}"
        excerpts = "\n\n".join(f"[{s}]\n{text.get(s, '(발췌 없음)')}" for s in c["sources"])
        rows.append({"id": cid, "주장(claim)": ILLEGAL_CHARACTERS_RE.sub("", c["claim"]), "근거 발췌(excerpts)": excerpts,
                     "판정(S/U/G)": "", "메모": ""})
        key[cid] = {"qid": c["qid"], "claim_index": c["n"], "llm_verdict": c["verdict"]}
    out = Path("results/annotation"); out.mkdir(parents=True, exist_ok=True)
    guide = pd.DataFrame({"판정 기준": [
        "S (supported): 주장의 모든 사실 내용이 발췌에 직접 나와 있거나 발췌로부터 명백히 도출된다.",
        "U (unsupported): 주장의 사실 내용 중 하나라도 발췌에 없거나 발췌와 어긋난다.",
        "G (generic): 확인할 사실이 없는 일반적·연결 문장이다(예: '다음과 같습니다').",
        "발췌만 보고 판정하고, 배경지식으로 보충하지 마십시오. 다른 사람과 상의하지 마십시오."]})
    with pd.ExcelWriter(out / "faithfulness_check_sheet.xlsx") as w:
        guide.to_excel(w, sheet_name="안내", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="판정", index=False)
    (out / "faithfulness_check_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"표본 {len(rows)}개 (LLM 기준 비지지 {len(rare)}개 포함) -> results/annotation/faithfulness_check_sheet.xlsx")

if __name__ == "__main__":
    main()
