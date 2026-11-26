"""사람 판정과 LLM 판정의 일치도(Cohen's kappa)를 계산한다.

실행: ./.venv/bin/python3 -m isaac_eval.runners.score_faithfulness_check
입력: results/annotation/faithfulness_check_sheet.xlsx (판정 열이 채워진 것), faithfulness_check_key.json
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from isaac_eval.runners.score_agreement import cohen_kappa

MAP = {"S": "supported", "U": "unsupported", "G": "generic"}

def main():
    d = Path("results/annotation")
    sheet = pd.read_excel(d / "faithfulness_check_sheet.xlsx", sheet_name="판정")
    key = json.loads((d / "faithfulness_check_key.json").read_text(encoding="utf-8"))
    pairs3, pairs2 = [], []
    for _, r in sheet.iterrows():
        h = MAP.get(str(r["판정(S/U/G)"]).strip().upper()[:1])
        if not h: continue
        l = key[r["id"]]["llm_verdict"]
        pairs3.append((h, l))
        pairs2.append(("supported" if h == "supported" else "not", "supported" if l == "supported" else "not"))
    res = {"n": len(pairs3), "three_class": cohen_kappa(pairs3), "supported_vs_not": cohen_kappa(pairs2),
           "confusion_human_x_llm": {f"{h}|{l}": sum(1 for a, b in pairs3 if (a, b) == (h, l))
                                      for h in MAP.values() for l in MAP.values()}}
    (d / "faithfulness_check_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
