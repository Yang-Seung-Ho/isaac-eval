"""C-5: 데이터셋 선택 정확도.

여러 데이터셋이 등록된 상태에서 질의가 올바른 데이터셋을 유사도로 선택하는지 측정한다.
각 데이터셋의 서술자(카탈로그의 data_name + 열 이름 + 데이터 사전)를 임베딩하고, 각
질의를 임베딩해 코사인 최댓값 데이터셋을 예측으로 삼아 gold(질의의 소속 데이터셋)와
비교한다. ASV 3종(BG11/NIES/OHM)은 이름이 유사해 미디어 미언급 질의에서 변별이 어려운
'최난조건'이며, 그 실패 양상 자체가 8.3절 한계의 실증이 된다.

임베딩 API만 사용(generate_content 아님). 실행:
  ./.venv/bin/python3 -m isaac_eval.runners.run_dataset_selection
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from isaac_eval.config import load_env, load_settings, get
from isaac_eval.retrieval import embed_texts

# gold 데이터셋 -> (카탈로그 JSON, 질의 jsonl)
DATASETS = {
    "jeju":     ("data/catalog/jeju_phytoplankton.json", "queries/structured.jeju.jsonl"),
    "asv_bg11": ("data/catalog/microbiome_asv_bg11.json", "queries/structured.asv.jsonl"),
    "asv_nies": ("data/catalog/microbiome_asv_nies.json", "queries/structured.asv.nies.jsonl"),
    "asv_ohm":  ("data/catalog/microbiome_asv_ohm.json",  "queries/structured.asv.ohm.jsonl"),
}


def _descriptor(cat: dict) -> str:
    dd = cat.get("data_dictionary", {})
    cols = ", ".join(dd.keys())
    desc = " ".join(str(v) for v in dd.values())
    return f"{cat.get('data_name','')}. 열: {cols}. {desc}"


def main():
    load_env()
    cfg = load_settings()
    model = cfg["embedding"]["model"]

    names, descs, qfiles = [], [], {}
    for name, (cat_path, qf) in DATASETS.items():
        cat = json.loads(Path(cat_path).read_text(encoding="utf-8"))
        names.append(name)
        descs.append(_descriptor(cat))
        qfiles[name] = qf

    dvecs = embed_texts(descs, model=model)  # (D, dim), L2 정규화됨

    records = []
    for gold in names:
        qs = [json.loads(l) for l in open(qfiles[gold], encoding="utf-8") if l.strip()]
        qvecs = embed_texts([q["text"] for q in qs], model=model)
        sims = qvecs @ dvecs.T  # (n, D)
        for q, srow in zip(qs, sims):
            pred = names[int(np.argmax(srow))]
            records.append({"qid": q["qid"], "gold": gold, "pred": pred,
                            "correct": pred == gold,
                            "scores": {names[i]: round(float(srow[i]), 4) for i in range(len(names))}})

    n = len(records); hit = sum(r["correct"] for r in records)
    print(f"=== 데이터셋 선택 정확도: {hit}/{n} = {round(hit/n,4)} ===")
    print("\n[gold별 정확도]")
    for name in names:
        sub = [r for r in records if r["gold"] == name]
        h = sum(r["correct"] for r in sub)
        print(f"  {name:9} {h}/{len(sub)} = {round(h/len(sub),3) if sub else 0}")
    print("\n[혼동행렬 gold→pred]")
    from collections import Counter
    for name in names:
        c = Counter(r["pred"] for r in records if r["gold"] == name)
        print(f"  {name:9} -> {dict(c)}")

    results_dir = Path(get(cfg, "paths.results_dir"))
    (results_dir / "runs").mkdir(parents=True, exist_ok=True)
    (results_dir / "runs" / "dataset_selection.json").write_text(
        json.dumps({"accuracy": hit/n, "n": n, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    with open(results_dir / "tables" / "table_dataset_selection.csv", "w", encoding="utf-8") as f:
        f.write("metric,value\n"); f.write(f"n,{n}\naccuracy,{round(hit/n,4)}\n")
        for name in names:
            sub = [r for r in records if r["gold"] == name]
            f.write(f"acc_{name},{round(sum(r['correct'] for r in sub)/len(sub),4)}\n")
    print(f"\n저장: results/runs/dataset_selection.json, results/tables/table_dataset_selection.csv")


if __name__ == "__main__":
    main()
