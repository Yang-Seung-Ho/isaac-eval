"""주석자 2인의 라벨로 Cohen's kappa를 산출한다 (논문 5.2절).

results/annotation/labels_annotator_a.csv 와 labels_annotator_b.csv 를 읽는다.
scikit-learn 의존을 피하기 위해 kappa를 직접 계산한다 — 정의가 단순하고,
의존성이 줄어들면 재현이 쉬워진다.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from isaac_eval.config import load_settings, get


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict:
    """pairs: [(주석자A 라벨, 주석자B 라벨), ...]"""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": None, "observed_agreement": None}

    observed = sum(1 for a, b in pairs if a == b) / n

    labels = sorted({lab for pair in pairs for lab in pair})
    expected = 0.0
    for lab in labels:
        pa = sum(1 for a, _ in pairs if a == lab) / n
        pb = sum(1 for _, b in pairs if b == lab) / n
        expected += pa * pb

    if expected == 1.0:
        # 두 주석자가 모든 항목에 같은 하나의 라벨만 사용한 경우 kappa는 정의되지 않는다.
        return {"n": n, "kappa": None, "observed_agreement": round(observed, 4),
                "note": "kappa undefined: no label variance"}

    kappa = (observed - expected) / (1 - expected)
    return {
        "n": n,
        "kappa": round(kappa, 4),
        "observed_agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "labels": labels,
    }


def read_labels(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "qid":
                continue
            qid = row[0].strip()
            out[qid] = {
                "route": (row[2] if len(row) > 2 else "").strip(),
                "paper_route": (row[3] if len(row) > 3 else "").strip(),
                "complexity": (row[4] if len(row) > 4 else "").strip(),
            }
    return out


def main():
    cfg = load_settings()
    ann_dir = Path(get(cfg, "paths.results_dir")) / "annotation"
    path_a = ann_dir / "labels_annotator_a.csv"
    path_b = ann_dir / "labels_annotator_b.csv"

    missing = [p for p in (path_a, path_b) if not p.exists()]
    if missing:
        raise SystemExit(
            f"주석 파일이 없다: {', '.join(str(p) for p in missing)}\n"
            "먼저 'make annotation-sheet'로 시트를 만들고 두 주석자가 각자 채워야 한다."
        )

    a, b = read_labels(path_a), read_labels(path_b)
    shared = sorted(set(a) & set(b))

    report = {}
    for field in ("route", "paper_route", "complexity"):
        pairs = [(a[q][field], b[q][field]) for q in shared
                 if a[q][field] and b[q][field]]
        report[field] = cohen_kappa(pairs)

    out_path = ann_dir / "agreement.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    tables_dir = Path(get(cfg, "paths.results_dir")) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "annotator_agreement.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "n", "observed_agreement", "cohen_kappa"])
        for field, r in report.items():
            writer.writerow([field, r["n"], r["observed_agreement"], r["kappa"]])

    print("=== 주석자 간 일치도 (협의 이전) ===")
    for field, r in report.items():
        if r["n"] == 0:
            print(f"  {field:<12} 라벨이 비어 있어 산출 불가")
            continue
        print(f"  {field:<12} n={r['n']:<3} 관측 일치={r['observed_agreement']}  kappa={r['kappa']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
