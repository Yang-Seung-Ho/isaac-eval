"""주석자 2인 독립 라벨링용 시트를 생성한다 (논문 5.2절).

정답 라벨을 노출하지 않은 질의 70건 시트를 만든다. 두 주석자가 각자 사본에
라벨을 채운 뒤 score_agreement.py로 Cohen's kappa를 산출한다.

시트는 질의 순서를 섞어 정형/비정형/복합이 블록으로 몰리지 않게 한다. 블록으로
제시되면 주석자가 앞 항목의 라벨을 관성적으로 반복하기 쉽기 때문이다.
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

from isaac_eval.config import load_settings, get

INSTRUCTIONS = [
    ["# 라우팅·경로 라벨링 시트"],
    ["# 두 주석자가 서로 상의하지 않고 각자 채운 뒤 파일명에 이름을 붙여 저장하십시오."],
    ["#"],
    ["# route 열: data / literature / hybrid / chat 중 하나"],
    ["#   data       = 등록된 수치 데이터셋에서 계산·집계·조회로 답할 수 있는 질의"],
    ["#   literature = 등록된 논문 본문의 내용으로 답해야 하는 질의"],
    ["#   hybrid     = 데이터 계산과 문헌 해석이 모두 필요한 질의"],
    ["#   chat       = 위 어디에도 해당하지 않는 일반 대화"],
    ["#"],
    ["# paper_route 열: route가 literature일 때만 채웁니다. A / B / C 중 하나"],
    ["#   A = 단일 논문으로 직접 요약이 가능한 낮은 복잡도의 질의"],
    ["#   B = 단일 논문 질의이나 복잡도가 높아 요약보다 논문 추천이 적절한 질의"],
    ["#   C = 여러 논문을 아울러야 답할 수 있는 질의"],
    ["#"],
    ["# complexity 열: route가 literature일 때만 채웁니다. low / high 중 하나"],
    ["#"],
]


def main():
    cfg = load_settings()
    out_dir = Path(get(cfg, "paths.results_dir")) / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(get(cfg, "paths.queries_structured"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                rows.append([q["qid"], q["text"]])
    for path in ("queries/unstructured.jsonl", "queries/hybrid.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    rows.append([q["qid"], q["text"]])

    random.Random(get(cfg, "llm.seed")).shuffle(rows)

    for name in ("annotator_a", "annotator_b"):
        out_path = out_dir / f"labels_{name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for line in INSTRUCTIONS:
                writer.writerow(line)
            writer.writerow(["qid", "query", "route", "paper_route", "complexity"])
            for qid, text in rows:
                writer.writerow([qid, text, "", "", ""])
        print(f"wrote {out_path} ({len(rows)} queries, labels blank)")

    print("\n두 주석자가 각자 채운 뒤:")
    print("  make agreement   # Cohen's kappa 산출")


if __name__ == "__main__":
    main()
