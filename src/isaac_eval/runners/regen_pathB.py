"""경로 B(추천) 로그 재생성 + 테스트 문서 2.2 섹션 갱신.

배경: 초기 구현의 경로 B는 요약을 '생성'했고, 이후 무생성(추천 목록)으로 바뀌었다.
그러나 러너가 이어달리기라 기존 qid 로그를 건너뛰어 U007/U010/U011/U024 로그가
낡은(요약) 상태로 남았다. 이 스크립트는 그 4건을 현재 코드의 경로 B 출력
(paper_index 상위 3편 추천, LLM 생성 호출 없음, 임베딩 검색만)으로 교체하고,
`테스트_입출력_경로별.md`의 2.2 섹션을 추천 논문 목록으로 다시 쓴다.

generate_content 쿼타(하루 20)를 쓰지 않는다 — 임베딩 API만 사용.
실행: ./.venv/bin/python3 -m isaac_eval.runners.regen_pathB
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from isaac_eval.config import load_env, load_settings, get
from isaac_eval.retrieval import load_index, search

TARGETS = ["U007", "U010", "U011", "U024"]
DOC = "테스트_입출력_경로별.md"


def _short(s, n):
    s = (s or "").replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s


def main():
    load_env()
    cfg = load_settings()
    embed_model = cfg["embedding"]["model"]
    paper_index = load_index("data/corpus/paper_index")
    runs_dir = Path(get(cfg, "paths.results_dir")) / "runs"

    queries = {}
    with open("queries/unstructured.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["qid"]] = q

    rows = []
    for qid in TARGETS:
        q = queries[qid]["text"]
        hits = search(paper_index, q, min(3, len(paper_index.ids)), embed_model)
        recs = [{"paper_id": pid, "score": round(sc, 4), "n_words": pl.get("n_words")}
                for pid, sc, pl in hits]
        sel = [r["paper_id"] for r in recs]
        titles = "\n".join(f"- {r['paper_id']} (관련도 {r['score']})" for r in recs)
        resp = ("질의의 복잡도가 임계값을 초과하여 직접 요약을 생성하지 않았습니다. "
                "대신 관련성이 높은 등록 논문을 추천합니다.\n" + titles)

        p = runs_dir / f"unstructured_{qid}.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        old = rec.get("selected_papers")
        rec["paper_route"] = "B"
        rec["degraded"] = True
        rec["selected_papers"] = sel
        rec["sources"] = []
        rec["final_response"] = resp
        rec["_note"] = "경로 B 출력 재생성(현재 무생성 추천 로직, 임베딩 검색만). 낡은 요약 로그 교체."
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"{qid}: old_selected={old} -> new_top3={sel}")
        for r in recs:
            print(f"     {r['paper_id']}  관련도 {r['score']}")
        rows.append({"qid": qid, "query": q,
                     "complexity": rec.get("complexity", {}).get("total"),
                     "recs": recs, "resp": resp})

    # --- 테스트 문서 2.2 섹션 재작성 ---
    doc_path = Path(DOC)
    if doc_path.exists():
        lines = ["## 2.2 논문 에이전트 — 경로 B — 논문 추천, %d건" % len(rows), "",
                 "경로 B는 고복잡도 단일 논문 질의에서 **직접 요약을 생성하지 않고**(LLM 생성 호출 0회) "
                 "paper_index 상위 3편을 추천으로 반환한다. 아래 ‘추천 논문(3편)’이 실제 반환값이다.", "",
                 "| qid | 질의 | 복잡도 | 추천 논문(3편, 관련도순) | 응답 |",
                 "|---|---|---|---|---|"]
        for r in rows:
            recstr = "<br>".join(f"{i+1}. {rc['paper_id']} (관련도 {rc['score']})"
                                 for i, rc in enumerate(r["recs"]))
            respcell = r["resp"].replace("\n", "<br>")
            lines.append(f"| {r['qid']} | {_short(r['query'],42)} | {r['complexity']} | {recstr} | {respcell} |")
        section = "\n".join(lines) + "\n\n"
        md = doc_path.read_text(encoding="utf-8")
        pat = re.compile(r"## 2\.2 .*?(?=\n## 2\.3 )", re.DOTALL)
        if pat.search(md):
            doc_path.write_text(pat.sub(section, md), encoding="utf-8")
            print(f"\n{DOC} 2.2 섹션 갱신 완료.")
        else:
            print(f"\n[경고] {DOC} 2.2 섹션을 못 찾아 문서는 그대로 둠(로그만 교체됨).")

    print("\n경로 B 4건 재생성 완료. (generate_content 쿼타 미사용)")


if __name__ == "__main__":
    main()
