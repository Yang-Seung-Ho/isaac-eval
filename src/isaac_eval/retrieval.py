"""임베딩 인덱스 구축·검색.

FAISS 대신 numpy 코사인 유사도를 쓴다. 코퍼스가 수백 청크 규모라 완전 탐색으로
충분하고, 근사 인덱스의 재현성 변수를 없앨 수 있다 (5.7절).
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Index:
    ids: list[str]
    vectors: np.ndarray          # (n, dim), L2 정규화됨
    payloads: list[dict]


def _client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (call config.load_env() first)")
    return genai.Client(api_key=api_key)


def embed_texts(texts: list[str], model: str, batch_size: int = 4, verbose: bool = False) -> np.ndarray:
    import time

    client = _client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(1, 7):
            try:
                response = client.models.embed_content(model=model, contents=batch)
                vectors.extend(e.values for e in response.embeddings)
                break
            except Exception as e:  # rate limits surface as ClientError 429
                if getattr(e, "code", None) != 429 or attempt == 6:
                    raise
                time.sleep(min(8 * attempt, 40))
        if verbose:
            print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def build_index(records: list[dict], text_key: str, id_key: str, model: str, verbose: bool = False) -> Index:
    texts = [r[text_key] for r in records]
    vectors = embed_texts(texts, model=model, verbose=verbose)
    return Index(ids=[r[id_key] for r in records], vectors=vectors, payloads=records)


def save_index(index: Index, dir_path: str | Path) -> None:
    directory = Path(dir_path)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", index.vectors)
    with open(directory / "payloads.jsonl", "w", encoding="utf-8") as f:
        for payload in index.payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    (directory / "ids.json").write_text(json.dumps(index.ids, ensure_ascii=False), encoding="utf-8")


def load_index(dir_path: str | Path) -> Index:
    directory = Path(dir_path)
    vectors = np.load(directory / "embeddings.npy")
    ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    payloads = [json.loads(line) for line in (directory / "payloads.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return Index(ids=ids, vectors=vectors, payloads=payloads)


def search(index: Index, query: str, top_k: int, model: str) -> list[tuple[str, float, dict]]:
    query_vector = embed_texts([query], model=model)[0]
    scores = index.vectors @ query_vector
    order = np.argsort(-scores)[:top_k]
    return [(index.ids[i], float(scores[i]), index.payloads[i]) for i in order]
