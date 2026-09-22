"""논문 5.5절 — 복잡도 임계값 캘리브레이션.

1) 비정형 질의를 dev/test로 층화 분할 (low/high 비율 유지, seed 고정)
2) dev에서 라벨별 점수 분포 산출
3) 관측 구간을 0.1 간격으로 스윕해 dev F1 최대 임계값 선택
4) 선택된 임계값을 test에 1회만 적용
5) 도달 가능성 점검: low 라벨 최댓값 < 임계값 인지 확인
"""
from __future__ import annotations
import random
from dataclasses import dataclass


def stratified_split(items: list[dict], label_key: str, dev_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for item in items:
        by_label.setdefault(item[label_key], []).append(item)

    dev: list[dict] = []
    test: list[dict] = []
    total = len(items)
    for label, group in sorted(by_label.items()):
        shuffled = sorted(group, key=lambda x: x["qid"])
        rng.shuffle(shuffled)
        take = round(len(group) * dev_size / total)
        dev.extend(shuffled[:take])
        test.extend(shuffled[take:])
    return sorted(dev, key=lambda x: x["qid"]), sorted(test, key=lambda x: x["qid"])


def distribution(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    ordered = sorted(scores)
    n = len(ordered)

    def quantile(q: float) -> float:
        position = q * (n - 1)
        low = int(position)
        high = min(low + 1, n - 1)
        frac = position - low
        return round(ordered[low] * (1 - frac) + ordered[high] * frac, 4)

    return {
        "n": n,
        "min": round(ordered[0], 4),
        "q1": quantile(0.25),
        "median": quantile(0.5),
        "q3": quantile(0.75),
        "max": round(ordered[-1], 4),
    }


@dataclass
class ThresholdEval:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    youden_j: float


def evaluate_threshold(items: list[dict], threshold: float) -> ThresholdEval:
    """high 라벨을 양성으로 둔다. score >= threshold 이면 high로 예측."""
    tp = fp = fn = tn = 0
    for item in items:
        predicted_high = item["score"] >= threshold
        actual_high = item["label"] == "high"
        if predicted_high and actual_high:
            tp += 1
        elif predicted_high and not actual_high:
            fp += 1
        elif not predicted_high and actual_high:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return ThresholdEval(
        threshold=round(threshold, 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        youden_j=round(recall + specificity - 1, 4),
    )


def sweep(items: list[dict], step: float = 0.1) -> list[ThresholdEval]:
    scores = [i["score"] for i in items]
    if not scores:
        return []
    low, high = min(scores), max(scores)
    results = []
    n_steps = max(1, int((high - low) / step) + 1)
    for k in range(n_steps + 1):
        results.append(evaluate_threshold(items, low + k * step))
    return results


def _rank_auc(positive: list[float], negative: list[float]) -> float | None:
    """high를 양성으로 둔 ROC AUC. 0.5는 무작위, 0.5 미만은 방향 역전을 뜻한다."""
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for n in negative:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return round(wins / (len(positive) * len(negative)), 4)


def calibrate(scored_items: list[dict], version: str, dev_size: int = 20, seed: int = 20260811,
              step: float = 0.1) -> dict:
    """scored_items: [{"qid", "label": "low"|"high", "score": float}, ...]"""
    dev, test = stratified_split(scored_items, "label", dev_size, seed)

    dist_low = distribution([i["score"] for i in dev if i["label"] == "low"])
    dist_high = distribution([i["score"] for i in dev if i["label"] == "high"])

    candidates = sweep(dev, step=step)
    best = max(candidates, key=lambda c: (c.f1, c.youden_j)) if candidates else None
    selected = best.threshold if best else None

    test_eval = evaluate_threshold(test, selected) if selected is not None else None

    all_low_scores = [i["score"] for i in scored_items if i["label"] == "low"]
    all_high_scores = [i["score"] for i in scored_items if i["label"] == "high"]
    reachable = bool(selected is not None and all_low_scores and max(all_low_scores) < selected)

    # route_a_reachable=False는 두 가지 서로 다른 실패를 가리킬 수 있다:
    # (1) 방향 역전 — low가 high보다 높게 채점되어 임계값을 올릴수록 low가 먼저 걸린다.
    # (2) 부분 겹침 — 방향은 맞지만 두 분포의 꼬리가 겹친다.
    # 논문에서 두 경우를 구분해야 하므로 방향성과 분리도를 함께 기록한다.
    low_median = distribution(all_low_scores)["median"]
    high_median = distribution(all_high_scores)["median"]
    direction_correct = bool(
        low_median is not None and high_median is not None and high_median > low_median
    )
    auc = _rank_auc(all_high_scores, all_low_scores)

    return {
        "direction_correct": direction_correct,
        "separation_auc": auc,
        "median_gap": round(high_median - low_median, 4) if (low_median is not None and high_median is not None) else None,
        "version": version,
        "seed": seed,
        "n_dev": len(dev),
        "n_test": len(test),
        "dist_low": dist_low,
        "dist_high": dist_high,
        "selected_threshold": selected,
        "dev_f1": best.f1 if best else None,
        "dev_youden_j": best.youden_j if best else None,
        "test_f1": test_eval.f1 if test_eval else None,
        "test_confusion": {"tp": test_eval.tp, "fp": test_eval.fp, "fn": test_eval.fn, "tn": test_eval.tn} if test_eval else None,
        "route_a_reachable": reachable,
        "max_low_score_all": round(max(all_low_scores), 4) if all_low_scores else None,
    }
