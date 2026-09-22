from __future__ import annotations
import math
from isaac_eval.numeric import extract_numbers, flatten_facts, compare


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """비율의 Wilson 점수 신뢰구간 (기본 95%).

    표본이 30건 내외로 작고 관측 비율이 0 또는 1에 붙는 경우가 있어, 정규근사
    (Wald) 구간은 폭이 0이 되거나 [0,1]을 벗어난다. Wilson 구간은 이 경우에도
    유효한 구간을 준다 — 논문 8.4절의 '구간 추정으로 보고' 요구사항에 대응한다.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def rate_with_ci(successes: int, n: int) -> dict:
    lo, hi = wilson_interval(successes, n)
    return {
        "successes": successes,
        "n": n,
        "rate": round(successes / n, 4) if n else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def ncr_detail(record: dict, policy: str = "round"):
    """NCR은 '응답의 수치가 샌드박스 결과에서 유래하는가'를 본다. 질의문에 이미
    주어진 수치(임계값, 월 번호 등)를 응답이 되풀이하는 것은 환각이 아니므로
    허용 집합에 포함한다."""
    reported = extract_numbers(record["final_response"])
    facts = flatten_facts(record["sandbox_value"])
    query_numbers = extract_numbers(record["query"])
    return compare(reported, facts + query_numbers, policy=policy)


def ncr(records: list[dict], policy: str = "round") -> float:
    total = len(records)
    if total == 0:
        return 0.0
    passed = 0
    for r in records:
        if not r["final_ok"]:
            continue
        if ncr_detail(r, policy=policy).pass_:
            passed += 1
    return round(passed / total, 4)


def answer_accuracy(records: list[dict], policy: str = "round") -> float:
    """AA는 sandbox_value와 gold_value의 딕셔너리 키가 아니라, 두 값에 담긴
    수치 집합이 (반올림 정책 하에) 서로 완전히 대응하는지로 판정한다.
    LLM이 고른 JSON 키 이름은 채점 대상이 아니다 — 수치 내용만 비교한다."""
    total = len(records)
    if total == 0:
        return 0.0
    correct = 0
    for r in records:
        if not r["final_ok"]:
            continue
        sandbox_facts = flatten_facts(r["sandbox_value"])
        gold_facts = flatten_facts(r["gold_value"])
        forward = compare(sandbox_facts, gold_facts, policy=policy)
        backward = compare(gold_facts, sandbox_facts, policy=policy)
        if forward.pass_ and backward.pass_:
            correct += 1
    return round(correct / total, 4)


def ncr_context_based(record: dict, policy: str = "round") -> "object":
    """기준선용 NCR. 기준선은 샌드박스가 없으므로 '프롬프트에 주어진 컨텍스트'가
    근거 집합이 된다. 제안 방식은 근거가 '샌드박스가 계산한 값'이라는 점이 대조점이다."""
    reported = extract_numbers(record["final_response"])
    allowed = extract_numbers(record.get("context_text", "")) + extract_numbers(record["query"])
    return compare(reported, allowed, policy=policy)


def answer_accuracy_from_text(records: list[dict], policy: str = "round") -> float:
    """기준선용 AA. gold의 모든 수치가 응답 텍스트에 등장하는지로 채점한다."""
    total = len(records)
    if total == 0:
        return 0.0
    correct = 0
    for r in records:
        reported = extract_numbers(r["final_response"])
        gold_facts = flatten_facts(r["gold_value"])
        if gold_facts and compare(gold_facts, reported, policy=policy).pass_:
            correct += 1
    return round(correct / total, 4)


def routing_accuracy(records: list[dict]) -> float:
    """전체 질의. routed_to == gold_route 비율."""
    scored = [r for r in records if r.get("gold_route") and r.get("routed_to")]
    if not scored:
        return 0.0
    hits = sum(1 for r in scored if r["routed_to"] == r["gold_route"])
    return round(hits / len(scored), 4)


def route_selection_accuracy(records: list[dict]) -> float:
    """비정형 질의. paper_route == gold_paper_route 비율."""
    scored = [r for r in records if r.get("gold_paper_route") and r.get("paper_route")]
    if not scored:
        return 0.0
    hits = sum(1 for r in scored if r["paper_route"] == r["gold_paper_route"])
    return round(hits / len(scored), 4)


def graceful_degradation_rate(records: list[dict]) -> float:
    """complexity_label == 'high' 인 질의 중 degraded == True 비율."""
    high = [r for r in records if r.get("complexity_label") == "high"]
    if not high:
        return 0.0
    return round(sum(1 for r in high if r.get("degraded")) / len(high), 4)


def confusion_matrix(records: list[dict], key_pred: str, key_gold: str) -> dict:
    matrix: dict[str, dict[str, int]] = {}
    for r in records:
        gold = r.get(key_gold)
        pred = r.get(key_pred)
        if gold is None or pred is None:
            continue
        matrix.setdefault(str(gold), {}).setdefault(str(pred), 0)
        matrix[str(gold)][str(pred)] += 1
    return matrix


def complexity_classification_accuracy(records: list[dict], threshold: float) -> float:
    """복잡도 점수가 라벨(low/high)을 임계값 기준으로 맞히는 비율.
    경로 선택 정확도와 달리 LLM 범위 판정의 영향을 받지 않는 순수 공식 성능이다."""
    scored = [r for r in records if r.get("complexity_label") and r.get("complexity")]
    if not scored:
        return 0.0
    hits = 0
    for r in scored:
        predicted = "high" if r["complexity"]["total"] >= threshold else "low"
        if predicted == r["complexity_label"]:
            hits += 1
    return round(hits / len(scored), 4)


def code_exec_success(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"first_attempt": 0.0, "with_retry": 0.0, "mean_attempts": 0.0}
    first_ok = sum(1 for r in records if r["attempts_used"] == 1 and r["final_ok"])
    with_retry_ok = sum(1 for r in records if r["final_ok"])
    mean_attempts = sum(r["attempts_used"] for r in records) / total
    return {
        "first_attempt": round(first_ok / total, 4),
        "with_retry": round(with_retry_ok / total, 4),
        "mean_attempts": round(mean_attempts, 2),
    }
