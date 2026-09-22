# isaac-eval — 환각 제어 멀티 에이전트 프레임워크 측정 하네스

논문 「연구 플랫폼에서 LLM 기반 연구자 맞춤형 서비스를 위한 멀티 에이전트 프레임워크」의
정량 평가를 재현하기 위한 코드와 데이터다. 논문 5.7절 재현성 명세에 따라 공개한다.

---

## 요구 사항

- Python 3.11 이상 (측정 환경: 3.13.2)
- Gemini API 키 — [ai.google.dev](https://ai.google.dev)에서 발급

```bash
git clone <repo> && cd isaac-eval
cp .env.example .env      # GEMINI_API_KEY=... 를 채운다
make setup                # 가상환경 + 의존성
```

> **무료 등급 주의.** 모델당 하루 20회 요청 제한이 있고 태평양 시간 자정에 리셋된다.
> 질의 70건 전체 측정에는 며칠이 걸리거나 유료 등급이 필요하다. 모든 러너는
> 이어달리기를 지원하므로 중단 후 다시 실행하면 미완료 항목만 처리한다.

---

## 실행 순서

```bash
make test         # 1. 회귀 테스트 (게이트) — 반드시 먼저 통과
make index        # 2. 카탈로그 + 표2(마스킹) + 재현성 스냅샷
make corpus       # 3. PDF 청킹 + 임베딩 인덱스
make calibrate    # 4. 복잡도 임계값 확정 ★ 본 측정보다 먼저
make resume       # 5. 정형 30건
make unstructured # 6. 비정형 30건
make hybrid       # 7. 복합 10건
make routing      # 8. 라우팅 70건
make baselines    # 9. 기준선 3종 × 30건
make report       # 10. 모든 지표·표 재생성 (LLM 호출 없음)
```

`make all`이 위 전체를 순서대로 실행한다.

**캘리브레이션을 건너뛰면 안 된다.** `settings.yaml`의 `complexity.threshold`가 `null`이면
`run_unstructured`가 에러로 중단된다. 임계값을 관찰 없이 확정하면 논문 4.3절이 보고한
문제가 반복된다.

---

## 데이터 준비

```
data/datasets/jeju_phytoplankton.xlsx   정형 데이터 (19열, 8개월)
data/papers/*.pdf                        논문 5편
```

논문 PDF는 저작권상 재배포하지 않는다. 4.1절에 서지정보가 있으니 직접 배치한 뒤
`make corpus`를 실행하면 청킹·임베딩이 재생성된다.

---

## LLM 재호출 없이 재집계하기

실행 로그(`results/runs/*.json`)에 생성된 코드·샌드박스 시도·응답·gold 값이 모두
저장되어 있다. 채점 로직만 바뀐 경우 API 호출 없이 지표를 다시 낼 수 있다.

```bash
make report        # NCR·AA·실행성공률·오류분포·표 전체
make refresh-gold  # gold.py를 고쳤을 때 AA만 재채점
make eval-scope    # 범위 판정 규칙 비교 (dev/test 분할)
```

---

## 주석자 라벨링

논문 5.2절은 주석자 2인의 독립 라벨링과 협의 이전 일치도 보고를 요구한다.

```bash
make annotation-sheet   # results/annotation/labels_annotator_{a,b}.csv (라벨 공란)
# 두 주석자가 서로 상의하지 않고 각자 채운다
make agreement          # Cohen's kappa 산출
```

---

## 주요 모듈

| 파일 | 역할 | 논문 |
|---|---|---|
| `ontology.py` | 수치 마스킹·구조적 온톨로지·유출 전수 대조 | 4.2 |
| `gold.py` | 참조 구현 (사람이 작성한 Pandas 코드, LLM 미사용) | 4.5 |
| `sandbox.py` | 격리 실행 (`python -I`, 네트워크 차단, 타임아웃, 재시도) | 3.4.2 |
| `numeric.py` | 수치 추출·대조 — NCR 채점의 핵심 | 5.4 |
| `complexity.py` | 복잡도 v1(초기)·v2(수정안)·v3(재보정) | 4.3 |
| `calibration.py` | 층화 분할·임계값 스윕·AUC·도달 가능성 점검 | 5.5 |
| `scope.py` | 질의 범위 판정 규칙 v1(초기)·v2(개정) | 7.3 |
| `paper_agent.py` | 경로 A/B/C 판정 및 응답 | 3.5.2 |
| `baselines/` | Standard RAG · Self-RAG · CRAG | 5.3 |

`complexity.py`의 v1과 `scope.py`의 v1은 **실패한 초기 버전이지만 논문의 실측
근거이므로 삭제하지 않고 보존한다.**

---

## 산출물

```
results/
├── RESULTS_FOR_PAPER.md   실측 결과 요약 (논문 위치 대응표 포함)
├── reproducibility.json    5.7절 명세 항목 전체
├── calibration.json        v1/v2/v3 분포·임계값·AUC
├── runs/*.json             질의별 전체 로그 (160건)
└── tables/*.csv            논문 표 1~7 원자료
```

---

## 측정 중 발견한 계측 결함

지표 산출 코드 자체의 결함 5건을 발견해 수정했고, 각각 회귀 테스트로 고정했다.
상세는 `results/RESULTS_FOR_PAPER.md` §9에 있다. 환각 통제를 주장하는 시스템에서는
계측 코드도 검증 대상이라는 것이 이 과정의 교훈이다.

```bash
make test   # 45 passed
```
