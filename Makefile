PY := ./.venv/bin/python3

setup:            ## 가상환경 + 의존성 설치
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

test:             ## STEP 1 — 회귀 테스트 (게이트)
	$(PY) -m pytest -q

index:            ## STEP 2 — 카탈로그 + 표2(마스킹) + 재현성 스냅샷
	$(PY) -m isaac_eval.runners.build_index

corpus:           ## STEP 2 — PDF 청킹 + 임베딩 인덱스
	$(PY) -m isaac_eval.runners.build_corpus

calibrate:        ## STEP 3 — 복잡도 임계값 캘리브레이션 (v1/v2/v3)
	$(PY) -m isaac_eval.runners.calibrate_threshold

structured:       ## STEP 4 — 정형 30건 (처음부터)
	$(PY) -m isaac_eval.runners.run_structured

resume:           ## STEP 4 — 미완료 정형 질의만 이어서 실행
	$(PY) -m isaac_eval.runners.resume_structured

unstructured:     ## STEP 5 — 비정형 30건
	$(PY) -m isaac_eval.runners.run_unstructured

hybrid:           ## STEP 6 — 복합 10건
	$(PY) -m isaac_eval.runners.run_hybrid

routing:          ## STEP 7 — 70건 라우팅 정확도
	$(PY) -m isaac_eval.runners.run_routing

baselines:        ## STEP 7 — 기준선 3종 × 정형 30건
	$(PY) -m isaac_eval.runners.run_baselines

aggregate:        ## 정형 지표 재집계 (NCR/AA/실행성공률)
	$(PY) -m isaac_eval.runners.aggregate_structured

agg-unstructured: ## 비정형 지표 재집계 (경로선택·우회전환율)
	$(PY) -m isaac_eval.runners.aggregate_unstructured

errors:           ## 5.6절 오류 유형별 빈도
	$(PY) -m isaac_eval.runners.error_analysis

refresh-gold:     ## gold.py 수정 후 AA만 재채점 (LLM 재호출 없음)
	$(PY) -m isaac_eval.runners.refresh_gold

annotation-sheet: ## 주석자 2인 독립 라벨링 시트 생성 (70건, 라벨 공란)
	$(PY) -m isaac_eval.runners.make_annotation_sheet

agreement:        ## 주석자 2인 라벨로 Cohen's kappa 산출
	$(PY) -m isaac_eval.runners.score_agreement

rerun-scope:      ## 범위 판정만 단일 모델로 재측정 (경로 선택 확정값)
	$(PY) -m isaac_eval.runners.rerun_scope

eval-scope:       ## 범위 판정 규칙 비교 (dev/test 분할, LLM 호출 없음)
	$(PY) -m isaac_eval.runners.eval_scope_rules

eval-scope-llm:   ## 범위 판정 규칙 비교 + 개정 프롬프트 LLM 재판정
	$(PY) -m isaac_eval.runners.eval_scope_rules --with-llm

tables:           ## STEP 9 — 논문 표 생성 (표4·하이브리드·충실성 시트)
	$(PY) -m isaac_eval.runners.make_tables

report:           ## LLM 재호출 없이 모든 지표·표 재생성
	$(MAKE) aggregate agg-unstructured errors tables

all: test index corpus calibrate resume unstructured hybrid routing baselines report

.PHONY: setup test index corpus calibrate structured resume unstructured hybrid \
        routing baselines aggregate agg-unstructured errors refresh-gold tables report all \
        annotation-sheet agreement rerun-scope eval-scope eval-scope-llm
