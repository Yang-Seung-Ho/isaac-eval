#!/bin/bash
# 유료 2차 — 단일모델(flash) 전량 재실행: C-7(제주 정형/기준선) + A-3(논문에이전트 현재구현)
# 기존 로그 백업 후 덮어씀. 유료 키라 429 없음 → gemini-3.5-flash 단일로 실행됨.
cd ~/Documents/sbkim/isaac-eval
export ISAAC_MODEL=gemini-3.5-flash
PY=./.venv/bin/python3
BK="results/_backup_rerun_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$BK" && cp -r results/runs "$BK/"
echo "백업: $BK"

echo; echo "==================== A-3 논문 에이전트(비정형 30건) 현재 구현 재실행 ===================="
rm -f results/runs/unstructured_U*.json
$PY -m isaac_eval.runners.run_unstructured

echo; echo "==================== C-7 제주 정형 제안(단일모델) 재실행 ===================="
rm -f results/runs/proposed_S*.json
$PY -m isaac_eval.runners.run_structured

echo; echo "==================== C-7 제주 기준선 3종(단일모델) 재실행 ===================="
rm -f results/runs/standard_rag_S*.json results/runs/self_rag_S*.json results/runs/crag_S*.json
$PY -m isaac_eval.runners.run_baselines

echo; echo "==================== 완료 ===================="
echo "요약 CSV:"; ls results/tables/*.csv 2>/dev/null | grep -E "structured_metrics.csv|baseline|routing|unstructured" | head
