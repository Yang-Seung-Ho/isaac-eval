#!/usr/bin/env bash
# 재제출용 추가 실험 2종 (단일 모델, 이어달리기 지원). 429가 나면 잠시 뒤 같은 명령을 다시 실행.
set -u
cd ~/Documents/sbkim/isaac-eval
export ISAAC_MODEL=gemini-3.5-flash
PY=./.venv/bin/python3

echo "=== [1] 적대적 40문항: PoT/PAL · 카탈로그만(마스킹 없음) · 제안 — 3조건 동일 설정 재실행 (--keep-first-col) ==="
$PY -m isaac_eval.runners.run_catalog_ablation \
  --catalog data/catalog/jeju_phytoplankton.json \
  --queries queries/structured.adversarial.jsonl \
  --gold-module gold --out-prefix adv40_v2 --keep-first-col \
  --systems pot_pal,proposed_unmasked,proposed

echo "=== [2] 하이브리드 10문항: 라우터 → 데이터셋 선택 → 두 에이전트 → 경로별 최종 응답 (종단 간) ==="
$PY -m isaac_eval.runners.run_hybrid_e2e

echo "=== 완료: results/runs/*adv40_v2_*.json, results/runs/hybrid_e2e_*.json 을 압축해 보내주세요 ==="
