#!/usr/bin/env bash
set -u
cd ~/Documents/sbkim/isaac-eval
export ISAAC_MODEL=gemini-3.5-flash
PY=./.venv/bin/python3

# CRAG 7건은 이미 30/30 완료(이 스크립트에서 제외).
# 어블레이션은 rename ON(--keep-first-col)로 미리보기·샌드박스 컬럼명 일치. 새 prefix(-v2)로 오염분 우회.

echo "==================== [1] 카탈로그 어블레이션 · 적대적 6건 (rename ON, 재실행) ===================="
$PY -m isaac_eval.runners.run_catalog_ablation \
  --catalog data/catalog/jeju_phytoplankton.json \
  --queries queries/structured.adversarial.jsonl \
  --gold-module gold --out-prefix jeju_adv_abl_v2 --keep-first-col

echo
echo "==================== [2] 카탈로그 어블레이션 · 제주 정형 30건 (rename ON, 재실행) ===================="
$PY -m isaac_eval.runners.run_catalog_ablation \
  --catalog data/catalog/jeju_phytoplankton.json \
  --queries queries/structured.jeju.jsonl \
  --gold-module gold --out-prefix jeju_abl_v2 --keep-first-col

echo
echo "==================== 완료 ===================="
