#!/bin/bash
# 유료 배치 1차 — 전부 '추가' 실험(기존 결과 비파괴). 단일모델 고정.
cd ~/Documents/sbkim/isaac-eval
export ISAAC_MODEL=gemini-3.5-flash
PY=./.venv/bin/python3
BK="results/_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK" && cp -r results/runs "$BK/" 2>/dev/null
echo "백업: $BK"

echo; echo "==================== [1] A-2 PoT/PAL 어블레이션(마스킹OFF+샌드박스ON) ===================="
echo "--- jeju ---"
$PY -m isaac_eval.runners.run_baselines_multi \
  --catalog data/catalog/jeju_phytoplankton.json --queries queries/structured.jeju.jsonl \
  --gold-module gold --out-prefix jeju --systems pot_pal --limit 30
for m in bg11 nies ohm; do
echo "--- asv_$m ---"
$PY -m isaac_eval.runners.run_baselines_multi \
  --catalog data/catalog/microbiome_asv_$m.json --queries queries/structured.asv.$m.jsonl \
  --gold-module gold_asv --out-prefix asv_$m --systems pot_pal --drop-cols ASV --limit 11
done

echo; echo "==================== [2] C-5 데이터셋 선택 정확도(63건) ===================="
$PY -m isaac_eval.runners.run_dataset_selection

echo; echo "==================== [3] E-3 스케일링(standard_rag + pot_pal, 행수별) ===================="
for n in n10 n100 n1000; do
echo "--- BG11 $n 행 ---"
$PY -m isaac_eval.runners.run_baselines_multi \
  --catalog data/catalog/microbiome_asv_bg11_$n.json --queries queries/structured.asv.jsonl \
  --gold-module gold_asv --out-prefix asv_bg11_$n --systems standard_rag,pot_pal --drop-cols ASV --limit 11
done

echo; echo "==================== [4] C-4 적대적 질의 재시도 검증(제주) ===================="
$PY -m isaac_eval.runners.run_structured_multi \
  --catalog data/catalog/jeju_phytoplankton.json --queries queries/structured.adversarial.jsonl \
  --gold-module gold --out-prefix jeju_adv --keep-first-col --limit 6

echo; echo "==================== 완료 ===================="
echo "표 CSV:"; ls results/tables/ | grep -E "pot_pal|dataset_selection|usage|jeju_adv|_n(10|100|1000)" 2>/dev/null
