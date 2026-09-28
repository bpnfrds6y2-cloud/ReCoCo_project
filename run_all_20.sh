#!/usr/bin/env bash
# run_all_20.sh
#
# sb3_recoco_reproduction.py を使って、SAC/TD3 x (9trace個別 + 汎用) = 20モデルを
# 順番に学習・保存する。GPU機の sac_project ディレクトリ直下で実行する想定。
#
# 使い方: bash run_all_20.sh
#
# 時間がかかるので、途中で止まっても再開しやすいよう1本ずつ独立した
# python呼び出しにしてある(既に保存済みの.zipがあればスキップする)。
set -eu

PER_TRACE_STEPS=200000
GLOBAL_STEPS=750000
OUT_DIR=./sb3_models

TRACES=(
  "4G_3mbps"
  "4G_500kbps"
  "4G_700kbps"
  "5G_12mbps"
  "5G_13mbps"
  "WIRED_200kbps"
  "WIRED_35mbps"
  "WIRED_900kbps"
  "trace_300k"
)

mkdir -p "$OUT_DIR"

for AGENT in SAC TD3; do
  # --- trace個別モデル(9本) ---
  for T in "${TRACES[@]}"; do
    OUT="$OUT_DIR/${AGENT}_${T}.zip"
    if [ -f "$OUT" ]; then
      echo "[skip] $OUT は既に存在します"
      continue
    fi
    echo "=== ${AGENT} / trace=${T} (${PER_TRACE_STEPS}steps) ==="
    python3 sb3_recoco_reproduction.py \
      --agent-type "$AGENT" --mode pertrace \
      --trace "./traces/${T}.json" \
      --total-timesteps "$PER_TRACE_STEPS" \
      --output "$OUT" \
      --device auto
  done

  # --- 汎用(グローバル)モデル(1本) ---
  OUT="$OUT_DIR/${AGENT}_global.zip"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT は既に存在します"
  else
    echo "=== ${AGENT} / GLOBAL (9trace random, ${GLOBAL_STEPS}steps) ==="
    python3 sb3_recoco_reproduction.py \
      --agent-type "$AGENT" --mode global \
      --total-timesteps "$GLOBAL_STEPS" \
      --output "$OUT" \
      --device auto
  fi
done

echo "=== 完了: ${OUT_DIR}/ に計20モデル ==="
ls -la "$OUT_DIR"
