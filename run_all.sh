#!/usr/bin/env bash
# Reproduce every experiment: download the data, run BOMiS on all datasets and seeds, and summarize.
#   bash run_all.sh
#   SEEDS="0 1 2" DATASETS="car vote" bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=${SEEDS:-$(seq -s " " 0 19)}
DATASETS=${DATASETS:-"nursery2 krk monks1 car tictactoe phish naticus moto krkp mushroom splice vote"}
OUT=${OUT:-results}

python scripts/download_data.py
for ds in $DATASETS; do
  python scripts/run.py --dataset "$ds" --seeds $SEEDS --out "$OUT"
done
python scripts/summarize.py "$OUT"
