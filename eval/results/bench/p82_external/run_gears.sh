#!/usr/bin/env bash
# R1-C GEARS on THP-1, five folds in parallel (K562 recipe; pre-registered 2026-10-28)
set -uo pipefail
cd /home/figo/software/GrnPFN/eval/bench; R=/home/figo/software/GrnPFN; O=$R/eval/results/bench/p82_external
for k in 0 1 2 3 4; do
  PYTHONWARNINGS=ignore $R/.venv-gears/bin/python -u x82_gears.py --dataset wessels --fold $k --epochs 20 > $O/gears_wessels_fold$k.log 2>&1 &
done
wait; echo "[$(date '+%F %T')] gears done"; ls $O/gears_wessels_fold*.npz | wc -l
