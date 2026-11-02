#!/usr/bin/env bash
# R1-C CPA on THP-1, A172, T98G, U87MG, five folds each, sequential (K562 recipe; pre-registered 2026-10-28)
set -uo pipefail
cd /home/figo/software/GrnPFN/eval/bench; R=/home/figo/software/GrnPFN; O=$R/eval/results/bench/p82_external
for ds in wessels sp4_U87MG sp4_T98G sp4_A172; do for k in 0 1 2 3 4; do
  PYTHONWARNINGS=ignore $R/.venv-cpa/bin/python -u x82_cpa.py --dataset $ds --fold $k --epochs 200 > $O/cpa_${ds}_fold$k.log 2>&1
  echo "[$(date '+%F %T')] cpa $ds fold $k rc=$?"
done; done
