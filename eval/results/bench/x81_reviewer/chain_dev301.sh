#!/usr/bin/env bash
# R1-A and R1-D on the development split (seed 301), pre-registered 2026-10-28
set -uo pipefail
cd /home/figo/software/GrnPFN/eval/bench; P=/home/figo/software/GrnPFN/.venv/bin/python; L=../results/bench/x81_reviewer
$P -u x81_reviewer.py --part calib --seed 301 > $L/calib_s301.log 2>&1; echo "calib 301 rc=$?"
for ds in norman wessels sp4_A172 sp4_T98G sp4_U87MG; do
  nice -n 5 $P -u x81_reviewer.py --part select --seed 301 --dataset $ds > $L/select_${ds}_s301.log 2>&1; echo "select $ds 301 rc=$?"
done
