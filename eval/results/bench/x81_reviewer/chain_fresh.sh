#!/usr/bin/env bash
# R1-A and R1-D on the confirmation splits 431/439/449/457, run once (pre-registered 2026-10-28)
set -uo pipefail
cd /home/figo/software/GrnPFN/eval/bench; P=/home/figo/software/GrnPFN/.venv/bin/python; L=../results/bench/x81_reviewer
for s in 431 439 449 457; do
  $P -u x81_reviewer.py --part calib --seed $s > $L/calib_s$s.log 2>&1; echo "calib $s rc=$?"
done
for s in 431 439 449 457; do
  for ds in norman wessels sp4_A172 sp4_T98G sp4_U87MG; do
    nice -n 5 $P -u x81_reviewer.py --part select --seed $s --dataset $ds > $L/select_${ds}_s$s.log 2>&1 &
  done
  wait; echo "select seed $s done"
done
