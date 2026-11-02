#!/usr/bin/env bash
# R2 (pre-registered 2026-10-28): GPU phase per seed (one process), then CPU phase per seed × screen in parallel; run once
set -uo pipefail
cd /home/figo/software/GrnPFN/eval/bench; P=/home/figo/software/GrnPFN/.venv/bin/python; L=../results/bench/x83_fixedtarget
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 301 431 439 449 457; do
  $P -u x83_fixedtarget.py --part gpu --seed $s > $L/gpu_s$s.log 2>&1; echo "gpu $s rc=$?"
  for ds in norman wessels sp4_A172 sp4_T98G sp4_U87MG; do
    ( nice -n 5 $P -u x83_fixedtarget.py --part cpu --seed $s --dataset $ds > $L/cpu_${ds}_s$s.log 2>&1; echo "cpu $ds $s rc=$?" ) &
  done
done
wait; echo "R2 done"
