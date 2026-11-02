#!/bin/bash
# C6 fresh-seed design part: waits for the 20 x79 GrnPFN design CSVs, then runs the baselines on each selection (run once)
cd /home/figo/software/GrnPFN/eval/bench; O=../results/bench/x80_baselines; P=../../.venv/bin/python; C=../results/bench/x79_confirm
until [ "$(ls $C/design_*_s4*.csv 2>/dev/null | wc -l)" -ge 20 ]; do sleep 120; done
for s in 431 439 449 457; do for d in norman wessels sp4_A172 sp4_T98G sp4_U87MG; do
  ( CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $P x80_baselines.py --part design --seed $s --dataset $d > $O/design_${d}_s$s.log 2>&1 || echo "FAIL design $d $s" ) &
done; done
wait; echo "chain c6 fresh design done"
