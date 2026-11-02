#!/bin/bash
# C6 standard baselines, fresh seeds, curve part (pre-registered 2026-10-27); run once
cd /home/figo/software/GrnPFN/eval/bench; O=../results/bench/x80_baselines; P=../../.venv/bin/python
for s in 431 439 449 457; do for d in norman wessels sp4_A172 sp4_T98G sp4_U87MG; do
  ( CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $P x80_baselines.py --part curve --seed $s --dataset $d > $O/curve_${d}_s$s.log 2>&1 || echo "FAIL curve $d $s" ) &
done; done
wait; echo "chain c6 fresh curve done"
