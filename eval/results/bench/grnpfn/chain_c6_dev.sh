#!/bin/bash
# C6 standard baselines, dev seed 301 (pre-registered 2026-10-27)
cd /home/figo/software/GrnPFN/eval/bench; O=../results/bench/x80_baselines; P=../../.venv/bin/python
for d in norman wessels sp4_A172 sp4_T98G sp4_U87MG; do
  for part in curve design; do
    ( CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $P x80_baselines.py --part $part --seed 301 --dataset $d > $O/${part}_${d}_s301.log 2>&1 || echo "FAIL $part $d" ) &
  done
done
wait; echo "chain c6 dev done"
