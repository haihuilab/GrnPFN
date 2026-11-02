#!/bin/bash
# C6 fresh-seed design part, incremental: each baseline design job starts as soon as its x79 GrnPFN design CSV exists
# (replaces chain_c6_fresh_design.sh, which only waited and ran nothing). Same jobs and settings; niced so the GPU runs keep CPU. Run once.
cd /home/figo/software/GrnPFN/eval/bench; O=../results/bench/x80_baselines; P=../../.venv/bin/python; C=../results/bench/x79_confirm
for s in 431 439 449 457; do for d in norman wessels sp4_A172 sp4_T98G sp4_U87MG; do
  ( until [ -f $C/design_${d}_s$s.csv ]; do sleep 120; done
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 $P x80_baselines.py --part design --seed $s --dataset $d > $O/design_${d}_s$s.log 2>&1 || echo "FAIL design $d $s" ) &
done; done
wait; echo "chain c6 fresh design done"
