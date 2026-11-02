#!/bin/bash
# C5 confirmation on fresh seeds 431/439/449/457 (pre-registered 2026-10-27). Run once.
cd /home/figo/software/GrnPFN/eval/bench; O=../results/bench/x79_confirm; P=../../.venv/bin/python
for s in 431 439 449 457; do
  ( CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $P x79_confirm.py --part curve --seed $s > $O/curve_s$s.log 2>&1 || echo "FAIL curve $s" ) &
done
( for s in 431 439 449 457; do CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $P x79_confirm.py --part external --seed $s > $O/external_s$s.log 2>&1 || echo "FAIL external $s"; done ) &
( for s in 431 439; do for d in sp4_A172 sp4_T98G sp4_U87MG norman wessels; do OMP_NUM_THREADS=4 $P x79_confirm.py --part design --seed $s --dataset $d > $O/design_${d}_s$s.log 2>&1 || echo "FAIL design $d $s"; done; done ) &
( for s in 449 457; do for d in sp4_A172 sp4_T98G sp4_U87MG norman wessels; do OMP_NUM_THREADS=4 $P x79_confirm.py --part design --seed $s --dataset $d > $O/design_${d}_s$s.log 2>&1 || echo "FAIL design $d $s"; done; done ) &
wait; echo "chain c5 done"
