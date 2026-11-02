#!/bin/bash
# C4 H-ADAPT: two parallel streams
cd /home/figo/software/GrnPFN/eval/bench
( for d in sp4_A172 sp4_U87MG norman; do env OMP_NUM_THREADS=4 ../../.venv/bin/python x76_adaptive_design.py --dataset $d > ../results/bench/x76_adaptive/$d.log 2>&1 || echo "FAIL $d"; done ) &
( for d in sp4_T98G wessels; do env OMP_NUM_THREADS=4 ../../.venv/bin/python x76_adaptive_design.py --dataset $d > ../results/bench/x76_adaptive/$d.log 2>&1 || echo "FAIL $d"; done ) &
wait; echo "chain c4 done"
