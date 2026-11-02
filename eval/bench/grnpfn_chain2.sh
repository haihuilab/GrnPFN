#!/bin/bash
# Continuation of grnpfn_chain.sh (2026-10-20): v1 and Gate 1 (v1) finished under grnpfn_chain.sh; this runs
# looped training with 10,000 steps (was 30,000) → Gate 1 (looped). Strictly sequential.
set -e
cd /home/figo/software/GrnPFN/src
R=../eval/results/bench/grnpfn



../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_loop_v1.pt --steps 10000 --workers 10 --looped --tmin 4 --tmax 12 > $R/train_loop_v1.log 2>&1
../.venv/bin/python -m grnpfn.evaluate --model $R/grnpfn_loop_v1.pt --episodes 150 --loops 4 8 16 24 --out $R/gate1_loop_v1.csv > $R/gate1_loop_v1.log 2>&1
echo done
