#!/bin/bash
# GRN-PFN: v1 training → Gate 1 (v1) → looped training → Gate 1 (looped). Strictly sequential, no process matching.
set -e
cd /home/figo/software/GrnPFN/src
R=../eval/results/bench/grnpfn
../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_v1.pt --steps 40000 --workers 10 > $R/train_v1.log 2>&1
../.venv/bin/python -m grnpfn.evaluate --model $R/grnpfn_v1.pt --episodes 150 --out $R/gate1_v1.csv > $R/gate1_v1.log 2>&1
../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_loop_v1.pt --steps 30000 --workers 10 --looped --tmin 4 --tmax 12 > $R/train_loop_v1.log 2>&1
../.venv/bin/python -m grnpfn.evaluate --model $R/grnpfn_loop_v1.pt --episodes 150 --loops 4 8 16 24 --out $R/gate1_loop_v1.csv > $R/gate1_loop_v1.log 2>&1
echo done
