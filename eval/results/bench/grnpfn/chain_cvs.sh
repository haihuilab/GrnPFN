#!/bin/bash
# C2 H-CVS: cross-validated TTT stopping; random-pair curves (x68) and design pipeline (x69), run as two parallel streams
cd /home/figo/software/GrnPFN/eval/bench
E="X60_BS=2 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4"
( for d in norman wessels sp4_A172 sp4_T98G; do env X60_BS=2 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 ../../.venv/bin/python x68_ttt.py --dataset $d --model grnpfn_rel --S 100 --cv 5 > ../results/bench/x68_ttt/${d}_rel_cv5.log 2>&1 || echo "FAIL x68 $d"; done; echo "x68 stream done" ) &
wait; echo "chain cvs done"
