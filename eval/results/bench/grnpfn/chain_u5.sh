#!/bin/bash
cd /home/figo/software/GrnPFN/eval/bench; O=../results/bench/x68_ttt
for d in wessels norman sp4_T98G sp4_A172; do
  X60_MEMFRAC=0.2 X60_BS=2 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 ../../.venv/bin/python x68_ttt.py --dataset $d --model grnpfn_rel --S 100 --split3 > $O/${d}_rel_split3.log 2>&1 || echo "FAIL $d"
done
echo "chain u5 done"
