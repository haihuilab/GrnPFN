#!/bin/bash
# U3: TTT on grnpfn_rel, 4 screens (runs alongside chains e and f; memory capped)
cd /home/figo/software/GrnPFN/eval/bench; O=../results/bench/x68_ttt; mkdir -p $O
for d in sp4_T98G sp4_A172 wessels norman; do
  X60_MEMFRAC=0.2 X60_BS=2 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 ../../.venv/bin/python x68_ttt.py --dataset $d --model grnpfn_rel --S 100 > $O/${d}_rel.log 2>&1 || echo "FAIL $d"
done
echo "chain u3 done"
