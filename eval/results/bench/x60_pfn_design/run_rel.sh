#!/bin/bash
cd /home/figo/software/GrnPFN/eval/bench
for d in norman wessels sp4_A172 sp4_T98G; do
  X60_MEMFRAC=0.2 X60_BS=3 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 ../../.venv/bin/python x60_pfn_design.py --dataset $d --model grnpfn_rel > ../results/bench/x60_pfn_design/${d}_rel.log 2>&1 || echo "FAILED $d"
done
echo "P4-2 done"
