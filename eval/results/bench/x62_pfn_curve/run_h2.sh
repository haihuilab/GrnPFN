#!/bin/bash
cd /home/figo/software/GrnPFN/eval/bench
for d in norman wessels sp4_A172 sp4_T98G; do
  OMP_NUM_THREADS=4 ../../.venv/bin/python x62_pfn_curve.py --dataset $d --models grnpfn_rel grnpfn_hybrid2 > ../results/bench/x62_pfn_curve/${d}_h2.log 2>&1 || echo "FAIL $d"
done
echo "h2 eval done"
