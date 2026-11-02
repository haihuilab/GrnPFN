#!/bin/bash
# P4-H6: after the P4-H5 design run (chain e): partner-aligned attention + matched control, then x62 evaluation
R=/home/figo/software/GrnPFN/eval/results/bench/grnpfn
cd /home/figo/software/GrnPFN/src
PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_pa.pt --steps 10000 --hybrid --rel --pa --init $R/grnpfn_rel.pt --workers 8 > $R/train_pa.log 2>&1; echo "exit $?" >> $R/train_pa.log
PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_rel2.pt --steps 10000 --hybrid --rel --init $R/grnpfn_rel.pt --workers 8 > $R/train_rel2.log 2>&1; echo "exit $?" >> $R/train_rel2.log
cd /home/figo/software/GrnPFN/eval/bench
for d in norman wessels sp4_A172 sp4_T98G; do
  OMP_NUM_THREADS=4 ../../.venv/bin/python x62_pfn_curve.py --dataset $d --models grnpfn_pa grnpfn_rel2 > ../results/bench/x62_pfn_curve/${d}_pa.log 2>&1 || echo "FAIL $d"
done
echo "chain f done"
