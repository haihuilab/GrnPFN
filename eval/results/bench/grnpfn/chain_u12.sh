#!/bin/bash
# U1 (dense-design prior) and U2 (global-epistasis prior), each from grnpfn_rel for 10k steps; shared matched control grnpfn_rel2 (chain f)
R=/home/figo/software/GrnPFN/eval/results/bench/grnpfn
until grep -q "^exit" $R/train_rel2.log 2>/dev/null; do sleep 60; done
cd /home/figo/software/GrnPFN/src
PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_u1dense.pt --steps 10000 --hybrid --rel --extra-bank ../data/grnpfn_bank_dense --extra-p 0.25 --init $R/grnpfn_rel.pt --workers 8 > $R/train_u1dense.log 2>&1; echo "exit $?" >> $R/train_u1dense.log
PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_u2ge.pt --steps 10000 --hybrid --rel --ge --init $R/grnpfn_rel.pt --workers 8 > $R/train_u2ge.log 2>&1; echo "exit $?" >> $R/train_u2ge.log
cd /home/figo/software/GrnPFN/eval/bench
for d in wessels sp4_T98G sp4_A172 norman; do
  OMP_NUM_THREADS=4 ../../.venv/bin/python x62_pfn_curve.py --dataset $d --models grnpfn_u1dense grnpfn_u2ge > ../results/bench/x62_pfn_curve/${d}_u12.log 2>&1 || echo "FAIL $d"
done
echo "chain u12 done"
