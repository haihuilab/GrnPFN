#!/bin/bash
# C3 quick iteration: member-signature prior vs matched control, 2k steps each, in parallel
cd /home/figo/software/GrnPFN/src; R=/home/figo/software/GrnPFN/eval/results/bench/grnpfn
PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_mem2k.pt --steps 2000 --hybrid --rel --member --init $R/grnpfn_rel.pt --workers 6 > $R/train_mem2k.log 2>&1 &
PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv/bin/python -m grnpfn.train --bank ../data/grnpfn_bank --out $R/grnpfn_ctl2k.pt --steps 2000 --hybrid --rel --init $R/grnpfn_rel.pt --workers 6 > $R/train_ctl2k.log 2>&1 &
wait; echo "chain c3q done"
