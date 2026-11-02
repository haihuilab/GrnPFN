#!/bin/bash
cd /home/figo/software/GrnPFN/eval/bench; D=sp4_U87MG; O=../results/bench
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 X60_BS=2
../../.venv/bin/python x60_pfn_design.py --dataset $D --model grnpfn_rel > $O/x60_pfn_design/${D}_rel.log 2>&1 || echo "FAIL x60"
cp $O/x60_pfn_design/${D}_s301_grnpfn_rel.csv $O/x60_pfn_design/${D}_s301.csv
../../.venv/bin/python x64_pfn_vdesign.py --dataset $D --model grnpfn_rel_var > $O/x60_pfn_design/${D}_vdesign.log 2>&1 || echo "FAIL x64"
../../.venv/bin/python x62_pfn_curve.py --dataset $D --models grnpfn_rel > $O/x62_pfn_curve/${D}_rel.log 2>&1 || echo "FAIL x62"
for cv in 0 5; do
  ../../.venv/bin/python x68_ttt.py --dataset $D --model grnpfn_rel --S 100 --cv $cv > $O/x68_ttt/${D}_rel_cv$cv.log 2>&1 || echo "FAIL x68 $cv"
done
echo "chain u87 done"
