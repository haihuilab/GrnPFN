#!/bin/bash
cd /home/figo/software/GrnPFN/eval/bench
printf "norman\nwessels\nsp4_A172\nsp4_T98G\n" | PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 xargs -P 2 -I{} bash -c '../../.venv/bin/python x60_pfn_design.py --dataset {} > ../results/bench/x60_pfn_design/{}.log 2>&1 || echo FAILED {}'
echo P4-1 done
