#!/usr/bin/env bash
# R1-C smoke tests (tiny settings; outputs deleted afterwards, never used)
set -uo pipefail
cd /home/figo/software/GrnPFN/eval/bench; R=/home/figo/software/GrnPFN; L=$R/eval/results/bench/p82_external/smoke
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONWARNINGS=ignore
$R/.venv-cpa/bin/python -u x82_cpa.py --dataset sp4_U87MG --fold 0 --epochs 2 --tag _smoke > $L/cpa_u87.log 2>&1; echo "cpa u87 rc=$?"
$R/.venv-cpa/bin/python -u x82_cpa.py --dataset wessels --fold 0 --epochs 2 --tag _smoke > $L/cpa_thp1.log 2>&1; echo "cpa thp1 rc=$?"
$R/.venv-cellflow/bin/python -u x82_cellflow.py --dataset sp4_U87MG --fold 0 --iters 200 --n-pred-ctrl 200 --tag _smoke > $L/cf_u87.log 2>&1; echo "cellflow u87 rc=$?"
$R/.venv-cellflow/bin/python -u x82_cellflow.py --dataset wessels --fold 0 --iters 200 --n-pred-ctrl 200 --tag _smoke > $L/cf_thp1.log 2>&1; echo "cellflow thp1 rc=$?"
$R/.venv-gears/bin/python -u x82_gears.py --dataset wessels --fold 0 --epochs 1 > $L/gears_thp1.log 2>&1; echo "gears thp1 rc=$?"
