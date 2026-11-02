# GrnPFN

GrnPFN is a prior-data fitted network for combinatorial perturbation screens. It is trained only on simulated screens
generated from random gene-regulatory networks and, applied unchanged to a real screen, predicts the interaction of every
query pair, $R_{AB} = \Delta_{AB} - \Delta_A - \Delta_B$, from the single-perturbation profiles and, optionally, the pairs
measured so far, together with a predictive variance. The variance is used by a greedy look-ahead that chooses which
pairs to measure.

This repository contains the simulator, the network, the cross-fitted benchmark and every analysis script of the
manuscript. Data, model weights and result files are not included (see below).

## Layout

| Path | Contents |
|---|---|
| `src/grnpfn/sim.py` | Gene-regulatory-network simulator of CRISPRa, knockdown and drug × anchor screens |
| `src/grnpfn/prior.py` | Simulator prior configuration and hybrid prior: screen-specific interaction component added to simulated screens |
| `src/grnpfn/bank.py` | Pre-generates the bank of simulated screens (shards) |
| `src/grnpfn/model.py` | The network (axial attention over populations × genes, membership-aware attention, variance head) |
| `src/grnpfn/train.py` | Training on the simulation bank (no real screen is loaded) |
| `src/grnpfn/ppc.py` | Prior predictive statistics computed identically on real and simulated screens |
| `eval/bench/grnpfn_real.py` | Real screens in GrnPFN form: cross-fitted cell thirds, outer pair folds, 256 genes |
| `eval/bench/x80_baselines.py` | Ridge and polynomial kernel ridge fitted to measured pairs; kernel design |
| `eval/bench/x77_zeroshot.py` | Leave-one-screen-out zero-shot comparators |
| `eval/bench/x64_pfn_vdesign.py` | Uncertainty-aware greedy look-ahead design |
| `eval/bench/x68_ttt.py` | Optional test-time fine-tuning on the measured pairs |
| `eval/bench/x79_confirm.py`, `x79_report.py`, `x80_report.py` | Pre-registered confirmation on fresh cell splits |
| `eval/bench/x81_reviewer.py`, `x83_fixedtarget.py`, `x83_report.py`, `x85_splithalf.py` | Selection controls, fixed-target rescoring, calibration and reproducibility analyses |
| `eval/bench/x82_*.py` | GEARS, CPA and CellFlow trained and rescored on the benchmark folds |
| `eval/results/**/*.sh` | The run scripts used for the reported analyses |

## Installation

Python 3.12. From the repository root:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

GEARS (0.1.2), CPA (cpa-tools 0.8.1) and CellFlow (0.0.9) were installed in separate environments (`.venv-gears`,
`.venv-cpa`, `.venv-cellflow`) and are needed only for `eval/bench/x82_*.py`.

## Data

All screens are public:

- K562 CRISPR activation pairs: GEO GSE133344 (Norman et al., 2019)
- THP-1 Cas13 knockdown pairs: GEO GSE213957 (Wessels et al., 2023)
- sci-Plex 4 drug combinations in A172, T98G and U87MG cells: GEO GSM7056151 (McFaline-Figueroa et al., 2024)
- Melanoma Perturb-CITE-seq knockouts (shared-cell analysis only): scPerturb release of Frangieh et al. (2021), SCP1064

The cached matrices (`data/bench_cache/{dataset}_hvg3000.h5ad`, 3,000 highly variable genes) are built by
`eval/bench/p5_export_norman.py`, `e9_wessels_build.py`, `e12_sciplex4_build.py`, `e8_frangieh_diag.py` and `x87_gxd_build.py` (see `eval/bench/README.md`).
Paths are resolved by `eval/bench/common.py` (`DATA`, `RESULTS`).

## Training

```bash
cd src
# 140,000 simulated screens in shards
python -m grnpfn.bank --out ../data/grnpfn_bank --n 20000 --procs 14 --seed 1000000      # repeat with further seeds/shards
# base model on the network prior (40,000 steps)
python -m grnpfn.train --bank ../data/grnpfn_bank --out ../eval/results/bench/grnpfn/grnpfn_v1.pt --steps 40000
# hybrid prior (10,000 steps), then membership-aware attention (10,000 steps)
python -m grnpfn.train --bank ../data/grnpfn_bank --out ../eval/results/bench/grnpfn/grnpfn_hybrid.pt --steps 10000 --hybrid --init ../eval/results/bench/grnpfn/grnpfn_v1.pt
python -m grnpfn.train --bank ../data/grnpfn_bank --out ../eval/results/bench/grnpfn/grnpfn_rel.pt --steps 10000 --hybrid --rel --init ../eval/results/bench/grnpfn/grnpfn_hybrid.pt
# predictive-variance head on the frozen network (5,000 steps)
python -m grnpfn.train --bank ../data/grnpfn_bank --out ../eval/results/bench/grnpfn/grnpfn_rel_var.pt --steps 5000 --hybrid --rel --var-head --freeze-trunk --init ../eval/results/bench/grnpfn/grnpfn_rel.pt
```

Training used one 24 GB GPU.

## Evaluation

```bash
cd eval/bench
python x79_confirm.py --part curve --seed 431                      # learning curves and zero-shot comparators, all screens
python x79_confirm.py --part design --seed 431 --dataset norman    # design at 25% of the training pairs
python x80_baselines.py --part curve --seed 431 --dataset norman  # baselines on the same folds and draws (also --part design)
python x83_fixedtarget.py --part gpu --seed 431                    # fixed-target rescoring: selections and GrnPFN predictions
python x83_fixedtarget.py --part cpu --seed 431 --dataset norman   # baselines on every selection; all metrics
python x83_report.py                                               # report (run once)
```

Cell-split seed 301 is the development split; 431, 439, 449 and 457 are the confirmation splits. Seeds 193, 197, 199 and
211 were spent on earlier confirmatory tests and are rejected by the scripts (`phase3.SPENT`).

## License

BSD 3-Clause License. See `LICENSE`.

## Citation

Zhang, H. A simulation-trained prior predicts non-additive interactions early in combinatorial perturbation screens.
Manuscript in preparation.
