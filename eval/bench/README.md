# GrnPFN benchmark scripts

Every number and figure in the manuscript is produced from the results folders listed here
(`eval/results/bench/…`); `manuscript/scripts/make_pfn_main_figures.py`, `pfn_supp_tables.py` and
`pfn_numbers.py` read them. Paths are resolved by `common.py` (`DATA`, `RESULTS`).

## Data caches

| Screen | Builder | Cache (`data/bench_cache/`) |
|---|---|---|
| K562 CRISPRa pairs (Norman 2019) | `p5_export_norman.py` | `norman_hvg3000.h5ad`, `norman_folds.json` |
| THP-1 Cas13 pairs (Wessels 2023) | `e9_wessels_build.py` | `wessels_hvg3000.h5ad`, `wessels_folds.json` |
| sci-Plex 4 drug pairs, A172 / T98G / U87MG | `e12_sciplex4_build.py` | `sp4_*_hvg3000.h5ad`, `sp4_*_folds.json` |
| Melanoma knockout × context (Frangieh 2021) | `e8_frangieh_diag.py` (cache), `frangieh_pairs_build.py` (pairs) | `frangieh_hvg3000.h5ad`, `frangieh_pairs_hvg3000.h5ad` |
| A375 CRISPRi × vemurafenib, held-out screen | `x87_gxd_build.py` | `gxd_pf_hvg3000.h5ad`, `gxd_go_hvg3000.h5ad` |
| Real-screen pool for prior statistics | `x13_real_pool.py`, `x13_realmeta.py` | `data/grnpfn_real_pool/` |

Outer pair folds and the kernel-ridge / polynomial fits shared by the comparators live in `benchmark.py`;
`grnpfn_real.py` puts a real screen into GrnPFN form (cross-fitted cell thirds, outer pair folds, 256 genes).

## Training and development (seed 301, development split)

| Step | Script | Results folder |
|---|---|---|
| Simulation bank and training | `src/grnpfn/{bank,train}.py`, run scripts `results/bench/grnpfn/chain_*.sh` | `grnpfn/` (checkpoints `grnpfn_rel.pt`, `grnpfn_rel_var.pt`) |
| Development learning curves, attention and prior variants | `x62_pfn_curve.py`, `x62_report.py`, `x72_attn_diag.py` | `x62_pfn_curve/` |
| Uncertainty-aware design (greedy look-ahead) | `x60_pfn_design.py`, `x64_pfn_vdesign.py` | `x60_pfn_design/` |
| Adaptive (two-stage) design | `x76_adaptive_design.py`, `x76_report.py` | `x76_adaptive/` (removed 2026-10-29; statistics kept in `manuscript/notes/pfn_numbers.md`) |
| Fine-tuning on measured pairs | `x68_ttt.py`, `x68_report.py` | `x68_ttt/` |
| Zero-shot comparators (leave-one-screen-out) | `x77_zeroshot.py` | `x77_zeroshot/` |
| Deep models rescored on the folds | `x78_external.py`, `x82_{export_folds,gears,cpa,cellflow,score}.py` | `x78_external/`, `p82_external/` |
| Learnability statistics | `x74_learnability.py` | `x74_learn/` |
| Diagnostics | `x71_diag.py`, `x71b_diag.py`, `x65_calib.py`, `x61_pfn_context.py`, `x70_u5_report.py`, `x73_cvs_report.py` | `x71_diag/`, `x61_pfn_context.csv` |
| Shared-cell artefact (cross-fitting) | `e8_crossfit.py`, `e8_iter2_otop.py`, `diag_norman.py`, `diag_norman_crossfit.py` | `e8_crossfit/`, `e8_iter2_otop/`, `diag_norman/` |

## Confirmation on fresh cell splits (seeds 431, 439, 449, 457; pre-registered)

| Step | Script | Results folder |
|---|---|---|
| GrnPFN learning curves, designs and held-out screen | `x79_confirm.py`, `x79_report.py` | `x79_confirm/` |
| Kernel ridge, gradient boosting, ridge, linear model | `x80_baselines.py`, `x80_report.py` | `x80_baselines/` |
| Selection controls | `x81_reviewer.py`, `x81_report.py` | `x81_reviewer/` |
| Fixed-target rescoring and calibration | `x83_fixedtarget.py`, `x83_report.py` | `x83_fixedtarget/` |
| Split-half reproducibility | `x85_splithalf.py` | `x85_splithalf/` |
| Unseen vs seen perturbations, confidence | `x86_novelty.py`, `x86_report.py` | `x86_novelty/` |
| Held-out screen (A375 CRISPRi × vemurafenib) | `x87_gxd_eval.py`, `x87_report.py` | `x87_gxd/` |
