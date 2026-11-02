"""R1-C (pre-registered 2026-10-28, manuscript/notes/autonomous_log.md): the K562 recipe of p5_gears_norman.py applied
unchanged to another screen (--dataset wessels|sp4_A172|sp4_T98G|sp4_U87MG); writes p82_external/.

Run with the GEARS environment:  ../../.venv-gears/bin/python p5_gears_norman.py --fold 0 [--epochs 20]
Reads data/bench_cache (written by p5_export_norman.py); writes
eval/results/bench/p5_external/gears_fold{k}.npz with the predicted Δ (mean expression − control mean) per test combo.
Scoring is done in the main environment by p5_score_external.py, with the same code as P4.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "bench_cache"
WORK = ROOT / "data" / "GEARS"
OUT = ROOT / "eval" / "results" / "bench" / "p82_external"


class _MaskableCSR(sp.csr_matrix):
    """GEARS 0.1.2 indexes adata.X with a pandas boolean Series, which current scipy rejects."""

    def __getitem__(self, key):
        if isinstance(key, pd.Series):
            key = key.to_numpy()
        return super().__getitem__(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    from gears import GEARS, PertData

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    folds = json.loads((CACHE / f"{args.dataset}_folds.json").read_text())
    f = folds[str(args.fold)]
    a = ad.read_h5ad(CACHE / f"{args.dataset}_hvg3000.h5ad")
    ctrl_mean = np.asarray(a.X[(a.obs["pert"] == "ctrl").to_numpy()].mean(0)).ravel()
    a = a[~a.obs["pert"].isin(f["test_combos"]).to_numpy()].copy()
    if "condition" not in a.obs:
        p_ = a.obs["pert"].astype(str).to_numpy()
        a.obs["condition"] = np.array(["ctrl" if p == "ctrl" else (p if "+" in p else f"{p}+ctrl") for p in p_])
    a.var["gene_name"] = np.asarray(a.var_names)
    a.obs["cell_type"] = args.dataset
    a.obs = a.obs[["condition", "cell_type"]]
    name = f"{args.dataset}_fold{args.fold}_{os.getpid()}"
    pdata = PertData(str(WORK))
    try:
        pdata.new_data_process(dataset_name=name, adata=a, skip_calc_de=False)
        pdata.prepare_split(split="no_test", seed=args.seed)
        pdata.get_dataloader(batch_size=32, test_batch_size=128)
        pdata.adata.X = _MaskableCSR(pdata.adata.X)
        m = GEARS(pdata, device="cuda")
        m.model_initialize(hidden_size=args.hidden)
        m.train(epochs=args.epochs)
        perts = set(m.pert_list)
        ok = [(A, B) for A, B in zip(f["test_A"], f["test_B"]) if A in perts and B in perts]
        print(f"gears {args.dataset} fold {args.fold}: {len(ok)}/{len(f['test_A'])} test combos queryable", flush=True)
        preds = m.predict([[A, B] for A, B in ok])
        combos, deltas = [], []
        for A, B in ok:
            key = next(k for k in (f"{A}_{B}", f"{B}_{A}") if k in preds)
            combos.append("+".join(sorted((A, B))))
            deltas.append(np.asarray(preds[key]).ravel() - ctrl_mean)
        np.savez(OUT / f"gears_{args.dataset}_fold{args.fold}.npz", combos=np.array(combos),
                 delta=np.stack(deltas).astype(np.float32), genes=np.asarray(a.var_names))
        print(f"done in {time.time() - t0:.0f}s", flush=True)
    finally:
        shutil.rmtree(WORK / name, ignore_errors=True)


if __name__ == "__main__":
    main()
