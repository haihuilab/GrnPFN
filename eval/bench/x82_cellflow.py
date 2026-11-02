"""R1-C (pre-registered 2026-10-28, manuscript/notes/autonomous_log.md): the K562 recipe of p5_cellflow_norman.py applied
unchanged to another screen (--dataset wessels|sp4_A172|sp4_T98G|sp4_U87MG); writes p82_external/.

Run with the CellFlow environment:  ../../.venv-cellflow/bin/python p5_cellflow_norman.py --fold 0
Conditional OT flow matching from control to perturbed cells in a PCA-50 fitted on the fold's training cells.
A cell's perturbation is the set {gene_1, gene_2} (one-hot gene representations; singles carry a zero 'none'
token), pooled by CellFlow's attention pooling.  Training: control → every single and training-fold combo.
Writes two predictions per test combo to eval/results/bench/p5_external/:
  cellflow_fold{k}.npz      raw CellFlow Δ_AB (decoded mean − control mean)
  cellflowres_fold{k}.npz   measured Δ_A + Δ_B + CellFlow's own interaction (Δ̂_AB − Δ̂_A − Δ̂_B),
                            the same construction as the composed maps in P4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "bench_cache"
OUT = ROOT / "eval" / "results" / "bench" / "p82_external"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pred-ctrl", type=int, default=2000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from cellflow.model import CellFlow

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = json.loads((CACHE / f"{args.dataset}_folds.json").read_text())[str(args.fold)]
    full = ad.read_h5ad(CACHE / f"{args.dataset}_hvg3000.h5ad")
    pert_all = full.obs["pert"].to_numpy()
    mean = lambda m: np.asarray(full.X[m].mean(0)).ravel()
    ctrl_e = mean(pert_all == "ctrl")
    a = full[~full.obs["pert"].isin(f["test_combos"]).to_numpy()].copy()
    X = a.X.toarray().astype(np.float32)
    rng = np.random.default_rng(args.seed + args.fold)
    pca = PCA(50, random_state=args.seed).fit(X[rng.choice(len(X), min(20000, len(X)), replace=False)])
    a.obsm["X_pca"] = pca.transform(X).astype(np.float32)
    pert = a.obs["pert"].to_numpy()
    g1 = np.array(["none" if p == "ctrl" else p.split("+")[0] for p in pert])
    g2 = np.array(["none" if ("+" not in p) else p.split("+")[1] for p in pert])
    genes = sorted((set(g1) | set(g2)) - {"none"})
    cats = genes + ["none"]
    a.obs = pd.DataFrame({"gene_1": pd.Categorical(g1, categories=cats),
                          "gene_2": pd.Categorical(g2, categories=cats),
                          "control": pert == "ctrl"}, index=a.obs_names)
    emb = {g: np.eye(len(genes), dtype=np.float32)[i] for i, g in enumerate(genes)}
    emb["none"] = np.zeros(len(genes), dtype=np.float32)
    a.uns["gene_emb"] = emb

    cf = CellFlow(a, solver="otfm")
    cf.prepare_data(sample_rep="X_pca", control_key="control",
                    perturbation_covariates={"gene": ["gene_1", "gene_2"]},
                    perturbation_covariate_reps={"gene": "gene_emb"})
    cf.prepare_model(seed=args.seed)
    cf.train(num_iterations=args.iters, batch_size=1024, valid_freq=10 ** 9)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)

    ctrl = a[a.obs["control"].to_numpy()]
    ctrl = ctrl[rng.choice(ctrl.n_obs, min(args.n_pred_ctrl, ctrl.n_obs), replace=False)].copy()
    z0 = ctrl.obsm["X_pca"].mean(0)
    queries = {}
    for A, B in zip(f["test_A"], f["test_B"]):
        queries[f"{A}+{B}"] = (A, B)
        queries[A] = (A, "none")
        queries[B] = (B, "none")
    cov = pd.DataFrame({"gene_1": pd.Categorical([v[0] for v in queries.values()], categories=cats),
                        "gene_2": pd.Categorical([v[1] for v in queries.values()], categories=cats),
                        "control": False, "cond": list(queries)})
    out = cf.predict(ctrl, covariate_data=cov, sample_rep="X_pca", condition_id_key="cond")
    dlat = {k: np.asarray(v).mean(0) - z0 for k, v in out.items()}
    combos, raw, res = [], [], []
    for A, B in zip(f["test_A"], f["test_B"]):
        d_ab = dlat[f"{A}+{B}"] @ pca.components_
        inter = (dlat[f"{A}+{B}"] - dlat[A] - dlat[B]) @ pca.components_
        add = mean(pert_all == A) - ctrl_e + mean(pert_all == B) - ctrl_e
        combos.append("+".join(sorted((A, B))))
        raw.append(d_ab)
        res.append(add + inter)
    for name, d in (("cellflow", raw), ("cellflowres", res)):
        np.savez(OUT / f"{name}{args.tag}_{args.dataset}_fold{args.fold}.npz", combos=np.array(combos),
                 delta=np.stack(d).astype(np.float32))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
