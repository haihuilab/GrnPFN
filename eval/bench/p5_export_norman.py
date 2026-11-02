"""Export the Norman benchmark (HVGs, log1p CP10k, perturbation labels, P1/P4 folds) for external tools.

External baselines run in their own environments (``.venv-gears``, ``.venv-cellflow``, ``.venv-cpa``) and read
this cache, so they see exactly the cells, genes and splits our methods see.

Writes data/bench_cache/norman_hvg3000.h5ad (X = log1p CP10k on the P1 HVGs, CSR; obs: pert, condition) and
data/bench_cache/norman_folds.json ({fold: {train_combos, test_combos}}).
"""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.model_selection import KFold

from common import DATA, load_norman, log1p_cp10k


def select_hvgs(adata, n_top: int, seed: int) -> list[str]:
    """HVGs from control + single-perturbation cells only (no combination cells → no leakage)."""
    pert = adata.obs["pert"].to_numpy()
    rows = np.where(np.array(["+" not in p for p in pert]))[0]
    rows = np.random.default_rng(seed).choice(rows, min(30000, len(rows)), replace=False)
    a = sc.AnnData(sp.csr_matrix(adata.layers["counts"][rows]), var=adata.var[[]].copy())
    a.layers["counts"] = a.X.copy()
    try:
        sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat_v3", layer="counts")
    except ImportError:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat")
    return a.var_names[a.var["highly_variable"].to_numpy()].tolist()

OUT = DATA / "bench_cache"


def main(seed: int = 0, folds: int = 5, n_hvg: int = 3000) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adata, triples = load_norman()
    feats = select_hvgs(adata, n_hvg, seed)
    X = log1p_cp10k(adata, feats)
    pert = adata.obs["pert"].astype(str).to_numpy()
    cond = np.array(["ctrl" if p == "ctrl" else (p if "+" in p else f"{p}+ctrl") for p in pert])
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.var_names = feats
    a.obs_names = adata.obs_names
    a.obs["pert"] = pert
    a.obs["condition"] = cond
    a.write_h5ad(OUT / f"norman_hvg{n_hvg}.h5ad", compression="gzip")
    kf = KFold(folds, shuffle=True, random_state=seed)
    out = {}
    for k, (tr, te) in enumerate(kf.split(triples)):
        out[k] = dict(train_combos=triples.iloc[tr].AB.tolist(), test_combos=triples.iloc[te].AB.tolist(),
                      test_A=triples.iloc[te].A.tolist(), test_B=triples.iloc[te].B.tolist())
    (OUT / "norman_folds.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {a.shape} and {folds} folds to {OUT}")


if __name__ == "__main__":
    main()
