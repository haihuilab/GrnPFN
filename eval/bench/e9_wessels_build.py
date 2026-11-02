"""Build data/bench_cache/wessels_hvg3000.h5ad from GSE213957 (THP-1 CaRPool-seq, Cas13d knockdown pairs).
obs['pert'] ∈ {'ctrl' (NT_NT), gene (gene_NT), 'A+B' (sorted)}; log1p CP10k on 3000 seurat_v3 HVGs."""
from __future__ import annotations

import gzip

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

from common import DATA

W = DATA / "wessels2023"


def main() -> None:
    meta = pd.read_csv(W / "GSE213957_THP1-CaRPool-seq.metadata.tsv.gz", sep="\t", index_col=0)
    mats, names = [], []
    for lane in (1, 2, 3, 4):
        base = W / "extracted" / f"THP1-CaRPool-seq_and_HEK293FTstabRNA.GEXGDO{lane}"
        with gzip.open(f"{base}.matrix.mtx.gz") as f:
            M = scipy.io.mmread(f).T.tocsr()
        feats = pd.read_csv(f"{base}.features.tsv.gz", sep="\t", header=None)
        bcs = pd.read_csv(f"{base}.barcodes.tsv.gz", header=None)[0]
        gex = (feats[2] == "Gene Expression").to_numpy()
        ids = np.array([f"L{lane}_{b}" for b in bcs])
        keep = np.isin(ids, meta.index)
        mats.append(M[keep][:, gex]); names += list(ids[keep])
        genes = feats.loc[gex, 1].to_numpy()
    X = sp.vstack(mats).tocsr().astype(np.float32)
    a = ad.AnnData(X, obs=meta.loc[names].copy(), var=pd.DataFrame(index=pd.Index(genes).astype(str)))
    a.var_names_make_unique()

    def lab(gp: str) -> str:
        g1, g2 = gp.split("_", 1)
        if g1 == "NT" and g2 == "NT":
            return "ctrl"
        if g2 == "NT":
            return g1
        if g1 == "NT":
            return g2
        return "+".join(sorted((g1, g2)))
    a.obs["pert"] = [lab(x) for x in a.obs.GenePair.astype(str)]
    lib = np.asarray(a.X.sum(1)).ravel()
    sub = a[np.random.default_rng(0).choice(a.n_obs, min(20000, a.n_obs), replace=False)].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=3000, flavor="seurat_v3")
    hv = sub.var_names[sub.var.highly_variable.to_numpy()]
    Y = a[:, hv].X.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr(); Y.data = np.log1p(Y.data)
    out = ad.AnnData(Y.astype(np.float32), obs=a.obs[["pert"]].copy(), var=pd.DataFrame(index=hv))
    out.write_h5ad(DATA / "bench_cache" / "wessels_hvg3000.h5ad")
    vc = out.obs.pert.value_counts()
    print(out, "\nctrl", vc.get("ctrl"), "singles", sum("+" not in p and p != "ctrl" for p in vc.index), "pairs", sum("+" in p for p in vc.index))


if __name__ == "__main__":
    main()
