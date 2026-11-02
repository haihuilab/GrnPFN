"""Build data/bench_cache/gxe_{line}_hvg3000.h5ad from the sci-Plex-GxE kinome screen (GSE225775, GSM7056149;
McFaline-Figueroa et al. 2024): CRISPRi knockdown of 522 kinases × {vehicle, lapatinib, nintedanib, trametinib,
ZSTK474} at doses 1 and 10 µM in A172 / T98G / U87MG. Input: data/sciplex_gxe/export (CSC dumps from the authors'
preprocessed cds list, see the R export in the methods). Single-guide cells only.
obs['pert'] ∈ {'ctrl' (NTC/random guide, vehicle), kinase (vehicle), drug (NTC, drug), 'drug+kinase'}; obs['dose'] kept.
log1p CP10k (library = all-gene UMI) on 3000 seurat_v3 HVGs per line."""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from common import DATA

E = DATA / "sciplex_gxe" / "export"


def load(line: str) -> ad.AnnData:
    ng, nc = np.fromfile(E / f"{line}_dim.bin", dtype=np.int32)
    i = np.fromfile(E / f"{line}_i.bin", dtype=np.int32); p = np.fromfile(E / f"{line}_p.bin", dtype=np.int32)
    x = np.fromfile(E / f"{line}_x.bin", dtype=np.float32)
    X = sp.csc_matrix((x, i, p), shape=(ng, nc)).T.tocsr()
    obs = pd.read_csv(E / f"{line}_obs.csv")
    var = pd.read_csv(E / "var.csv")
    names = pd.read_csv(DATA / "sciplex_gxe" / "GSM7056149_sciPlexGxE_2_gene.annotations.txt.gz", sep="\t", header=None,
                        index_col=0)[1]
    a = ad.AnnData(X, obs=obs.set_index("cell"), var=pd.DataFrame(index=var.id.map(names).fillna(var.id).astype(str).values))
    a.var_names_make_unique()
    return a


def main(line: str) -> None:
    a = load(line)
    a = a[a.obs.guide_number == 1].copy()
    g = a.obs.gene_id.astype(str); ctrl_guide = g.isin(["NTC", "random"])
    t = a.obs.treatment.astype(str); veh = t == "vehicle"
    a.obs["pert"] = np.where(ctrl_guide & veh, "ctrl", np.where(veh, g, np.where(ctrl_guide, t, t + "+" + g)))
    lib = a.obs["n.umi"].to_numpy(dtype=np.float32)
    sub = np.random.default_rng(0).choice(a.n_obs, min(80000, a.n_obs), replace=False)
    h = a[sub].copy()
    sc.pp.highly_variable_genes(h, n_top_genes=3000, flavor="seurat_v3")
    hv = h.var_names[h.var.highly_variable]
    b = a[:, hv].copy()
    b.X = sp.csr_matrix(b.X.multiply(1e4 / np.maximum(lib, 1)[:, None]).log1p().astype(np.float32))
    out = DATA / "bench_cache" / f"gxe_{line}_hvg3000.h5ad"
    b.obs = b.obs[["pert", "treatment", "dose", "gene_id", "n.umi", "replicate"]].astype(
        {"pert": str, "treatment": str, "gene_id": str, "replicate": str})
    b.write_h5ad(out)
    print(line, b.shape, out, flush=True)


if __name__ == "__main__":
    for ln in sys.argv[1:] or ["A172", "T98G", "U87MG"]:
        main(ln)
