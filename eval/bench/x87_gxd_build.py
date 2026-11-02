"""R4 (pre-registered 2026-10-28): build drug × gene screens in benchmark form → data/bench_cache/{gxd_pf,gxd_go}_hvg3000.h5ad.
obs['pert'] ∈ {'ctrl' (non-targeting + vehicle), GENE (guide + vehicle), DRUG (non-targeting + drug), 'GENE+DRUG'}.
log1p CP10k (library = all retained genes), 3000 Seurat-v3 HVGs on control and single-perturbation cells.
  python x87_gxd_build.py pf <export_dir>          PerturbFate (A375 CRISPRi × vemurafenib; R export of GSE291147)
  python x87_gxd_build.py go <raw_dir>             gastric organoid CRISPRi × cisplatin (GSE280506)"""
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from common import DATA


def finish(X, genes, pert, out):
    a = ad.AnnData(sp.csr_matrix(X, dtype=np.float32), obs=pd.DataFrame({"pert": pert}), var=pd.DataFrame(index=genes))
    a.var_names_make_unique(); a = a[a.obs.pert.notna()].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    ref = a[~a.obs.pert.str.contains("+", regex=False)].copy()
    raw = ref.copy(); raw.X = np.expm1(raw.X)                                   # seurat_v3 wants counts-like values
    sc.pp.highly_variable_genes(raw, n_top_genes=3000, flavor="seurat_v3")
    a = a[:, raw.var.highly_variable.values].copy(); a.write_h5ad(DATA / "bench_cache" / f"{out}_hvg3000.h5ad", compression="gzip")
    vc = a.obs.pert.value_counts(); pairs = [p for p in vc.index if "+" in p]
    ok = [p for p in pairs if vc[p] >= 20 and all(vc.get(x, 0) >= 20 for x in p.split("+"))]
    print(out, a.shape, "ctrl", vc.get("ctrl", 0), "pairs", len(pairs), "usable pairs (≥20 cells pair and singles)", len(ok),
          "median cells pair/gene single", int(np.median([vc[p] for p in ok])) if ok else 0, int(np.median([vc[p.split('+')[0]] for p in ok])) if ok else 0)


def pf(d):
    dim = np.fromfile(f"{d}/dim.bin", dtype=np.int32); ng, nc = int(dim[0]), int(dim[1])
    X = sp.csc_matrix((np.fromfile(f"{d}/x.bin", dtype=np.float32), np.fromfile(f"{d}/i.bin", dtype=np.int32), np.fromfile(f"{d}/p.bin", dtype=np.int32)), shape=(ng, nc)).T.tocsr()
    obs = pd.read_csv(f"{d}/obs.csv"); genes = pd.read_csv(f"{d}/var.csv").id.astype(str).values
    tgt = obs.guide.str.replace(r"_\d+$", "", regex=True); nt = tgt.str.startswith("NO-TARGET"); drug = obs.condition.str.startswith("PLX")
    pert = np.where(nt & ~drug, "ctrl", np.where(nt & drug, "vemurafenib", np.where(~drug, tgt, tgt + "+vemurafenib")))
    finish(X, genes, pert, "gxd_pf")


def go(d):
    a = sc.read_10x_h5(f"{d}/GSE280506_filtered_feature_bc_matrix.h5")
    ids = pd.read_csv(f"{d}/GSE280506_cell_identities.csv.gz")
    ids = ids[(ids.good_coverage == True) & (ids.number_of_cells == 1) & (ids.gemgroup.between(1, 6))]  # noqa: E712
    ids = ids[ids.guide_identity != "*"].set_index("cell_barcode")
    tgt = ids.guide_identity.str.split("_").str[0]; drug = ids.gemgroup >= 4; nt = tgt == "negative"
    lab = pd.Series(np.where(nt & ~drug, "ctrl", np.where(nt & drug, "cisplatin", np.where(~drug, tgt, tgt + "+cisplatin"))), index=ids.index)
    a = a[a.obs_names.isin(lab.index)].copy(); pert = lab.loc[a.obs_names].values
    finish(a.X, a.var_names.values, pert, "gxd_go")


if __name__ == "__main__":
    pf(sys.argv[2]) if sys.argv[1] == "pf" else go(sys.argv[2])
