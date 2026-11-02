"""Build data/bench_cache/sp4_{line}_hvg3000.h5ad from sci-Plex 4 (GSE225775, GSM7056151; McFaline-Figueroa et al. 2024):
22 compounds at 4 doses, each with or without trametinib (0.01–10 µM), in A172 / T98G / U87MG.
Input: data/sciplex_gxe/export4 (CSC dumps of the authors' preprocessed cds list).
obs['pert'] ∈ {'ctrl' (vehicle, no trametinib), 'X@dose' (compound alone), 'tram@t' (trametinib alone),
'X@dose+tram@t'}; log1p CP10k (library = all-gene UMI) on 3000 seurat_v3 HVGs per line."""
from __future__ import annotations

import sys

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from common import DATA
from e11_gxe_build import load

E = DATA / "sciplex_gxe" / "export4"


def fmt(v: float) -> str:
    return f"{v:g}"


def main(line: str) -> None:
    import e11_gxe_build
    e11_gxe_build.E = E
    a = load(line)
    t = a.obs.treatment.astype(str).to_numpy(); d = a.obs.dose.astype(float).to_numpy(); tr = a.obs.trametinib_dose.astype(float).to_numpy()
    veh = t == "vehicle"
    # trametinib alone is stored as treatment 'Trametinib' at dose d with trametinib_dose = d; compound X at dose level d
    # is co-exposed with trametinib at the same level d
    is_tram = t == "Trametinib"; t = np.where(is_tram, "vehicle", t); veh = t == "vehicle"
    drug = np.array([f"{x}@{fmt(y)}" for x, y in zip(t, d)]); tram = np.array([f"tram@{fmt(y)}" for y in tr])
    a.obs["pert"] = np.where(veh & (tr == 0), "ctrl", np.where(veh, tram, np.where(tr == 0, drug,
                             np.char.add(np.char.add(drug.astype(str), "+"), tram.astype(str)))))
    lib = a.obs["n.umi"].to_numpy(dtype=np.float32)
    sub = np.random.default_rng(0).choice(a.n_obs, min(80000, a.n_obs), replace=False)
    h = a[sub].copy(); sc.pp.highly_variable_genes(h, n_top_genes=3000, flavor="seurat_v3")
    b = a[:, h.var_names[h.var.highly_variable]].copy()
    b.X = sp.csr_matrix(b.X.multiply(1e4 / np.maximum(lib, 1)[:, None]).log1p().astype(np.float32))
    b.obs = b.obs[["pert", "treatment", "dose", "trametinib_dose", "replicate", "n.umi"]].astype({"pert": str, "treatment": str, "replicate": str})
    out = DATA / "bench_cache" / f"sp4_{line}_hvg3000.h5ad"; b.write_h5ad(out)
    vc = b.obs.pert.value_counts()
    print(line, b.shape, "conditions", len(vc), "pairs", sum("+" in p for p in vc.index), "median cells/pair",
          int(vc[[p for p in vc.index if "+" in p]].median()), "ctrl", vc.get("ctrl", 0), flush=True)


if __name__ == "__main__":
    for ln in sys.argv[1:] or ["A172", "T98G", "U87MG"]:
        main(ln)
