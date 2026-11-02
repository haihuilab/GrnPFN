"""Melanoma knockout × context screen (Frangieh et al. 2021, scPerturb release; data/bench_cache/frangieh_hvg3000.h5ad) in
pair form for the design analyses (manuscript/notes/autonomous_log.md): NT guide in the unstimulated condition = control;
KO in the unstimulated condition = single KO; NT in IFN-γ or co-culture = single context (ctx_IFN, ctx_CC); KO in a
stimulated condition = the pair "KO + context".   python frangieh_pairs_build.py → data/bench_cache/frangieh_pairs_hvg3000.h5ad"""
from __future__ import annotations

import anndata as ad
import numpy as np

from common import DATA

CTX = {"ifn": "ctx_IFN", "cc": "ctx_CC"}


def main():
    a = ad.read_h5ad(DATA / "bench_cache" / "frangieh_hvg3000.h5ad"); ko, cx = a.obs["ko"].astype(str).to_numpy(), a.obs["context"].astype(str).to_numpy()
    lab = np.where(ko == "NT", np.where(cx == "ctrl", "ctrl", [CTX.get(c, c) for c in cx]),
                   np.where(cx == "ctrl", ko, ["+".join(sorted((k, CTX.get(c, c)))) for k, c in zip(ko, cx)]))
    a.obs["pert"] = lab; a.write_h5ad(DATA / "bench_cache" / "frangieh_pairs_hvg3000.h5ad")
    vc = a.obs["pert"].value_counts(); pairs = [p for p in vc.index if "+" in p]
    print(f"cells {a.n_obs}; control {vc['ctrl']}; contexts {vc.get('ctx_IFN')}/{vc.get('ctx_CC')}; pairs {len(pairs)} (≥20 cells: {(vc[pairs] >= 20).sum()})")


if __name__ == "__main__":
    main()
