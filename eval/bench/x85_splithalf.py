"""R2-F (pre-registered 2026-10-28): split-half reproducibility of training interactions, dev split 301.
partial = R^fit vs R^val with the pair's own cells shared (as x71b_diag.py); full = the pair's cells also split into halves
(seeded). Centred cosine over the 256 benchmark genes, centring on each version's training-pair mean, per fold as x71b.
  python x85_splithalf.py   → results/bench/x85_splithalf/splithalf.csv"""
import anndata as ad
import numpy as np
import pandas as pd

from common import DATA, RESULTS
from grnpfn_real import real_folds


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


rows = []
import sys
DSETS = sys.argv[1:] or ["norman", "wessels", "sp4_A172", "sp4_T98G", "sp4_U87MG"]
for ds in DSETS:
    full = ad.read_h5ad(DATA / "bench_cache" / f"{ds}_hvg3000.h5ad"); X = full.X.tocsr(); pert = full.obs["pert"].to_numpy()
    rows_of = pd.Series(np.arange(len(pert))).groupby(pert).apply(np.asarray).to_dict()
    for fd in real_folds(ds, "pair", 301, 0):
        keep = fd["keep"]; key = fd["key"]; mean = lambda r: np.asarray(X[r].mean(0)).ravel()[keep]
        Rf, Rv = fd["R"]["fit"], fd["R"]["val"]
        hf, hv = [], []
        for t in fd["tr"]:
            r = np.random.default_rng([301, 11, sum(map(ord, key(*t)))]).permutation(rows_of[key(*t)]); a, b = r[: len(r) // 2], r[len(r) // 2:]
            mab = mean(rows_of[key(*t)]); hf.append(mean(a) - mab); hv.append(mean(b) - mab)
        Ff, Fv = Rf + np.array(hf), Rv + np.array(hv)                                    # swap the pair mean for its halves
        for lab, A, B in (("partial", Rf, Rv), ("full", Ff, Fv)):
            rep = np.sum(unit(A - A.mean(0)) * unit(B - B.mean(0)), 1)
            rows += [dict(dataset=ds, fold=fd["k"], AB=key(*t), version=lab, rep=x) for t, x in zip(fd["tr"], rep)]
        print(ds, fd["k"], flush=True)
d = pd.DataFrame(rows); (RESULTS / "x85_splithalf").mkdir(parents=True, exist_ok=True); d.to_csv(RESULTS / "x85_splithalf" / ("splithalf.csv" if not sys.argv[1:] else f"splithalf_{'_'.join(DSETS)}.csv"), index=False)
print(d.groupby(["dataset", "version"]).rep.mean().unstack().round(3).to_string())
