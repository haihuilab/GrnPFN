"""P4-H5 calibration of the PFN posterior-predictive variance on real held-out pairs (dev 301): ratio of realised squared
error to predicted variance (1 = calibrated) and rank correlation of predicted vs realised per-pair error, at each g.
  python x65_calib.py --model grnpfn_rel_var   → x60_pfn_design/calib_{model}.csv"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS
from grnpfn_real import real_folds
from grnpfn.model import load_grnpfn
from x60_pfn_design import DEV, OUT, pfn_batch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    net = load_grnpfn(str(RESULTS / "grnpfn" / f"{a.model}.pt"), DEV); rows = []
    for ds in ("norman", "wessels", "sp4_A172", "sp4_T98G"):
        for fd in real_folds(ds, "pair", 301, 0):
            k = fd["k"]; pool, te = fd["tr"], fd["te"]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
            pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te)
            n_pool, n_te = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float)
            for g in (0.15, 0.4, 1.0):
                sel = np.sort(np.random.default_rng(10_000 * k + int(100 * g)).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
                F, V = pfn_batch(net, fd, [(pool_ix[sel], fd["R"]["train"][sel], te_ix, n_pool[sel], n_te)], return_var=True)[0]
                e2 = ((fd["R_q_test"] - F) ** 2).sum(1); v = V.sum(1)
                rows.append(dict(dataset=ds, fold=k, g=g, ratio=float(e2.sum() / v.sum()), spearman=float(spearmanr(v, e2)[0])))
    d = pd.DataFrame(rows); d.to_csv(OUT / f"calib_{a.model}.csv", index=False)
    print(d.groupby(["dataset", "g"])[["ratio", "spearman"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
