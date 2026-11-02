"""External deep baselines (GEARS, CellFlow, CPA; trained earlier on all cells of the K562 folds, eval/bench/p5_*.py)
rescored with the PFN benchmark's dev-301 cross-fitted interactions: predicted R = predicted pair Δ − (Δ_A + Δ_B) on the
input third, the 256 benchmark genes. Paired against x62's grnpfn_rel at g = 1 (all training pairs; the deep
models were also trained on all training pairs), legacy centring (training-pair mean) and test-mean centring.
  python x78_external.py   → results/bench/x78_external/norman.csv"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import RESULTS
from grnpfn_real import real_folds
from phase3 import score

EXT = RESULTS / "p5_external"; OUT = RESULTS / "x78_external"
METHODS = ["gears", "cellflow", "cellflowres", "cpa", "cpares"]


def main():
    recs = []
    for fd in real_folds("norman", "pair", 301, 0):
        k = fd["k"]; te_key = [fd["key"](*t) for t in fd["te"]]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
        for m in METHODS:
            z = np.load(EXT / f"{m}_fold{k}.npz", allow_pickle=True); cols = fd["keep"]                    # all saved in the hvg3000 gene order
            assert "genes" not in z.files or (z["genes"][cols] == fd["genes"]).all(); rows = {"+".join(sorted(str(c).split("+"))): i for i, c in enumerate(z["combos"])}
            have = [j for j, c in enumerate(te_key) if c in rows]
            pr = np.stack([z["delta"][rows[te_key[j]]][cols] - fd["D"][ix[fd["te"][j][0]]] - fd["D"][ix[fd["te"][j][1]]] for j in have])
            obs = fd["R_q_test"][have]
            sl, sr = score(pr, obs, fd["R"]["train"].mean(0)); st = score(pr, obs, obs.mean(0))[0]
            recs += [dict(dataset="norman", fold=k, AB=te_key[j], method=m, cos_pair=a, cos_raw=b, cos_pair_testc=c) for j, a, b, c in zip(have, sl, sr, st)]
    d = pd.DataFrame(recs); OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / "norman.csv", index=False)
    x = pd.concat([pd.read_csv(f) for f in glob.glob(str(RESULTS / "x62_pfn_curve" / "norman_*.csv"))]).drop_duplicates(["dataset", "fold", "g", "draw", "AB", "method"])
    x = x[(x.g == 1.0) & x.method == "grnpfn_rel"]
    pp = lambda df, m: df[df.method == m].groupby("AB").cos_pair.mean()
    print("K562, all training pairs, legacy centring (pair-specific top-100 cosine; per-pair means):")
    ref = {"PFN grnpfn_rel": pp(x, "grnpfn_rel")}
    for m in METHODS:
        e = pp(d, m); print(f"- {m}: {e.mean():.3f} (n = {len(e)}; raw {d[d.method == m].cos_raw.mean():.3f}; test-mean centring {d[d.method == m].cos_pair_testc.mean():.3f})")
        for lab, r in ref.items():
            dd = (r - e).dropna(); rng = np.random.default_rng(0); bs = [dd.values[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            print(f"  - {lab} − {m}: {dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] P={wilcoxon(dd).pvalue:.2g} n={len(dd)}")
    print("References: " + ", ".join(f"{lab} {r.mean():.3f}" for lab, r in ref.items()))


if __name__ == "__main__":
    main()
