"""R1-C (pre-registered 2026-10-28): rescore the deep-model predictions of the other four screens exactly as
x79_confirm.external does for K562 (Δ̂_AB − Δ_A − Δ_B on the input third, benchmark genes, benchmark centering).
  python x82_score.py --seed 431      → results/bench/p82_external/score_s{seed}.csv"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import RESULTS
from grnpfn_real import real_folds
from phase3 import SPENT
from x79_confirm import rec

P = RESULTS / "p82_external"
MODELS = {"wessels": ("gears", "cellflow", "cpa"), "sp4_A172": ("cellflow", "cpa"), "sp4_T98G": ("cellflow", "cpa"), "sp4_U87MG": ("cellflow", "cpa")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args(); assert a.seed not in SPENT
    recs, q = [], []
    for ds, models in MODELS.items():
        for fd in real_folds(ds, "pair", a.seed, 0):
            k = fd["k"]; te_key = [fd["key"](*t) for t in fd["te"]]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
            for m in models:
                z = np.load(P / f"{m}_{ds}_fold{k}.npz", allow_pickle=True)
                if "genes" in z:
                    assert list(z["genes"]) == list(np.load(P / f"cpa_{ds}_fold{k}.npz", allow_pickle=True)["genes"])
                assert z["delta"].shape[1] == 3000
                rows = {"+".join(sorted(str(c).split("+"))): i for i, c in enumerate(z["combos"])}
                have = [j for j, c in enumerate(te_key) if c in rows]; te = [fd["te"][j] for j in have]
                q.append(dict(dataset=ds, fold=k, method=m, queryable=len(have), test=len(te_key)))
                pr = np.stack([z["delta"][rows[te_key[j]]][fd["keep"]] - fd["D"][ix[fd["te"][j][0]]] - fd["D"][ix[fd["te"][j][1]]] for j in have])
                recs += rec(ds, a.seed, k, 1.0, 0, te, fd["key"], m, pr, fd["R_q_test"][have], fd["R"]["train"].mean(0))
        print("scored", ds, a.seed, flush=True)
    d = pd.DataFrame(recs); d.to_csv(P / f"score_s{a.seed}.csv", index=False); pd.DataFrame(q).to_csv(P / f"queryable_s{a.seed}.csv", index=False)
    print(pd.DataFrame(q).groupby(["dataset", "method"])[["queryable", "test"]].sum().to_string())


if __name__ == "__main__":
    main()
