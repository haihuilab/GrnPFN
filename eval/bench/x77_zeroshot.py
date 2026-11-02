"""Z0 zero-shot check, pre-registered in manuscript/notes/autonomous_log.md (2026-10-27). With NO measured pair of the
target screen: frozen PFN (grnpfn_rel, empty context) vs leave-one-screen-out predictors trained on the other real
screens (gene-local gradient boosting, ridge, global γ). Reference with pairs: PFN at g = .15 (x62 draws).
Primary metric: pair-specific top-100 cosine with observed test interactions centred on the test-pair mean.
  python x77_zeroshot.py        → results/bench/x77_zeroshot/z0.csv and the pre-registered report"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from common import RESULTS
from grnpfn.model import load_grnpfn, predict
from grnpfn_real import real_folds
from phase3 import score
from x76_report import pooled

OUT = RESULTS / "x77_zeroshot"
DS = [("norman", "K562"), ("wessels", "THP-1"), ("sp4_A172", "A172"), ("sp4_T98G", "T98G"), ("sp4_U87MG", "U87MG")]


def feats(D, mu, pairs, s):
    a, b = D[pairs[:, 0]] / s, D[pairs[:, 1]] / s; m = np.broadcast_to(np.log1p(np.maximum(mu, 0)), a.shape)
    return np.stack([a + b, a * b, np.abs(a - b), np.minimum(a, b), np.maximum(a, b), m], -1).reshape(-1, 6)


def main():
    torch.set_num_threads(4); folds = {ds: real_folds(ds, "pair", 301, 0) for ds, _ in DS}
    nets = {m: load_grnpfn(str(RESULTS / "grnpfn" / f"{m}.pt"), "cpu") for m in ("grnpfn_rel", "grnpfn_v1")}
    train = {}
    for ds, _ in DS:                                                                     # all pairs, training-third interactions
        f = folds[ds][0]["full"]; s = np.sqrt(np.mean(f["D"] ** 2))
        train[ds] = (feats(f["D"], f["mu"], f["pairs"], s), (f["R_tr"] / s).ravel(), ((f["D"][f["pairs"][:, 0]] + f["D"][f["pairs"][:, 1]]) / s).ravel())
    recs = []
    for ds, name in DS:
        oth = [o for o, _ in DS if o != ds]
        X = np.vstack([train[o][0] for o in oth]); y = np.concatenate([train[o][1] for o in oth]); S = np.concatenate([train[o][2] for o in oth])
        gbm = HistGradientBoostingRegressor(random_state=0).fit(X, y); rdg = Ridge(alpha=1.0).fit(X, y); gam = float(S @ y / (S @ S))
        print(ds, "LOSO fitted; γ =", round(gam, 4), flush=True)
        for fd in folds[ds]:
            k = fd["k"]; pool, te = fd["tr"], fd["te"]; s = np.sqrt(np.mean(fd["D"] ** 2)); G = fd["D"].shape[1]; obs = fd["R_q_test"]
            Xq = feats(fd["D"], fd["mu"], fd["pq"], s); nte = np.asarray(fd["n_pair_q"], float); npl = np.asarray(fd["n_pair_c"], float)
            preds = {(m + " [empty]", 0.0, 0): predict(net, fd["D"], fd["mu"], np.zeros((0, 2), int), np.zeros((0, G)), fd["pq"], fd["n_single"], np.zeros(0), nte, fd["n_ctrl"])
                     for m, net in nets.items()}
            preds[("C-GBM", 0.0, 0)] = gbm.predict(Xq).reshape(-1, G) * s
            preds[("C-ridge", 0.0, 0)] = rdg.predict(Xq).reshape(-1, G) * s
            preds[("C-gamma", 0.0, 0)] = gam * (fd["D"][fd["pq"][:, 0]] + fd["D"][fd["pq"][:, 1]])
            centres = {}
            for d in range(3):                                                          # x62's g = .15 draws
                sel = np.sort(np.random.default_rng([k, d, 15]).choice(len(pool), max(10, int(round(0.15 * len(pool)))), replace=False))
                Y = fd["R"]["train"][sel]; centres[d] = Y.mean(0)
                preds[("grnpfn_rel", 0.15, d)] = predict(nets["grnpfn_rel"], fd["D"], fd["mu"], fd["pc"][sel], Y, fd["pq"], fd["n_single"], npl[sel], nte, fd["n_ctrl"])
            for (m, g, d), pr in preds.items():
                sp, sr = score(pr, obs, obs.mean(0)); sl = score(pr, obs, centres[d])[0] if g > 0 else [np.nan] * len(obs)
                recs += [dict(dataset=ds, fold=k, g=g, draw=d, AB=fd["key"](*t), method=m, cos_pair=x, cos_raw=y, cos_pair_legacy=z) for t, x, y, z in zip(te, sp, sr, sl)]
            print(ds, "fold", k, flush=True)
    OUT.mkdir(parents=True, exist_ok=True); d = pd.DataFrame(recs); d.to_csv(OUT / "z0.csv", index=False); report(d)


def report(d):
    pp = lambda ds, m, col="cos_pair": d[(d.dataset == ds) & (d.method == m)].groupby("AB")[col].mean()
    print("\n# Z0 zero-shot check (dev 301; pre-registered)\n")
    print(d.groupby(["dataset", "method"])[["cos_pair", "cos_raw", "cos_pair_legacy"]].mean().round(3).to_string(), "\n")
    for a, b, lab in (("grnpfn_rel [empty]", "C-GBM", "PRIMARY: PFN(empty) − C-GBM"), ("grnpfn_rel [empty]", "C-ridge", "PFN(empty) − C-ridge"),
                      ("grnpfn_rel [empty]", "C-gamma", "PFN(empty) − C-γ"),
                      ("grnpfn_v1 [empty]", "grnpfn_rel [empty]", "v1(empty) − rel(empty)")):
        for col in ("cos_pair", "cos_raw"):
            diffs = [(pp(ds, a, col) - pp(ds, b, col)).dropna() for ds, _ in DS]; m, lo, hi = pooled([x.values for x in diffs]); npos = sum(x.mean() > 0 for x in diffs)
            v = f"  → {'PASS' if (m > 0 and lo > 0 and npos >= 4) else 'FAIL'}" if lab.startswith("PRIMARY") and col == "cos_pair" else ""
            print(f"- {lab} [{col}]: pooled {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; screens > 0: {npos}/5{v}")
            print("  - " + "; ".join(f"{n} {x.mean():+.4f}" for (_, n), x in zip(DS, diffs)))


if __name__ == "__main__":
    main()
