"""R4 step 3 (pre-registered 2026-10-28): frozen GrnPFN on the A375 CRISPRi × vemurafenib screen (gxd_pf), held out from
all development. Zero shot vs gradient boosting / ridge / sub-additivity trained on all five benchmark screens (x77
features); learning curve vs POLY / RIDGE / RIDGE-SUM with the x79/x80 draws; zero-shot confidence (relative uncertainty).
  python x87_gxd_eval.py --seed 431 [--dataset gxd_pf]   → results/bench/x87_gxd/{curve,confidence}_{ds}_s{seed}.csv"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from common import RESULTS
from grnpfn.model import load_grnpfn, predict
from grnpfn_real import real_folds
from phase3 import SPENT, score
from x60_pfn_design import DEV, pfn_batch
from x77_zeroshot import DS, feats
from x80_baselines import baselines, rec

OUT = RESULTS / "x87_gxd"; GS = (0.15, 0.25, 0.4, 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--dataset", default="gxd_pf"); a = ap.parse_args()
    assert a.seed not in SPENT; OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2); ds, seed = a.dataset, a.seed
    net = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel.pt"), DEV); vnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel_var.pt"), DEV)
    Xs, ys, Ss = [], [], []
    for b, _ in DS:                                                                      # comparators: all five benchmark screens
        f = real_folds(b, "pair", seed, 0)[0]["full"]; s = np.sqrt(np.mean(f["D"] ** 2))
        Xs.append(feats(f["D"], f["mu"], f["pairs"], s)); ys.append((f["R_tr"] / s).ravel()); Ss.append(((f["D"][f["pairs"][:, 0]] + f["D"][f["pairs"][:, 1]]) / s).ravel())
    X, y, S = np.vstack(Xs), np.concatenate(ys), np.concatenate(Ss)
    gbm = HistGradientBoostingRegressor(random_state=0).fit(X, y); rdg = Ridge(alpha=1.0).fit(X, y); gam = float(S @ y / (S @ S))
    recs, conf = [], []
    for fd in real_folds(ds, "pair", seed, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; s = np.sqrt(np.mean(fd["D"] ** 2)); G = fd["D"].shape[1]; obs = fd["R_q_test"]; Dd = dict(zip(fd["singles"], fd["D"]))
        nte, npl = np.asarray(fd["n_pair_q"], float), np.asarray(fd["n_pair_c"], float); Xq = feats(fd["D"], fd["mu"], fd["pq"], s)
        z = {"GrnPFN": predict(net, fd["D"], fd["mu"], np.zeros((0, 2), int), np.zeros((0, G)), fd["pq"], fd["n_single"], np.zeros(0), nte, fd["n_ctrl"], device=DEV),
             "C-GBM": gbm.predict(Xq).reshape(-1, G) * s, "C-ridge": rdg.predict(Xq).reshape(-1, G) * s, "C-gamma": gam * (fd["D"][fd["pq"][:, 0]] + fd["D"][fd["pq"][:, 1]])}
        for m, pr in z.items():
            sl, sr = score(pr, obs, obs.mean(0))
            recs += [dict(dataset=ds, seed=seed, fold=k, g=0.0, draw=0, AB=fd["key"](*t), method=m, cos_pair=np.nan, cos_pair_testc=a_, cos_raw=c_) for t, a_, c_ in zip(te, sl, sr)]
        F, V = pfn_batch(vnet, fd, [(np.zeros((0, 2), int), np.zeros((0, G)), fd["pq"], np.zeros(0), nte)], return_var=True)[0]
        ct = score(F, obs, obs.mean(0))[0]; f = F - F.mean(0)
        conf += [dict(dataset=ds, seed=seed, fold=k, AB=fd["key"](*t), cos_testc=ct[i], trV=float(V[i].sum()), sig=float(f[i] @ f[i]), rel_unc=float(V[i].sum() / (f[i] @ f[i] + 1e-12))) for i, t in enumerate(te)]
        for g in GS:
            for d in range(1 if g == 1.0 else 3):
                sel = np.arange(len(pool)) if g == 1.0 else np.sort(np.random.default_rng([k, d, int(100 * g)]).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
                tr = [pool[i] for i in sel]; Y = fd["R"]["train"][sel]
                pr = predict(net, fd["D"], fd["mu"], fd["pc"][sel], Y, fd["pq"], fd["n_single"], npl[sel], nte, fd["n_ctrl"], device=DEV)
                recs += rec(ds, seed, k, g, d, te, fd["key"], "GrnPFN", pr, obs, Y)
                for m, fn in baselines(Dd, tr, fd["R"]["fit"][sel], fd["R"]["val"][sel], Y, k).items():
                    recs += rec(ds, seed, k, g, d, te, fd["key"], m, fn(te), obs, Y)
        print(ds, seed, "fold", k, flush=True)
    pd.DataFrame(recs).to_csv(OUT / f"curve_{ds}_s{seed}.csv", index=False); pd.DataFrame(conf).to_csv(OUT / f"confidence_{ds}_s{seed}.csv", index=False)


if __name__ == "__main__":
    main()
