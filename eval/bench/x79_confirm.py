"""C5 confirmation of the frozen PFN on fresh cell-split seeds (pre-registered in manuscript/notes/autonomous_log.md,
2026-10-27). Reuses the frozen functions of x77 (zero-shot comparators), x64 (uncertainty-aware design) and x62 draws.
  python x79_confirm.py --part curve --seed 431              all 5 screens: PFN at g = .15/.25/.40/1, PFN empty, LOSO
  python x79_confirm.py --part design --seed 431 --dataset norman    g = .25: PFN vdesign, random × 5
  python x79_confirm.py --part external --seed 431           K562: GEARS/CellFlow/CPA rescored under the seed
→ results/bench/x79_confirm/{part}[_{ds}]_s{seed}.csv"""
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
from x64_pfn_vdesign import pfn_vdesign
from x77_zeroshot import DS, feats

OUT = RESULTS / "x79_confirm"; GS = (0.15, 0.25, 0.4, 1.0)


def rec(ds, seed, k, g, d, te, key, m, pr, obs, centre):
    sl, sr = score(pr, obs, centre) if centre is not None else ([np.nan] * len(obs), score(pr, obs, obs.mean(0))[1])
    st = score(pr, obs, obs.mean(0))[0]
    return [dict(dataset=ds, seed=seed, fold=k, g=g, draw=d, AB=key(*t), method=m, cos_pair=a, cos_pair_testc=b, cos_raw=c) for t, a, b, c in zip(te, sl, st, sr)]


def curve(seed):
    folds = {ds: real_folds(ds, "pair", seed, 0) for ds, _ in DS}; net = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel.pt"), DEV)
    train = {}
    for ds, _ in DS:
        f = folds[ds][0]["full"]; s = np.sqrt(np.mean(f["D"] ** 2))
        train[ds] = (feats(f["D"], f["mu"], f["pairs"], s), (f["R_tr"] / s).ravel(), ((f["D"][f["pairs"][:, 0]] + f["D"][f["pairs"][:, 1]]) / s).ravel())
    recs = []
    for ds, _ in DS:
        oth = [o for o, _ in DS if o != ds]
        X = np.vstack([train[o][0] for o in oth]); y = np.concatenate([train[o][1] for o in oth]); S = np.concatenate([train[o][2] for o in oth])
        gbm = HistGradientBoostingRegressor(random_state=0).fit(X, y); rdg = Ridge(alpha=1.0).fit(X, y); gam = float(S @ y / (S @ S))
        for fd in folds[ds]:
            k = fd["k"]; pool, te = fd["tr"], fd["te"]; s = np.sqrt(np.mean(fd["D"] ** 2)); G = fd["D"].shape[1]; obs = fd["R_q_test"]
            nte, npl = np.asarray(fd["n_pair_q"], float), np.asarray(fd["n_pair_c"], float); Xq = feats(fd["D"], fd["mu"], fd["pq"], s)
            z = {"PFN [empty]": predict(net, fd["D"], fd["mu"], np.zeros((0, 2), int), np.zeros((0, G)), fd["pq"], fd["n_single"], np.zeros(0), nte, fd["n_ctrl"], device=DEV),
                 "C-GBM": gbm.predict(Xq).reshape(-1, G) * s, "C-ridge": rdg.predict(Xq).reshape(-1, G) * s,
                 "C-gamma": gam * (fd["D"][fd["pq"][:, 0]] + fd["D"][fd["pq"][:, 1]])}
            for m, pr in z.items():
                recs += rec(ds, seed, k, 0.0, 0, te, fd["key"], m, pr, obs, None)
            for g in GS:
                for d in range(1 if g == 1.0 else 3):
                    sel = np.arange(len(pool)) if g == 1.0 else np.sort(np.random.default_rng([k, d, int(100 * g)]).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
                    Y = fd["R"]["train"][sel]
                    pr = predict(net, fd["D"], fd["mu"], fd["pc"][sel], Y, fd["pq"], fd["n_single"], npl[sel], nte, fd["n_ctrl"], device=DEV)
                    recs += rec(ds, seed, k, g, d, te, fd["key"], "PFN", pr, obs, Y.mean(0))
            print("curve", seed, ds, "fold", k, flush=True)
    return pd.DataFrame(recs)


def design(seed, ds):
    dnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel_var.pt"), DEV); pnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel.pt"), DEV); recs = []
    for fd in real_folds(ds, "pair", seed, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
        pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te); cand_ix = np.vstack([pool_ix, te_ix])
        n_pool, n_te = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float); n_cand = np.r_[n_pool, n_te]; g = 0.25
        b = max(10, int(round(g * len(pool))))
        sels = {("pfn-vdesign", 0): sorted(pfn_vdesign(dnet, fd, pool_ix, cand_ix, n_pool, n_cand, b, np.random.default_rng(k)))}
        for d in range(5):
            sels[("random", d)] = sorted(np.random.default_rng(10_000 * k + 100 * d + int(100 * g)).choice(len(pool), b, replace=False).tolist())
        for (scheme, d), sel in sels.items():
            sel = np.asarray(sel); Yt = fd["R"]["train"][sel]
            pr = pfn_batch(pnet, fd, [(pool_ix[sel], Yt, te_ix, n_pool[sel], n_te)])[0]
            recs += [dict(r, scheme=scheme, sel=" ".join(map(str, sel.tolist()))) for r in rec(ds, seed, k, g, d, te, fd["key"], "PFN", pr, fd["R_q_test"], Yt.mean(0))]
        print("design", seed, ds, "fold", k, flush=True)
    return pd.DataFrame(recs)


def external(seed):
    recs = []
    for fd in real_folds("norman", "pair", seed, 0):
        k = fd["k"]; te_key = [fd["key"](*t) for t in fd["te"]]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
        for m in ("gears", "cellflow", "cpa"):
            z = np.load(RESULTS / "p5_external" / f"{m}_fold{k}.npz", allow_pickle=True); rows = {"+".join(sorted(str(c).split("+"))): i for i, c in enumerate(z["combos"])}
            have = [j for j, c in enumerate(te_key) if c in rows]; te = [fd["te"][j] for j in have]
            pr = np.stack([z["delta"][rows[te_key[j]]][fd["keep"]] - fd["D"][ix[fd["te"][j][0]]] - fd["D"][ix[fd["te"][j][1]]] for j in have])
            recs += rec("norman", seed, k, 1.0, 0, te, fd["key"], m, pr, fd["R_q_test"][have], fd["R"]["train"].mean(0))
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["curve", "design", "external"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset"); a = ap.parse_args(); assert a.seed not in SPENT; torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    d = curve(a.seed) if a.part == "curve" else design(a.seed, a.dataset) if a.part == "design" else external(a.seed)
    tag = f"{a.part}_{a.dataset}_s{a.seed}" if a.part == "design" else f"{a.part}_s{a.seed}"
    d.to_csv(OUT / f"{tag}.csv", index=False); print(d.groupby(["dataset", "g", "method"] + (["scheme"] if "scheme" in d else [])).cos_pair.mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
