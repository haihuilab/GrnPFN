"""R2 fixed-target rescoring (pre-registered 2026-10-28 in manuscript/notes/autonomous_log.md; descriptive, no decision
criterion). Two phases so that only one process holds the GPU:
  python x83_fixedtarget.py --part gpu --seed 431            selections + GrnPFN predictions for all five screens → gpu_{ds}_s{seed}.pkl
  python x83_fixedtarget.py --part cpu --seed 431 --dataset norman   baselines on every selection/draw, all metrics → {design,curve}_{ds}_s{seed}.csv
Selectors (g = .25): GrnPFN look-ahead (saved selections), random × 5, kernel design, single-norm, pred-norm, pred-var,
pred-norm-div. Curve (fresh seeds only): x79/x80 draws at g = .15/.25/.40/1 and the zero-shot comparators at g = 0.
Metrics: cos_reg (registered), cos_fix (observed centred on ALL training pairs of the fold), cos_testc, cos_raw,
cos256_fix, nse (uncentred squared error / fold mean ‖R‖²)."""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from common import RESULTS
from grnpfn.model import load_grnpfn
from grnpfn_real import real_folds
from phase3 import SPENT
from common import tc
from x60_pfn_design import DEV, pfn_batch
from x77_zeroshot import DS, feats
from x80_baselines import baselines, poly_design

OUT = RESULTS / "x83_fixedtarget"; G25 = 0.25; GS = (0.15, 0.25, 0.4, 1.0)


def setup(fd):
    ix = {gn: i for i, gn in enumerate(fd["singles"])}; pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps])
    pool_ix, te_ix = pidx(fd["tr"]), pidx(fd["te"]); n_pool, n_te = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float)
    return pool_ix, te_ix, np.vstack([pool_ix, te_ix]), n_pool, n_te, np.r_[n_pool, n_te]


def vdesign_sel(seed, ds):
    if seed == 301:
        v = pd.read_csv(RESULTS / "x60_pfn_design" / f"{ds}_s301_grnpfn_rel_var.csv"); v = v[(v.scheme == "pfn-vdesign") & (v.g == 0.25)]
    else:
        v = pd.read_csv(RESULTS / "x79_confirm" / f"design_{ds}_s{seed}.csv"); v = v[(v.scheme == "pfn-vdesign") & (v.method == "PFN")]
    return {k: np.array(list(map(int, v[v.fold == k].sel.iloc[0].split()))) for k in sorted(v.fold.unique())}


def norm_div(f, b):
    """magnitude-plus-diversity: argmax ‖f_j‖ · (1 − max_{s∈S} max(0, cos(f_j, f_s))), first pick = largest ‖f_j‖"""
    m = np.linalg.norm(f, axis=1); u = f / (m[:, None] + 1e-12); S = []; red = np.zeros(len(f))
    while len(S) < b:
        sc = m * (1 - red); sc[S] = -np.inf; j = int(np.argmax(sc)); S.append(j); red = np.maximum(red, np.maximum(u @ u[j], 0))
    return np.sort(np.array(S))


def gpu(seed):
    dnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel_var.pt"), DEV); pnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel.pt"), DEV)
    folds = {ds: real_folds(ds, "pair", seed, 0) for ds, _ in DS}; zs = {}
    if seed != 301:                                                                     # x79 zero-shot comparators (identical code)
        train = {}
        for ds, _ in DS:
            f = folds[ds][0]["full"]; s = np.sqrt(np.mean(f["D"] ** 2))
            train[ds] = (feats(f["D"], f["mu"], f["pairs"], s), (f["R_tr"] / s).ravel(), ((f["D"][f["pairs"][:, 0]] + f["D"][f["pairs"][:, 1]]) / s).ravel())
        for ds, _ in DS:
            oth = [o for o, _ in DS if o != ds]; X = np.vstack([train[o][0] for o in oth]); y = np.concatenate([train[o][1] for o in oth])
            S_ = np.concatenate([train[o][2] for o in oth])
            zs[ds] = (HistGradientBoostingRegressor(random_state=0).fit(X, y), Ridge(alpha=1.0).fit(X, y), float(S_ @ y / (S_ @ S_)))
    for ds, _ in DS:
        vs = vdesign_sel(seed, ds); out = []
        for fd in folds[ds]:
            k = fd["k"]; pool, te = fd["tr"], fd["te"]; Dd = dict(zip(fd["singles"], fd["D"])); G = fd["D"].shape[1]; n = len(pool)
            pool_ix, te_ix, cand_ix, n_pool, n_te, n_cand = setup(fd); b = max(10, int(round(G25 * n)))
            F, V = pfn_batch(dnet, fd, [(np.zeros((0, 2), int), np.zeros((0, G)), cand_ix, np.zeros(0), n_cand)], return_var=True)[0]
            f = (F - F.mean(0))[:n]; nrm = np.linalg.norm(f, axis=1); var = V.sum(1)[:n]
            sn = np.array([np.linalg.norm(Dd[a]) + np.linalg.norm(Dd[c]) for a, c in pool])
            sels = {("pfn-vdesign", 0): vs[k], ("poly-design", 0): np.array(poly_design(Dd, pool, te, b)),
                    ("single-norm", 0): np.sort(np.argsort(-sn, kind="stable")[:b]), ("pred-norm", 0): np.sort(np.argsort(-nrm, kind="stable")[:b]),
                    ("pred-var", 0): np.sort(np.argsort(-var, kind="stable")[:b]), ("pred-norm-div", 0): norm_div(f, b)}
            for d in range(5):
                sels[("random", d)] = np.array(sorted(np.random.default_rng(10_000 * k + 100 * d + int(100 * G25)).choice(n, b, replace=False).tolist()))
            ctx = {("design",) + key: sel for key, sel in sels.items()}
            if seed != 301:
                for g in GS:
                    for d in range(1 if g == 1.0 else 3):
                        ctx[("curve", f"g{g}", d)] = np.arange(n) if g == 1.0 else np.sort(np.random.default_rng([k, d, int(100 * g)]).choice(n, max(10, int(round(g * n))), replace=False))
            pr = {}
            for key, sel in ctx.items():
                assert len(sel) == (b if key[0] == "design" else len(sel)); Y = fd["R"]["train"][sel]
                pr[key] = pfn_batch(pnet, fd, [(pool_ix[sel], Y, te_ix, n_pool[sel], n_te)])[0]
            zero = {}
            if seed != 301:
                s = np.sqrt(np.mean(fd["D"] ** 2)); Xq = feats(fd["D"], fd["mu"], fd["pq"], s); gbm, rdg, gam = zs[ds]
                zero = {"GrnPFN": pfn_batch(pnet, fd, [(np.zeros((0, 2), int), np.zeros((0, G)), te_ix, np.zeros(0), n_te)])[0],
                        "C-GBM": gbm.predict(Xq).reshape(-1, G) * s, "C-ridge": rdg.predict(Xq).reshape(-1, G) * s,
                        "C-gamma": gam * (fd["D"][fd["pq"][:, 0]] + fd["D"][fd["pq"][:, 1]])}
            out.append(dict(k=k, ctx=ctx, pfn=pr, zero=zero, props=dict(nrm=nrm, var=var, sn=sn)))
            print("gpu", ds, seed, "fold", k, flush=True)
        with open(OUT / f"gpu_{ds}_s{seed}.pkl", "wb") as fh:
            pickle.dump(out, fh)


def metrics(pred, obs, c_reg, c_pool):
    """per test pair: registered, fixed-target, test-centred, raw, all-gene fixed-target cosines and normalised SE"""
    pc = pred - pred.mean(0); c_te = obs.mean(0); den = np.mean(np.sum(obs ** 2, 1))
    cos256 = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
    return [dict(cos_reg=(tc(pc[i], obs[i] - c_reg) if c_reg is not None else np.nan), cos_fix=tc(pc[i], obs[i] - c_pool), cos_testc=tc(pc[i], obs[i] - c_te),
                 cos_raw=tc(pred[i], obs[i]), cos256_fix=cos256(pc[i], obs[i] - c_pool), nse=float(np.sum((pred[i] - obs[i]) ** 2) / den))
            for i in range(len(obs))]


def cpu(seed, ds):
    with open(OUT / f"gpu_{ds}_s{seed}.pkl", "rb") as fh:
        gp = {r["k"]: r for r in pickle.load(fh)}
    recs = {"design": [], "curve": []}
    for fd in real_folds(ds, "pair", seed, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; Dd = dict(zip(fd["singles"], fd["D"])); obs = fd["R_q_test"]; c_pool = fd["R"]["train"].mean(0)
        R = gp[k]; keys = [fd["key"](*t) for t in te]
        Rn = np.linalg.norm(fd["R"]["train"] - c_pool, axis=1); top = Rn >= np.quantile(Rn, 0.8)
        for m, pr in R["zero"].items():
            recs["curve"] += [dict(dataset=ds, seed=seed, fold=k, g=0.0, draw=0, AB=a, method=m, **x) for a, x in zip(keys, metrics(pr, obs, None, c_pool))]
        for key, sel in R["ctx"].items():
            part = key[0]; tr = [pool[i] for i in sel]; Y = fd["R"]["train"][sel]; c_reg = Y.mean(0)
            preds = {"GrnPFN": R["pfn"][key]}
            preds.update({m: f(te) for m, f in baselines(Dd, tr, fd["R"]["fit"][sel], fd["R"]["val"][sel], Y, k).items()})
            base = dict(dataset=ds, seed=seed, fold=k)
            if part == "design":
                base.update(g=G25, scheme=key[1], draw=key[2], top20=float(top[sel].sum() / top.sum()), R_norm=float(Rn[sel].mean() / Rn.mean()), sel=" ".join(map(str, sel.tolist())))
            else:
                base.update(g=float(key[1][1:]), draw=key[2])
            for m, pr in preds.items():
                recs[part] += [dict(base, AB=a, method=m, **x) for a, x in zip(keys, metrics(pr, obs, c_reg, c_pool))]
        print("cpu", ds, seed, "fold", k, flush=True)
    for part, r in recs.items():
        if r:
            pd.DataFrame(r).to_csv(OUT / f"{part}_{ds}_s{seed}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["gpu", "cpu"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset"); a = ap.parse_args(); assert a.seed not in SPENT; OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "gpu":
        torch.set_num_threads(4); gpu(a.seed)
    else:
        torch.set_num_threads(1); cpu(a.seed, a.dataset)


if __name__ == "__main__":
    main()
