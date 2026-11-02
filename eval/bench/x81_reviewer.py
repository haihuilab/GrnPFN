"""R1 reviewer controls (pre-registered 2026-10-28 in manuscript/notes/autonomous_log.md; secondary, descriptive, no
decision criterion).
  python x81_reviewer.py --part select --seed 431 --dataset norman   R1-A: one-shot selection rules from GrnPFN without
        context ("pred-norm": largest predicted centered interaction; "pred-var": largest predictive variance), g = .25,
        scored with GrnPFN, POLY, RIDGE, RIDGE-SUM exactly as x79/x80 design; selection properties as Fig. 4c
  python x81_reviewer.py --part calib --seed 431                     R1-D: variance-head calibration at g = 0/.15/.40/1
        (x65_calib definitions and draws), all five screens
→ results/bench/x81_reviewer/{part}[_{ds}]_s{seed}.csv"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import RESULTS
from grnpfn.model import load_grnpfn
from grnpfn_real import real_folds
from phase3 import SPENT
from x60_pfn_design import DEV, pfn_batch
from x77_zeroshot import DS
from x80_baselines import baselines, rec

OUT = RESULTS / "x81_reviewer"


def setup(fd):
    ix = {gn: i for i, gn in enumerate(fd["singles"])}; pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps])
    pool_ix, te_ix = pidx(fd["tr"]), pidx(fd["te"]); n_pool, n_te = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float)
    return pool_ix, te_ix, np.vstack([pool_ix, te_ix]), n_pool, n_te, np.r_[n_pool, n_te]


def select(seed, ds):
    dnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel_var.pt"), DEV); pnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel.pt"), DEV)
    recs, props = [], []; g = 0.25
    for fd in real_folds(ds, "pair", seed, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; Dd = dict(zip(fd["singles"], fd["D"])); G = fd["D"].shape[1]
        pool_ix, te_ix, cand_ix, n_pool, n_te, n_cand = setup(fd); n = len(pool); b = max(10, int(round(g * n)))
        F, V = pfn_batch(dnet, fd, [(np.zeros((0, 2), int), np.zeros((0, G)), cand_ix, np.zeros(0), n_cand)], return_var=True)[0]
        nrm = np.linalg.norm(F - F.mean(0), axis=1)[:n]; var = V.sum(1)[:n]
        sels = {"pred-norm": np.sort(np.argsort(-nrm, kind="stable")[:b]), "pred-var": np.sort(np.argsort(-var, kind="stable")[:b])}
        Rn = np.linalg.norm(fd["R"]["train"] - fd["R"]["train"].mean(0), axis=1); top = Rn >= np.quantile(Rn, 0.8)
        for scheme, sel in sels.items():
            assert len(sel) == b; tr = [pool[i] for i in sel]; Y = fd["R"]["train"][sel]
            props.append(dict(dataset=ds, seed=seed, fold=k, scheme=scheme, top20=top[sel].sum() / top.sum(), R_norm=Rn[sel].mean() / Rn.mean(), sel=" ".join(map(str, sel.tolist()))))
            pr = pfn_batch(pnet, fd, [(pool_ix[sel], Y, te_ix, n_pool[sel], n_te)])[0]
            recs += rec(ds, seed, k, g, 0, te, fd["key"], "GrnPFN", pr, fd["R_q_test"], Y, scheme)
            for m, f in baselines(Dd, tr, fd["R"]["fit"][sel], fd["R"]["val"][sel], Y, k).items():
                recs += rec(ds, seed, k, g, 0, te, fd["key"], m, f(te), fd["R_q_test"], Y, scheme)
        print("select", ds, seed, "fold", k, flush=True)
    pd.DataFrame(props).to_csv(OUT / f"selprops_{ds}_s{seed}.csv", index=False)
    return pd.DataFrame(recs)


def calib(seed):
    net = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel_var.pt"), DEV); rows = []
    for ds, _ in DS:
        for fd in real_folds(ds, "pair", seed, 0):
            k = fd["k"]; pool = fd["tr"]; G = fd["D"].shape[1]; pool_ix, te_ix, _, n_pool, n_te, _ = setup(fd)
            for g in (0.0, 0.15, 0.4, 1.0):
                sel = np.zeros(0, int) if g == 0 else np.sort(np.random.default_rng(10_000 * k + int(100 * g)).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
                F, V = pfn_batch(net, fd, [(pool_ix[sel], fd["R"]["train"][sel].reshape(len(sel), G), te_ix, n_pool[sel], n_te)], return_var=True)[0]
                e2 = ((fd["R_q_test"] - F) ** 2).sum(1); v = V.sum(1)
                rows.append(dict(dataset=ds, seed=seed, fold=k, g=g, ratio=float(e2.sum() / v.sum()), spearman=float(spearmanr(v, e2)[0])))
        print("calib", ds, seed, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["select", "calib"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset"); a = ap.parse_args(); assert a.seed not in SPENT; torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    d = select(a.seed, a.dataset) if a.part == "select" else calib(a.seed)
    tag = f"select_{a.dataset}_s{a.seed}" if a.part == "select" else f"calib_s{a.seed}"
    d.to_csv(OUT / f"{tag}.csv", index=False)
    print(d.groupby(["dataset", "method", "scheme"]).cos_pair.mean().round(4).to_string() if a.part == "select" else d.groupby(["dataset", "g"])[["ratio", "spearman"]].mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
