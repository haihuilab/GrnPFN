"""C6 standard baselines (pre-registered in manuscript/notes/autonomous_log.md, 2026-10-27): degree-2 polynomial kernel
ridge (POLY, main), ridge on [Δ_A+Δ_B, Δ_A⊙Δ_B] (RIDGE) and on Δ_A+Δ_B (RIDGE-SUM), with cross-fitted inner
selection; POLY-DESIGN = greedy relative-IMSE under the normalised polynomial kernel. Same folds/draws as x62/x79.
  python x80_baselines.py --part curve --seed 431 --dataset norman
  python x80_baselines.py --part design --seed 431 --dataset norman    (needs the GrnPFN design selections: x79 or x64)
→ results/bench/x80_baselines/{part}_{ds}_s{seed}.csv"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import RESULTS
from benchmark import fit_poly, fit_ridge, inner_splits
from grnpfn_real import real_folds
from phase3 import SPENT, score
from common import tc

OUT = RESULTS / "x80_baselines"
RIDGE_GRID = list(np.logspace(-2, 5, 15)); POLY_A = np.logspace(-3, 3, 7); POLY_G = np.logspace(-2, 1, 7)


def select(fit, grid, tr, Yf, Yv, k):
    """inner selection for any baseline: fit on R^fit, validate on R^val over the repeated inner splits; mean of the
    ranks of the median raw and the mean pair-specific inner-validation cosine"""
    Tfit = dict(zip(tr, Yf)); Rval = dict(zip(tr, Yv)); cen_v = Yv.mean(0); sc = []
    for q in grid:
        r_, p_ = 0.0, 0.0
        for inn in inner_splits(tr, k, False):
            sr, sp_ = [], []
            for itr, iva in inn:
                vp = [tr[i] for i in iva]; P = fit([tr[i] for i in itr], Tfit, q)(vp); Pc = P - P.mean(0)
                sr += [tc(x, Rval[t]) for x, t in zip(P, vp)]; sp_ += [tc(x, Rval[t] - cen_v) for x, t in zip(Pc, vp)]
            r_ += np.median(sr); p_ += np.mean(sp_)
        sc.append((r_, p_))
    rk = (rankdata([-a for a, _ in sc]) + rankdata([-b for _, b in sc])) / 2
    return grid[int(np.argmin(rk))]


def baselines(Dd, tr, Yf, Yv, Yt, k):
    """fitted predictors (callables on pair lists) of RIDGE, RIDGE-SUM and POLY on the measured pairs tr"""
    m = np.mean([Dd[a] @ Dd[a] + Dd[b] @ Dd[b] for a, b in tr]); T = dict(zip(tr, Yt)); out = {}
    for name, prod in (("RIDGE", True), ("RIDGE-SUM", False)):
        a = select(lambda trp, T_, q: fit_ridge(Dd, trp, T_, q, prod), RIDGE_GRID, tr, Yf, Yv, k); out[name] = fit_ridge(Dd, tr, T, a, prod)
    q = select(lambda trp, T_, q: fit_poly(Dd, trp, T_, *q), [(a, g / m) for a in POLY_A for g in POLY_G], tr, Yf, Yv, k)
    out["POLY"] = fit_poly(Dd, tr, T, *q)
    return out


def poly_design(Dd, pool, targets, b):
    """greedy relative-IMSE selection under the normalised polynomial kernel, averaged over 12 (γ, ν) settings"""
    cand = list(pool) + list(targets); n = len(pool); A = np.stack([Dd[a] for a, _ in cand]); B = np.stack([Dd[b] for _, b in cand])
    m = np.mean(np.sum(A[:n] ** 2, 1) + np.sum(B[:n] ** 2, 1)); Sigs, nus = [], []
    for g in (0.01, 0.1, 1.0, 10.0):
        K = 0.5 * ((g / m * (A @ A.T + B @ B.T) + 1) ** 2 + (g / m * (A @ B.T + B @ A.T) + 1) ** 2); d = np.sqrt(np.diag(K))
        for nu in (0.3, 1.0, 3.0):
            Sigs.append(K / np.outer(d, d)); nus.append(nu)
    chosen = []
    while len(chosen) < b:
        gain = sum(np.sum(S[:, :n] ** 2, 0) / (np.diag(S)[:n] + nu) for S, nu in zip(Sigs, nus)); gain[chosen] = -np.inf
        j = int(np.argmax(gain)); chosen.append(j)
        for i, (S, nu) in enumerate(zip(Sigs, nus)):
            s = S[:, j].copy(); Sigs[i] = S - np.outer(s, s) / (s[j] + nu)
    return sorted(chosen)


def rec(ds, seed, k, g, d, te, key, m, pr, obs, Y, scheme=None):
    sl, sr = score(pr, obs, Y.mean(0)); st = score(pr, obs, obs.mean(0))[0]
    r = [dict(dataset=ds, seed=seed, fold=k, g=g, draw=d, AB=key(*t), method=m, cos_pair=a, cos_pair_testc=b, cos_raw=c) for t, a, b, c in zip(te, sl, st, sr)]
    return [dict(x, scheme=scheme) for x in r] if scheme else r


def curve(ds, seed):
    recs = []
    for fd in real_folds(ds, "pair", seed, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; Dd = dict(zip(fd["singles"], fd["D"]))
        for g in (0.15, 0.25, 0.4, 1.0):
            for d in range(1 if g == 1.0 else 3):
                sel = np.arange(len(pool)) if g == 1.0 else np.sort(np.random.default_rng([k, d, int(100 * g)]).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
                tr = [pool[i] for i in sel]; Y = fd["R"]["train"][sel]
                for m, f in baselines(Dd, tr, fd["R"]["fit"][sel], fd["R"]["val"][sel], Y, k).items():
                    recs += rec(ds, seed, k, g, d, te, fd["key"], m, f(te), fd["R_q_test"], Y)
        print("curve", ds, seed, "fold", k, flush=True)
    return pd.DataFrame(recs)


def design(ds, seed):
    src = RESULTS / ("x60_pfn_design" if seed == 301 else "x79_confirm") / (f"{ds}_s301_grnpfn_rel_var.csv" if seed == 301 else f"design_{ds}_s{seed}.csv")
    v = pd.read_csv(src); v = v[(v.scheme == "pfn-vdesign") & (v.g == 0.25)] if seed == 301 else v[(v.scheme == "pfn-vdesign") & (v.method == "PFN")]
    recs = []; g = 0.25
    for fd in real_folds(ds, "pair", seed, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; Dd = dict(zip(fd["singles"], fd["D"])); b = max(10, int(round(g * len(pool))))
        sels = {("pfn-vdesign", 0): np.array(list(map(int, v[v.fold == k].sel.iloc[0].split()))), ("poly-design", 0): np.array(poly_design(Dd, pool, te, b))}
        for d in range(5):
            sels[("random", d)] = np.array(sorted(np.random.default_rng(10_000 * k + 100 * d + int(100 * g)).choice(len(pool), b, replace=False).tolist()))
        for (scheme, d), sel in sels.items():
            assert len(sel) == b; tr = [pool[i] for i in sel]; Y = fd["R"]["train"][sel]
            for m, f in baselines(Dd, tr, fd["R"]["fit"][sel], fd["R"]["val"][sel], Y, k).items():
                recs += rec(ds, seed, k, g, d, te, fd["key"], m, f(te), fd["R_q_test"], Y, scheme)
            if scheme == "poly-design":
                recs[-1]["sel"] = " ".join(map(str, sel.tolist()))
        print("design", ds, seed, "fold", k, flush=True)
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["curve", "design"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", required=True); a = ap.parse_args(); assert a.seed not in SPENT; OUT.mkdir(parents=True, exist_ok=True)
    d = curve(a.dataset, a.seed) if a.part == "curve" else design(a.dataset, a.seed)
    d.to_csv(OUT / f"{a.part}_{a.dataset}_s{a.seed}.csv", index=False)
    print(d.groupby(["g", "method"] + (["scheme"] if a.part == "design" else [])).cos_pair.mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
