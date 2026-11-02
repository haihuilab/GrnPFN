"""Cross-fitted re-evaluation of knockout × context prediction on Frangieh (fixes a shared-noise artifact).

Artifact: R = Δ_KO|t − Δ_KO|ctrl and the input Δ_KO|ctrl were computed from the same control-context cells, so
any predictor ∝ −Δ_KO|ctrl scored well from shared sampling noise ('effect vanishes' scored 0.90 > ceiling).
Fix: every population's control-context cells are split in two halves (fixed seed). Inputs (Δ_KO|ctrl and the OT
cell term) use half 1; the target's control term uses half 2. Training knockouts are cross-fitted the same way.
Methods: vanish (−Δ), ridge, OTOP (fused, inner CV), OTOP NT-only, NT random-coupling field. Learning curve sizes.
"""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from common import DATA, RESULTS, cosine

CACHE = DATA / "bench_cache" / "frangieh_hvg3000.h5ad"
OUT = RESULTS / "e8_crossfit"


def pairs(z0, z1, n, rng, how="ot"):
    i0 = rng.choice(len(z0), min(n, len(z0)), replace=False)
    i1 = rng.choice(len(z1), min(n, len(z1)), replace=False)
    m = min(len(i0), len(i1)); i0, i1 = i0[:m], i1[:m]
    if how == "ot":
        r, c = linear_sum_assignment(((z0[i0, None] - z1[None, i1]) ** 2).sum(-1))
    else:
        r, c = np.arange(m), rng.permutation(m)
    z, v = z0[i0[r]], z1[i1[c]] - z0[i0[r]]
    return z - z.mean(0), v - v.mean(0)


def fuse(cs, pops, D, R, kos, gamma, alpha):
    N = sum(cs[p][2] for p in pops)
    A = sum(cs[p][0] for p in pops) / N; B = sum(cs[p][1] for p in pops) / N
    if kos and gamma > 0:
        Dm = np.stack([D[k] for k in kos]); Rm = np.stack([R[k] for k in kos])
        A = A + gamma * Dm.T @ Dm / len(kos); B = B + gamma * Rm.T @ Dm / len(kos)
    return B @ np.linalg.inv(A + alpha * np.trace(A) / len(A) * np.eye(len(A)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,50,0")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(CACHE); X = a.X.tocsr()
    ko, ctx = a.obs.ko.to_numpy(), a.obs.context.to_numpy()
    kos = sorted(set(ko) - {"NT"})
    idx = pd.Series(np.arange(len(ko))).groupby([ko, ctx]).apply(np.asarray).to_dict()
    split = np.random.default_rng(123)
    h1, h2 = {}, {}
    for g in kos + ["NT"]:
        r = split.permutation(idx[(g, "ctrl")]); h1[g], h2[g] = r[: len(r) // 2], r[len(r) // 2:]
    mean = lambda r: np.asarray(X[r].mean(0)).ravel()
    m1 = {g: mean(h1[g]) for g in kos + ["NT"]}; m2 = {g: mean(h2[g]) for g in kos + ["NT"]}
    Mt = {(g, t): mean(idx[(g, t)]) for g in kos + ["NT"] for t in ("ifn", "cc")}
    rng = np.random.default_rng(0)
    G, AL = [0.0, 0.1, 0.3, 1, 3, 10, 30], [1e-3, 1e-2, 1e-1, 1.0]
    recs = []
    for fold, (tr_i, te_i) in enumerate(KFold(5, shuffle=True, random_state=0).split(kos)):
        tr_all, te_k = [kos[i] for i in tr_i], [kos[i] for i in te_i]
        hidden = np.isin(ko, te_k) & (ctx != "ctrl")
        pca = PCA(50, random_state=0).fit(X[rng.choice(np.flatnonzero(~hidden), 30000, replace=False)].toarray())
        Pc = pca.components_
        Z = lambda r: pca.transform(X[r].toarray())
        for t in ("ifn", "cc"):
            Din = {g: m1[g] - m1["NT"] for g in kos}                        # input: half 1
            Rg = {g: (Mt[(g, t)] - Mt[("NT", t)]) - (m2[g] - m2["NT"]) for g in kos}   # target: half 2
            D = {g: Din[g] @ Pc.T for g in kos}; R = {g: Rg[g] @ Pc.T for g in kos}
            cs, cs_rand = {}, {}
            for p in ["NT"] + tr_all:
                z0, z1 = Z(h1[p]), Z(idx[(p, t)])
                z, v = pairs(z0, z1, 3000 if p == "NT" else 600, rng)
                cs[p] = (z.T @ z, v.T @ z, len(z))
                if p == "NT":
                    z, v = pairs(z0, z1, 3000, rng, "random"); cs_rand[p] = (z.T @ z, v.T @ z, len(z))

            def mc(J, ks):
                out = []
                for k in ks:
                    top = np.argsort(-np.abs(Rg[k]))[:100]
                    out.append(cosine(((J @ D[k]) @ Pc)[top], Rg[k][top]))
                return np.median(out)

            for size in [int(s) for s in args.sizes.split(",")]:
                for rep in range(args.reps if size else 1):
                    tr_k = list(np.random.default_rng(100 * fold + rep).choice(tr_all, size, replace=False)) if size else tr_all
                    inner = [([tr_k[i] for i in x], [tr_k[i] for i in y]) for x, y in KFold(5, shuffle=True, random_state=rep).split(tr_k)]
                    grid = {(g, al): np.mean([mc(fuse(cs, ["NT"] + it, D, R, it, g, al), iv) for it, iv in inner]) for g in G for al in AL}
                    gb, ab = max(grid, key=grid.get)
                    an = max(AL, key=lambda al: grid[(0.0, al)])
                    J = fuse(cs, ["NT"] + tr_k, D, R, tr_k, gb, ab)
                    Jnt = fuse(cs, ["NT"], D, R, [], 0.0, an)
                    Jr = fuse(cs_rand, ["NT"], D, R, [], 0.0, an)
                    ridge = RidgeCV(alphas=np.logspace(-3, 6, 28)).fit(np.stack([D[k] for k in tr_k]), np.stack([R[k] for k in tr_k]))
                    for k in te_k:
                        top = np.argsort(-np.abs(Rg[k]))[:100]
                        preds = {"vanish": -Din[k], "ridge": ridge.predict(D[k][None])[0] @ Pc,
                                 "otop": (J @ D[k]) @ Pc, "otop_nt_only": (Jnt @ D[k]) @ Pc, "nt_random": (Jr @ D[k]) @ Pc}
                        for m, Rh in preds.items():
                            recs.append(dict(fold=fold, context=t, size=len(tr_k), rep=rep, ko=k, method=m,
                                             gamma=gb, cos=cosine(Rh[top], Rg[k][top])))
            print(f"fold {fold} {t} done", flush=True)
    d = pd.DataFrame(recs); d.to_csv(OUT / "metrics.csv", index=False)
    L = ["# Cross-fitted Frangieh knockout × context", ""]
    for t in ("ifn", "cc"):
        dt = d[d.context == t]
        dt = dt.assign(sz=np.where(dt["size"] > 100, "all", dt["size"].astype(str)))
        tab = dt.groupby(["sz", "method", "ko"]).cos.mean().groupby(["sz", "method"]).median().unstack("method")
        L += [f"## {t}", tab.round(3).to_markdown(), ""]
        for sz in tab.index:
            w = dt[dt.sz == sz].groupby(["method", "ko"]).cos.mean().unstack("method")
            for m in ("ridge", "vanish", "otop_nt_only", "nt_random"):
                x = w.otop - w[m]
                L.append(f"- n={sz}: otop − {m}: median {x.median():+.3f}, win {(x > 0).mean():.2f}, p = {wilcoxon(x).pvalue:.2g}")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
