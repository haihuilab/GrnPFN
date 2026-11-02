"""Iteration 2 — joint OT-identified response operator (Frangieh knockout × context).

Finding from e8_frangieh_diag: a LINEAR displacement field fitted to OT-paired control cells (no knockout
supervision) predicts knockout × context interactions almost as well as ridge trained on ~170 knockouts, and
averaging the two beats ridge on 82% of knockouts. Here the two information sources are fused in one estimator:

    J_t = argmin  (1/N) Σ_cells ‖v_i − J z̃_i‖²          z̃, v: OT-paired displacement, centred within population
                + γ (1/P) Σ_train KOs ‖R_p − J Δ_p‖²     population-mean interactions (the ridge term)
                + α ‖J‖²                                    closed form; γ, α by inner CV over training KOs
    R̂_KO = J_t Δ_KO|ctrl

Cells give within-population state heterogeneity (thousands of samples); knockouts give between-population
contrasts. Controls for "it is just variance reduction": ridge+KRR ensemble, PLS, ridge with wider CV.
Learning curve: number of training knockouts ∈ {10, 25, 50, all}.
"""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold

from common import DATA, RESULTS, cosine

OUT = RESULTS / "e8_iter2_otop"
CACHE = DATA / "bench_cache" / "frangieh_hvg3000.h5ad"


def ot_pairs(z0, z1, n, rng):
    i0 = rng.choice(len(z0), min(n, len(z0)), replace=False)
    i1 = rng.choice(len(z1), min(n, len(z1)), replace=False)
    m = min(len(i0), len(i1)); i0, i1 = i0[:m], i1[:m]
    r, c = linear_sum_assignment(((z0[i0, None] - z1[None, i1]) ** 2).sum(-1))
    z, v = z0[i0[r]], z1[i1[c]] - z0[i0[r]]
    return z - z.mean(0), v - v.mean(0)


class OTOperator:
    """Closed-form fused estimator; sufficient statistics per population so CV is cheap."""

    def __init__(self, cell_stats, D, R):
        self.cs, self.D, self.R = cell_stats, D, R          # cs[p] = (ZtZ, VtZ, n)

    def fit(self, pops, kos, gamma, alpha):
        d = next(iter(self.cs.values()))[0].shape[0]
        N = sum(self.cs[p][2] for p in pops)
        A = sum(self.cs[p][0] for p in pops) / max(N, 1)
        B = sum(self.cs[p][1] for p in pops) / max(N, 1)
        if len(kos) and gamma > 0:
            Dm, Rm = np.stack([self.D[k] for k in kos]), np.stack([self.R[k] for k in kos])
            A = A + gamma * Dm.T @ Dm / len(kos)
            B = B + gamma * Rm.T @ Dm / len(kos)
        s = np.trace(A) / d
        return B @ np.linalg.inv(A + alpha * s * np.eye(d))


def med_cos(J, kos, D, R, Pc, Rg):
    out = []
    for k in kos:
        Rt = Rg[k]; top = np.argsort(-np.abs(Rt))[:100]
        out.append(cosine(((J @ D[k]) @ Pc)[top], Rt[top]))
    return np.median(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ot", type=int, default=600)
    ap.add_argument("--sizes", default="10,25,50,0")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(CACHE)
    X = a.X.tocsr()
    ko, ctx = a.obs.ko.to_numpy(), a.obs.context.to_numpy()
    kos = sorted(set(ko) - {"NT"})
    idx = pd.Series(np.arange(len(ko))).groupby([ko, ctx]).apply(np.asarray).to_dict()
    mean = lambda r: np.asarray(X[r].mean(0)).ravel()
    M = {key: mean(r) for key, r in idx.items()}
    rng = np.random.default_rng(0)
    GAMMAS, ALPHAS = [0.0, 0.03, 0.1, 0.3, 1, 3, 10], [1e-3, 1e-2, 1e-1, 1.0]
    recs, chosen = [], []
    for fold, (tr_i, te_i) in enumerate(KFold(5, shuffle=True, random_state=0).split(kos)):
        tr_all, te_k = [kos[i] for i in tr_i], [kos[i] for i in te_i]
        hidden = np.isin(ko, te_k) & (ctx != "ctrl")
        pca = PCA(50, random_state=0).fit(X[rng.choice(np.flatnonzero(~hidden), 30000, replace=False)].toarray())
        Pc = pca.components_
        Z = lambda r: pca.transform(X[r].toarray()).astype(np.float32)
        Zc = {g: Z(idx[(g, "ctrl")]) for g in kos + ["NT"]}
        for t in ("ifn", "cc"):
            Dg = {g: M[(g, "ctrl")] - M[("NT", "ctrl")] for g in kos}
            Rg = {g: (M[(g, t)] - M[("NT", t)]) - Dg[g] for g in kos}
            D = {g: Dg[g] @ Pc.T for g in kos}; R = {g: Rg[g] @ Pc.T for g in kos}
            cs = {}
            for p in ["NT"] + tr_all:
                z, v = ot_pairs(Zc[p], Z(idx[(p, t)]), 3000 if p == "NT" else args.n_ot, rng)
                cs[p] = (z.T @ z, v.T @ z, len(z))
            op = OTOperator(cs, D, R)
            sizes = [int(s) for s in args.sizes.split(",")]
            for size in sizes:
                for rep in range(args.reps if size else 1):
                    tr_k = list(np.random.default_rng(100 * fold + rep).choice(tr_all, size, replace=False)) if size else tr_all
                    # inner CV over the training knockouts for (γ, α)
                    inner = KFold(min(5, len(tr_k)), shuffle=True, random_state=rep).split(tr_k)
                    inner = [([tr_k[i] for i in a_], [tr_k[i] for i in b_]) for a_, b_ in inner]
                    best = {}
                    for g in GAMMAS:
                        for al in ALPHAS:
                            sc = np.mean([med_cos(op.fit(["NT"] + itr, itr, g, al), iva, D, R, Pc, Rg) for itr, iva in inner])
                            best[(g, al)] = sc
                    g_b, a_b = max(best, key=best.get)
                    a_cell = max(ALPHAS, key=lambda al: best[(0.0, al)])
                    J = op.fit(["NT"] + tr_k, tr_k, g_b, a_b)
                    J_nt = op.fit(["NT"], [], 0.0, a_cell)
                    chosen.append(dict(fold=fold, context=t, size=len(tr_k), rep=rep, gamma=g_b, alpha=a_b))
                    xtr = np.stack([D[k] for k in tr_k]); ytr = np.stack([R[k] for k in tr_k])
                    ridge = RidgeCV(alphas=np.logspace(-3, 6, 28)).fit(xtr, ytr)
                    s = xtr.std()
                    krr = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": np.logspace(-3, 1, 5),
                                       "gamma": np.logspace(-3, 0, 4) / (s ** 2 * 50)}, cv=min(5, len(tr_k))).fit(xtr, ytr).best_estimator_
                    pls = PLSRegression(n_components=min(8, len(tr_k) - 1)).fit(xtr, ytr)
                    for k in te_k:
                        Rt = Rg[k]; top = np.argsort(-np.abs(Rt))[:100]
                        x = D[k][None]
                        preds = {"otop": (J @ D[k]) @ Pc, "otop_nt_only": (J_nt @ D[k]) @ Pc,
                                 "ridge": ridge.predict(x)[0] @ Pc, "krr": krr.predict(x)[0] @ Pc,
                                 "pls": pls.predict(x)[0] @ Pc}
                        preds["ens_ridge_krr"] = 0.5 * (preds["ridge"] + preds["krr"])
                        for m, Rh in preds.items():
                            recs.append(dict(fold=fold, context=t, size=len(tr_k), rep=rep, ko=k, method=m,
                                             cos=cosine(Rh[top], Rt[top])))
            print(f"fold {fold} {t} done", flush=True)
    d = pd.DataFrame(recs); d.to_csv(OUT / "metrics.csv", index=False)
    ch = pd.DataFrame(chosen); ch.to_csv(OUT / "chosen.csv", index=False)
    L = ["# Iteration 2: fused OT-identified operator", ""]
    for t in ("ifn", "cc"):
        dt = d[d.context == t]
        # median over knockouts of the per-knockout mean over reps
        tab = dt.groupby(["size", "method", "ko"]).cos.mean().groupby(["size", "method"]).median().unstack("method")
        L += [f"## {t}: median cos(R̂, R) by number of training knockouts", tab.round(3).to_markdown(), ""]
        for size in sorted(dt["size"].unique()):
            w = dt[dt["size"] == size].groupby(["method", "ko"]).cos.mean().unstack("method")
            for m in ("ridge", "ens_ridge_krr", "pls", "otop_nt_only"):
                x = w.otop - w[m]
                L.append(f"- n={size}: otop − {m}: median {x.median():+.3f}, win {(x > 0).mean():.2f}, p = {wilcoxon(x).pvalue:.2g}")
        L += ["", "chosen γ (fraction): " + ch[ch.context == t].gamma.value_counts(normalize=True).round(2).to_dict().__repr__(), ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
