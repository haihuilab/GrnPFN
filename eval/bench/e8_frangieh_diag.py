"""E8′ diagnostic — knockout × context on Frangieh 2021 (melanoma; Control / IFN-γ / TIL co-culture).

Question that decides the model: does the cell-level state distribution of a knockout population predict how
its knockout effect changes in a new context, beyond what its mean predicts?  An affine-in-state map (our
current operator, Gaussian OT, ridge) only sees the mean, so state dependence needs a nonlinear field.

Interaction per knockout and target context t:  R = Δ_KO|t − Δ_KO|ctrl,  Δ_KO|c = mean(KO|c) − mean(NT|c).
5-fold over knockouts; a test knockout's stimulated cells are hidden (its Control cells are available).
  ridge_shared  shared linear map Δ_KO|ctrl → R from training knockouts (mean level; Norman's strong baseline)
  krr           RBF kernel ridge, same inputs
  gauss_nt      closed-form Gaussian OT NT|ctrl → NT|t applied to the KO mean (affine, NT heterogeneity only)
  field_nt      OT-paired displacement field from NT cells only, kNN-regressed on state, applied to KO cells
  field_pool    same field learned from NT and every training knockout population
  ens           mean of ridge_shared and field_pool (does the field add information to the mean-level model?)
Writes eval/results/bench/e8_frangieh/{pairs.csv,summary.md}; cache data/bench_cache/frangieh_hvg3000.h5ad.
"""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor

from common import DATA, RESULTS, cosine, pearson

OUT = RESULTS / "e8_frangieh"
CACHE = DATA / "bench_cache" / "frangieh_hvg3000.h5ad"
CTX = {"Control": "ctrl", "IFNγ": "ifn", "Co-culture": "cc"}


def build_cache(min_cells: int = 40) -> ad.AnnData:
    a = ad.read_h5ad(DATA / "frangieh2021" / "frangieh_2021_rna.h5ad")
    o = a.obs
    keep = (o.MOI == 1).to_numpy()
    a = a[keep].copy()
    a.obs["ko"] = np.where(a.obs.perturbation.astype(str) == "control", "NT", a.obs.perturbation.astype(str))
    a.obs["context"] = a.obs.perturbation_2.astype(str).map(CTX)
    n = pd.crosstab(a.obs.ko, a.obs.context)
    ok = n.index[(n.min(1) >= min_cells)]
    a = a[a.obs.ko.isin(ok).to_numpy()].copy()
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    lib = np.asarray(a.X.sum(1)).ravel()
    sub = a[np.random.default_rng(0).choice(a.n_obs, min(40000, a.n_obs), replace=False)].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=3000, flavor="seurat_v3")
    hv = sub.var_names[sub.var.highly_variable.to_numpy()]
    X = a[:, hv].X.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr()
    X.data = np.log1p(X.data)
    out = ad.AnnData(X.astype(np.float32), obs=a.obs[["ko", "context"]].copy(), var=pd.DataFrame(index=hv))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(CACHE)
    return out


def ot_pairs(z0, z1, n, rng):
    i0 = rng.choice(len(z0), min(n, len(z0)), replace=False)
    i1 = rng.choice(len(z1), min(n, len(z1)), replace=False)
    m = min(len(i0), len(i1))
    i0, i1 = i0[:m], i1[:m]
    c = ((z0[i0, None, :] - z1[None, i1, :]) ** 2).sum(-1)
    r, col = linear_sum_assignment(c)
    return z0[i0[r]], z1[i1[col]] - z0[i0[r]]


def gauss_map(z0, z1, shrink=1e-2):
    m0, m1 = z0.mean(0), z1.mean(0)
    def cov(z):
        c = np.cov(z.T)
        return c + shrink * np.trace(c) / len(c) * np.eye(len(c))
    S0, S1 = cov(z0), cov(z1)
    w, V = np.linalg.eigh(S0)
    s0h = V @ np.diag(np.sqrt(w)) @ V.T
    s0ih = V @ np.diag(1 / np.sqrt(w)) @ V.T
    w2, V2 = np.linalg.eigh(s0h @ S1 @ s0h)
    A = s0ih @ (V2 @ np.diag(np.sqrt(np.maximum(w2, 0))) @ V2.T) @ s0ih
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-ot", type=int, default=600)
    ap.add_argument("--nn-dims", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(CACHE) if CACHE.exists() else build_cache()
    X = a.X.tocsr()
    ko, ctx = a.obs.ko.to_numpy(), a.obs.context.to_numpy()
    kos = sorted(set(ko) - {"NT"})
    print(f"cells {a.n_obs:,}; knockouts {len(kos)}", flush=True)
    idx = pd.Series(np.arange(len(ko))).groupby([ko, ctx]).apply(np.asarray).to_dict()
    mean = lambda r: np.asarray(X[r].mean(0)).ravel()
    M = {key: mean(r) for key, r in idx.items()}
    rng = np.random.default_rng(0)

    def split_half_R(g, t):
        cs = []
        for _ in range(5):
            h = {}
            for key in ((g, t), (g, "ctrl"), ("NT", t), ("NT", "ctrl")):
                r = rng.permutation(idx[key]); h[key] = (r[: len(r) // 2], r[len(r) // 2:])
            Rs = [mean(h[(g, t)][s]) - mean(h[("NT", t)][s]) - mean(h[(g, "ctrl")][s]) + mean(h[("NT", "ctrl")][s]) for s in (0, 1)]
            cs.append(Rs)
        return cs

    recs = []
    for fold, (tr_i, te_i) in enumerate(KFold(5, shuffle=True, random_state=0).split(kos)):
        tr_k, te_k = [kos[i] for i in tr_i], [kos[i] for i in te_i]
        hidden = np.isin(ko, te_k) & (ctx != "ctrl")
        fit_rows = rng.choice(np.flatnonzero(~hidden), 30000, replace=False)
        pca = PCA(50, random_state=0).fit(X[fit_rows].toarray())
        Pc = pca.components_
        Z = lambda r: pca.transform(X[r].toarray()).astype(np.float32)
        Zc = {g: Z(idx[(g, "ctrl")]) for g in kos + ["NT"]}
        for t in ("ifn", "cc"):
            D0 = {g: (M[(g, "ctrl")] - M[("NT", "ctrl")]) for g in kos}
            Rg = {g: (M[(g, t)] - M[("NT", t)]) - D0[g] for g in kos}
            xtr = np.stack([D0[g] @ Pc.T for g in tr_k]); ytr = np.stack([Rg[g] @ Pc.T for g in tr_k])
            ridge = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(xtr, ytr)
            s = xtr.std()
            krr = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": np.logspace(-3, 1, 5),
                               "gamma": np.logspace(-3, 0, 4) / (s ** 2 * xtr.shape[1])}, cv=5).fit(xtr, ytr).best_estimator_
            Zt_nt = Z(idx[("NT", t)])
            A = gauss_map(Zc["NT"], Zt_nt)
            # displacement fields
            zs, vs = ot_pairs(Zc["NT"], Zt_nt, 3000, rng)
            f_nt = KNeighborsRegressor(args.k).fit(zs[:, : args.nn_dims], vs)
            P0, V0 = [zs], [vs]
            for g in tr_k:
                z0p, v0p = ot_pairs(Zc[g], Z(idx[(g, t)]), args.n_ot, rng)
                P0.append(z0p); V0.append(v0p)
            P0, V0 = np.concatenate(P0), np.concatenate(V0)
            f_pool = KNeighborsRegressor(args.k).fit(P0[:, : args.nn_dims], V0)
            from sklearn.linear_model import Ridge
            l_nt = Ridge(1.0).fit(zs, vs)
            l_pool = Ridge(1.0).fit(P0, V0)
            base_nt = f_nt.predict(Zc["NT"][:, : args.nn_dims]).mean(0)
            base_pool = f_pool.predict(Zc["NT"][:, : args.nn_dims]).mean(0)
            for g in te_k:
                R = Rg[g]
                top = np.argsort(-np.abs(R))[:100]
                d0p = D0[g] @ Pc.T
                preds = {
                    "ridge_shared": ridge.predict(d0p[None])[0] @ Pc,
                    "krr": krr.predict(d0p[None])[0] @ Pc,
                    "gauss_nt": ((A - np.eye(50)) @ d0p) @ Pc,
                    "field_nt": (f_nt.predict(Zc[g][:, : args.nn_dims]).mean(0) - base_nt) @ Pc,
                    "field_pool": (f_pool.predict(Zc[g][:, : args.nn_dims]).mean(0) - base_pool) @ Pc,
                }
                preds["field_nt_lin"] = (l_nt.predict(Zc[g]).mean(0) - l_nt.predict(Zc["NT"]).mean(0)) @ Pc
                preds["field_pool_lin"] = (l_pool.predict(Zc[g]).mean(0) - l_pool.predict(Zc["NT"]).mean(0)) @ Pc
                preds["ens"] = 0.5 * (preds["ridge_shared"] + preds["field_pool"])
                preds["ens_lin"] = 0.5 * (preds["ridge_shared"] + preds["field_pool_lin"])
                hs = [cosine(r1[top], r2[top]) for r1, r2 in split_half_R(g, t)]
                rh = float(np.mean(hs))
                true_d = M[(g, t)] - M[("NT", t)]
                t20 = np.argsort(-np.abs(true_d))[:20]
                base = dict(fold=fold, ko=g, context=t, n_ctrl=len(idx[(g, "ctrl")]), n_t=len(idx[(g, t)]),
                            norm_R=float(np.linalg.norm(R)), norm_D0=float(np.linalg.norm(D0[g])),
                            split_half=rh, ceiling=float(np.sqrt(max(2 * rh / (1 + rh), 0))) if rh > 0 else 0.0)
                for m, Rh in preds.items():
                    recs.append(dict(base, method=m, cos_R_top100=cosine(Rh[top], R[top]),
                                     pearson_de20=pearson((D0[g] + Rh)[t20], true_d[t20])))
            print(f"fold {fold} {t} done", flush=True)
    d = pd.DataFrame(recs)
    d.to_csv(OUT / "pairs.csv", index=False)
    L = ["# E8′ diagnostic: Frangieh knockout × context", "",
         f"knockouts {len(kos)}; k={args.k}, nn_dims={args.nn_dims}, n_ot={args.n_ot}", ""]
    for t in ("ifn", "cc"):
        dt = d[d.context == t]
        rel = dt[dt.method == "ridge_shared"].set_index("ko")
        good = rel.index[rel.split_half > 0.5]
        L += [f"## context {t}", f"- split-half median {rel.split_half.median():.3f}; ceiling median {rel.ceiling.median():.3f};"
              f" reliable knockouts (split-half > 0.5): {len(good)}", ""]
        w = dt.pivot(index="ko", columns="method", values="cos_R_top100")
        tab = pd.DataFrame({"all": w.median(), "reliable": w.loc[good].median(),
                            "pearson_de20": dt.groupby("method").pearson_de20.mean()})
        L += [tab.round(3).to_markdown(), ""]
        for m in ("field_pool", "field_pool_lin", "field_nt", "field_nt_lin", "ens", "ens_lin", "krr", "gauss_nt"):
            for name, sub in (("all", w), ("reliable", w.loc[good])):
                diff = sub[m] - sub["ridge_shared"]
                L.append(f"- {m} − ridge_shared ({name}): median {diff.median():+.3f}, p = {wilcoxon(diff).pvalue:.2g}")
        for m1, m2 in (("field_nt", "field_nt_lin"), ("field_pool", "field_pool_lin"), ("ens", "ens_lin")):
            diff = w[m1] - w[m2]
            L.append(f"- {m1} − {m2} (nonlinearity, all): median {diff.median():+.3f}, p = {wilcoxon(diff).pvalue:.2g}")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
