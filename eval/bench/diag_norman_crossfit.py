"""Cross-fitted check of the Norman interaction benchmark (shared-noise artifact).
Inputs Δ_A, Δ_B (and training-pair inputs) use half 1 of each single's cells; the target
R = Δ_AB − Δ_A − Δ_B uses half 2 of the singles. Compares, same folds, contaminated vs cross-fitted:
scaled-add (generic buffering), intercept only (mean training R), ridge on Δ_A+Δ_B, ridge on [sum, product]."""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from common import DATA, RESULTS, cosine


def main() -> None:
    C = DATA / "bench_cache"
    full = ad.read_h5ad(C / "norman_hvg3000.h5ad"); X = full.X.tocsr()
    pert = full.obs["pert"].to_numpy()
    folds = json.loads((C / "norman_folds.json").read_text())
    rows_of = pd.Series(np.arange(len(pert))).groupby(pert).apply(np.asarray).to_dict()
    mean = lambda r: np.asarray(X[r].mean(0)).ravel()
    sp = np.random.default_rng(123)
    halves = {}
    for p, r in rows_of.items():
        r = sp.permutation(r); halves[p] = (r[: len(r) // 2], r[len(r) // 2:])
    ce1, ce2, ce = mean(halves["ctrl"][0]), mean(halves["ctrl"][1]), mean(rows_of["ctrl"])
    singles = sorted(p for p in rows_of if "+" not in p and p != "ctrl")
    D1 = {p: mean(halves[p][0]) - ce1 for p in singles}
    D2 = {p: mean(halves[p][1]) - ce2 for p in singles}
    Dall = {p: mean(rows_of[p]) - ce for p in rows_of if p != "ctrl"}
    rng = np.random.default_rng(0)
    recs = []
    for k in range(5):
        f = folds[str(k)]
        tr = [c.split("+") for c in f["train_combos"]]
        Pc = PCA(50, random_state=0).fit(X[rng.choice(np.flatnonzero(~np.isin(pert, f["test_combos"])), 20000, replace=False)].toarray()).components_
        for mode in ("contaminated", "crossfit"):
            Din = Dall if mode == "contaminated" else D1
            Dtg = Dall if mode == "contaminated" else D2
            Rt = lambda a, b: Dall["+".join(sorted((a, b)))] - Dtg[a] - Dtg[b]
            S = np.stack([(Din[a] + Din[b]) @ Pc.T for a, b in tr]); P = np.stack([(Din[a] @ Pc.T) * (Din[b] @ Pc.T) for a, b in tr])
            Y = np.stack([Rt(a, b) @ Pc.T for a, b in tr])
            lam = float(np.sum(np.stack([Dall["+".join(sorted((a, b)))] for a, b in tr]) * np.stack([Din[a] + Din[b] for a, b in tr])) /
                        np.sum(np.stack([Din[a] + Din[b] for a, b in tr]) ** 2))
            r_sum = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(S, Y)
            r_sp = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(np.hstack([S, P]), Y)
            r_int = Y.mean(0)
            for a, b in zip(f["test_A"], f["test_B"]):
                R = Rt(a, b); top = np.argsort(-np.abs(R))[:100]
                s = ((Din[a] + Din[b]) @ Pc.T)[None]; pp = ((Din[a] @ Pc.T) * (Din[b] @ Pc.T))[None]
                preds = {"scaled_add": (lam - 1) * (Din[a] + Din[b]), "intercept": r_int @ Pc,
                         "ridge_sum": r_sum.predict(s)[0] @ Pc, "ridge_sum_prod": r_sp.predict(np.hstack([s, pp]))[0] @ Pc}
                for m, Rh in preds.items():
                    recs.append(dict(fold=k, mode=mode, AB="+".join(sorted((a, b))), method=m, cos=cosine(Rh[top], R[top])))
        print(f"fold {k} done", flush=True)
    d = pd.DataFrame(recs)
    out = RESULTS / "diag_norman"; d.to_csv(out / "crossfit.csv", index=False)
    tab = d.groupby(["mode", "method"]).cos.median().unstack("mode")
    L = ["# Norman cross-fit check (median cos(R̂, R) top-100)", tab.round(3).to_markdown(), ""]
    w = d[d["mode"] == "crossfit"].pivot(index="AB", columns="method", values="cos")
    for m in ("intercept", "scaled_add"):
        x = w.ridge_sum - w[m]; L.append(f"- crossfit ridge_sum − {m}: median {x.median():+.3f}, p = {wilcoxon(x).pvalue:.2g}")
    (out / "crossfit.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
