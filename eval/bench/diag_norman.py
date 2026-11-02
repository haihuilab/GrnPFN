"""Diagnostics that decide the next model design (Norman, same 5 folds as P1/P4).

D1 identifiability: does singles-only composition fail where δ_A leaves the control cloud?
D2 where the pair-data gain comes from: stratify by how many training pairs involve A or B.
D3 noise ceiling: split-half reliability of the interaction R = Δ_AB − Δ_A − Δ_B.
D4 is ridge's win global? ablate its features: shared linear map of (Δ_A+Δ_B), product term, both.
Writes eval/results/bench/diag_norman/{pairs.csv,summary.md}.
"""
from __future__ import annotations

import glob
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from common import DATA, RESULTS, cosine

ALPHAS = np.logspace(-2, 4, 13)
OUT = RESULTS / "diag_norman"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C = DATA / "bench_cache"
    full = ad.read_h5ad(C / "norman_hvg3000.h5ad")
    X = full.X.tocsr()
    pert = full.obs["pert"].to_numpy()
    folds = json.loads((C / "norman_folds.json").read_text())
    rows_of = pd.Series(np.arange(len(pert))).groupby(pert).apply(np.asarray).to_dict()
    mean = lambda r: np.asarray(X[r].mean(0)).ravel()
    ctrl = rows_of["ctrl"]
    ce = mean(ctrl)
    dE = {p: mean(r) - ce for p, r in rows_of.items() if p != "ctrl"}

    # control-cloud geometry (gene space): top-20 PCs of control cells
    Xc = X[ctrl].toarray()
    cp = PCA(20, random_state=0).fit(Xc)
    V, lam = cp.components_, cp.explained_variance_
    def cover(d):
        proj = V @ d
        return float(proj @ proj / (d @ d)), float(np.sqrt(np.sum(proj ** 2 / lam)))

    rng = np.random.default_rng(0)
    def half_R(AB, A, B):
        out = []
        for _ in range(5):
            h = {}
            for p, r in (("ab", rows_of[AB]), ("a", rows_of[A]), ("b", rows_of[B]), ("c", ctrl)):
                r = rng.permutation(r)
                h[p] = (r[: len(r) // 2], r[len(r) // 2:])
            Rs = [mean(h["ab"][k]) - mean(h["a"][k]) - mean(h["b"][k]) + mean(h["c"][k]) for k in (0, 1)]
            out.append(Rs)
        return out

    recs = []
    for k in range(5):
        f = folds[str(k)]
        tr = f["train_combos"]
        trA = [c.split("+") for c in tr]
        Xtr_rows = np.flatnonzero(~np.isin(pert, f["test_combos"]))
        pca = PCA(50, random_state=0).fit(X[rng.choice(Xtr_rows, 20000, replace=False)].toarray())
        Pc = pca.components_
        pa = np.stack([dE[a] @ Pc.T for a, _ in trA]); pb = np.stack([dE[b] @ Pc.T for _, b in trA])
        y = np.stack([dE[c] @ Pc.T for c in tr]) - pa - pb
        feats = {"sum": lambda a, b: a + b, "prod": lambda a, b: a * b,
                 "sum+prod": lambda a, b: np.hstack([a + b, a * b])}
        fits = {n: RidgeCV(alphas=ALPHAS).fit(fn(pa, pb), y) for n, fn in feats.items()}
        npart = pd.Series([g for c in trA for g in c]).value_counts()
        for A, B in zip(f["test_A"], f["test_B"]):
            AB = "+".join(sorted((A, B)))
            R = dE[AB] - dE[A] - dE[B]
            top = np.argsort(-np.abs(R))[:100]
            ra, rb = dE[A] @ Pc.T, dE[B] @ Pc.T
            rec = dict(fold=k, AB=AB, A=A, B=B, n_AB=len(rows_of[AB]),
                       partners=int(npart.get(A, 0) + npart.get(B, 0)),
                       partners_min=int(min(npart.get(A, 0), npart.get(B, 0))))
            for n, fn in feats.items():
                Rh = fits[n].predict(fn(ra[None], rb[None]))[0] @ Pc
                rec[f"cos_ridge_{n}"] = cosine(Rh[top], R[top])
            rec["cos_pca_oracle"] = cosine(((R @ Pc.T) @ Pc)[top], R[top])
            hs = [cosine(r1[top], r2[top]) for r1, r2 in half_R(AB, A, B)]
            r = float(np.mean(hs))
            rec["split_half_cos"] = r
            rec["ceiling_cos"] = float(np.sqrt(max(2 * r / (1 + r), 0))) if r > 0 else 0.0
            ca, ma = cover(dE[A]); cb, mb = cover(dE[B])
            rec.update(cover_min=min(ca, cb), mahal_max=max(ma, mb), norm_R=float(np.linalg.norm(R)))
            recs.append(rec)
        print(f"fold {k} done", flush=True)

    d = pd.DataFrame(recs)
    p1 = pd.concat([pd.read_csv(x) for x in glob.glob(str(RESULTS / "p1_transport_composition/metrics_fold*.csv"))])
    p4 = pd.concat([pd.read_csv(x) for x in glob.glob(str(RESULTS / "p4_composition_baselines/metrics_fold?_aug.csv"))])
    for src, m, col in ((p1, "tc_ctrl_r8", "cos_lr_singles"), (p1, "tc_aug_r8", "cos_lr_aug"),
                        (p4, "lowrank_r8", "cos_lr_aug_p4"), (p4, "expr_interact", "cos_ridge_p4"),
                        (p4, "flow_mlp_seq", "cos_flow_aug")):
        d = d.merge(src[src.method == m].set_index("AB").cos_R_top100.rename(col), left_on="AB", right_index=True, how="left")
    d.to_csv(OUT / "pairs.csv", index=False)

    L = ["# Norman diagnostics", "", f"pairs: {len(d)}", ""]
    L += ["## D3 noise ceiling (top-100 |R| genes)",
          f"- split-half cos(R1, R2) median {d.split_half_cos.median():.3f}; implied ceiling for cos(R̂, R) median {d.ceiling_cos.median():.3f}",
          f"- PCA-50 oracle median {d.cos_pca_oracle.median():.3f}",
          f"- low-rank (aug) median {d.cos_lr_aug.median():.3f}; ridge median {d.cos_ridge_p4.median():.3f}",
          f"- fraction of ceiling reached (median of ratio): low-rank {(d.cos_lr_aug / d.ceiling_cos).median():.2f}, ridge {(d.cos_ridge_p4 / d.ceiling_cos).median():.2f}",
          f"- Spearman(per-pair cos low-rank, ceiling) {spearmanr(d.cos_lr_aug, d.ceiling_cos)[0]:.2f}; (ridge, ceiling) {spearmanr(d.cos_ridge_p4, d.ceiling_cos)[0]:.2f}",
          f"- Spearman(per-pair cos low-rank, ridge) {spearmanr(d.cos_lr_aug, d.cos_ridge_p4)[0]:.2f}", ""]
    L += ["## D4 ridge ablation (median cos)"] + [f"- {n}: {d[f'cos_ridge_{n}'].median():.3f}" for n in ("sum", "prod", "sum+prod")] + [""]
    L += ["## D1 identifiability of singles-only composition",
          f"- Spearman(cos singles-only, coverage_min) {spearmanr(d.cos_lr_singles, d.cover_min)[0]:.2f} (p={spearmanr(d.cos_lr_singles, d.cover_min)[1]:.2g})",
          f"- Spearman(cos singles-only, Mahalanobis_max) {spearmanr(d.cos_lr_singles, d.mahal_max)[0]:.2f} (p={spearmanr(d.cos_lr_singles, d.mahal_max)[1]:.2g})",
          f"- median coverage of δ in control top-20 PCs: {d.cover_min.median():.3f}; median Mahalanobis {d.mahal_max.median():.1f} SD", ""]
    L += ["## D2 gain by partner pairs in training (A and B combined)"]
    d["pbin"] = pd.cut(d.partners, [-1, 2, 5, 10, 100])
    g = d.groupby("pbin", observed=True).agg(n=("AB", "size"), singles=("cos_lr_singles", "median"), aug=("cos_lr_aug", "median"),
                                             ridge=("cos_ridge_p4", "median"), flow=("cos_flow_aug", "median"), ceiling=("ceiling_cos", "median"))
    L += [g.round(3).to_markdown(), ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
