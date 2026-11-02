"""Prior predictive check for the GRN simulator: statistics computed identically on real screens (cached h5ad, input /
train / test thirds of singles and control, pair cells split in halves, 256 genes by max |Δ| on the input third) and on
simulated episodes.
  rms_D, rms_R      : root mean squared norm of single effects and of interactions (test-third version)
  rep_D             : median cos(Δ from third 1, Δ from third 2), top-100 genes of the first
  rep_R             : median cos(R half 1, R half 2), fully independent cells, top-100 genes of the first
  buf               : median cos(R half 1, −(Δ_A + Δ_B)) (alignment with buffering), top-100 of R
  frac_rep          : fraction of pairs with rep_R > 0.2
  snr_R             : median ⟨R¹, R²⟩ / (‖R¹ − R²‖²/4) (interaction signal-to-noise)
  mu_q50            : median control log-expression of kept genes"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def _tc(a, b):
    t = np.argsort(-np.abs(a))[:100]; return float(a[t] @ b[t] / (np.linalg.norm(a[t]) * np.linalg.norm(b[t]) + 1e-12))


def stats(D, D1, R1, R2, Rte, pairs, mu):
    s = D[pairs[:, 0]] + D[pairs[:, 1]]
    rep_R = np.array([_tc(a, b) for a, b in zip(R1, R2)])
    sig = (R1 * R2).sum(1); nz = ((R1 - R2) ** 2).sum(1) / 4
    return dict(rms_D=np.sqrt((D ** 2).sum(1).mean()), rms_R=np.sqrt((Rte ** 2).sum(1).mean()), rep_D=np.median([_tc(a, b) for a, b in zip(D, D1)]),
                rep_R=np.median(rep_R), buf=np.median([_tc(r, -x) for r, x in zip(R1, s)]), frac_rep=float((rep_R > 0.2).mean()),
                snr_R=float(np.median(sig / nz)), rho=float(np.median(sig / ((s ** 2).sum(1) + 1e-12))), mu_q50=float(np.median(mu)))


def real(ds, G=256, seed=0):
    import anndata as ad
    a = ad.read_h5ad(ROOT / "data" / "bench_cache" / f"{ds}_hvg3000.h5ad"); X = a.X.tocsr(); pert = a.obs["pert"].astype(str).to_numpy()
    rows = pd.Series(np.arange(len(pert))).groupby(pert).apply(np.asarray).to_dict()
    ok = lambda p: len(rows.get(p, [])) >= 20
    pl = sorted(p for p in rows if "+" in p and ok(p) and all(ok(g) for g in p.split("+")))
    singles = sorted({g for p in pl for g in p.split("+")}); idx = {p: i for i, p in enumerate(singles)}
    rng = np.random.default_rng(seed); mean = lambda r: np.asarray(X[r].mean(0)).ravel()
    th = {p: np.array_split(rng.permutation(rows[p]), 3) for p in singles + ["ctrl"]}
    M = {p: [mean(t) for t in th[p]] for p in th}
    D = np.stack([M[p][0] - M["ctrl"][0] for p in singles]); D1 = np.stack([M[p][1] - M["ctrl"][1] for p in singles])
    keep = np.sort(np.argsort(-np.abs(D).max(0))[:G])
    R1, R2, Rte = [], [], []
    for p in pl:
        A, B = p.split("+"); r = rng.permutation(rows[p]); h = (mean(r[: len(r) // 2]), mean(r[len(r) // 2:])); full = mean(rows[p])
        R1.append(h[0] - M[A][1] - M[B][1] + M["ctrl"][1]); R2.append(h[1] - M[A][2] - M[B][2] + M["ctrl"][2]); Rte.append(full - M[A][2] - M[B][2] + M["ctrl"][2])
    pairs = np.array([(idx[p.split("+")[0]], idx[p.split("+")[1]]) for p in pl])
    k = lambda Z: np.stack(Z)[:, keep]
    return stats(D[:, keep], D1[:, keep], k(R1), k(R2), k(Rte), pairs, M["ctrl"][0][keep])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    from grnpfn.sim import sample_grn_episode
    out = str(ROOT / "eval" / "results" / "bench" / "grnpfn")
    import os; os.makedirs(out, exist_ok=True)
    rf = f"{out}/ppc_real.csv"
    if os.path.exists(rf):
        R = pd.read_csv(rf, index_col=0)
    else:
        R = pd.DataFrame({ds: real(ds) for ds in ["norman", "wessels", "sp4_A172", "sp4_T98G", "sp4_U87MG"]}).T; R.to_csv(rf)
    rows = []
    for e in range(n):
        ep = sample_grn_episode(np.random.default_rng(60_000 + e))
        rows.append(dict(kind=ep["kind"], **stats(ep["D"], ep["D1"], ep["R_h1"], ep["R_h2"], ep["R_te"], ep["pairs"], ep["mu_ctrl"])))
    S = pd.DataFrame(rows); S.to_csv(f"{out}/ppc_sim{sys.argv[2] if len(sys.argv) > 2 else ''}.csv", index=False)
    pd.set_option("display.width", 200)
    print("real:"); print(R.round(3).to_string())
    print("sim 5/50/95%:"); print(S.drop(columns="kind").quantile([.05, .5, .95]).round(3).to_string())
    pct = pd.DataFrame({c: [100 * (S[c] <= R.loc[ds, c]).mean() for ds in R.index] for c in R.columns}, index=R.index)
    print("percentile of real among sim:"); print(pct.round(1).to_string())
    bad = int(((pct < 2.5) | (pct > 97.5)).sum().sum()); print(f"outside central 95%: {bad}/{pct.size}")


if __name__ == "__main__":
    main()
