"""Real-screen runs in GRN-PFN form, built from the benchmark: benchmark cell roles and outer folds,
256 genes by max |Δ| over input-third singles. One dict per outer fold."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from common import DATA
from benchmark import outer_folds

G_KEEP = 256


def real_folds(dataset, setting, seed, rotation, pair_frac=None):
    """pair_frac: if set, TRAINING-side interactions (fit/val/train/f3/v3/s3 and noise_tr) use a random fraction of each
    pair's cells (breadth-vs-depth analysis); test targets always use all of the pair's cells."""
    zs = setting == "zero-shot"
    full = ad.read_h5ad(DATA / "bench_cache" / f"{dataset}_hvg3000.h5ad"); X = full.X.tocsr(); pert = full.obs["pert"].to_numpy()
    rows_of = pd.Series(np.arange(len(pert))).groupby(pert).apply(np.asarray).to_dict()
    folds = outer_folds(dataset, rows_of, zs); singles = sorted({g for tr, te in folds for t in tr + te for g in t}); idx = {p: i for i, p in enumerate(singles)}
    mean = lambda r: np.asarray(X[r].mean(0)).ravel(); key = lambda x, y: "+".join(sorted((x, y)))
    rot = lambda parts: tuple(parts[(i + rotation) % 3] for i in range(3)); sp = np.random.default_rng(seed)
    thirds = {p: rot(tuple(np.array_split(sp.permutation(rows_of[p]), 3))) for p in singles + ["ctrl"]}
    half = lambda r: (r[: len(r) // 2], r[len(r) // 2:])
    role = {"input": {p: t[0] for p, t in thirds.items()}, "train": {p: t[1] for p, t in thirds.items()}, "test": {p: t[2] for p, t in thirds.items()},
            "fit": {p: half(t[1])[0] for p, t in thirds.items()}, "val": {p: half(t[1])[1] for p, t in thirds.items()}}
    for j, r in enumerate(("f3", "v3", "s3")):                                          # train third in three parts: independent
        role[r] = {p: np.array_split(t[1], 3)[j] for p, t in thirds.items()}            # context / target / early-stopping noise
    M = {r: {p: mean(c) for p, c in role[r].items()} for r in role}
    Dall = np.stack([M["input"][p] - M["input"]["ctrl"] for p in singles])
    keep = np.sort(np.argsort(-np.abs(Dall).max(0))[:G_KEEP])
    Dm = Dall[:, keep]; mu = M["input"]["ctrl"][keep]
    Xc = X[role["input"]["ctrl"]][:, keep]; sd_ctrl = np.sqrt(np.maximum(np.asarray(Xc.multiply(Xc).mean(0)).ravel() - np.asarray(Xc.mean(0)).ravel() ** 2, 0))
    MAB = {key(*t): mean(rows_of[key(*t)])[keep] for tr, te in folds for t in tr + te}
    sub = {}
    if pair_frac is not None:                                                           # per-pair cell subsample (seeded by pair name)
        for tr_, te_ in folds:
            for t in tr_ + te_:
                c = rows_of[key(*t)]; rs = np.random.default_rng([seed, sum(map(ord, key(*t)))])
                sub[key(*t)] = rs.permutation(c)[: max(3, int(round(pair_frac * len(c))))]
    MABs = {k_: mean(c)[keep] for k_, c in sub.items()}
    Rg = lambda t, r: (MABs[key(*t)] if (pair_frac is not None and r != "test") else MAB[key(*t)]) - M[r][t[0]][keep] - M[r][t[1]][keep] + M[r]["ctrl"][keep]
    ncell = lambda p: float(len(rows_of[p]))
    def vmean(r):                                                                       # per-gene variance / n of a cell set
        Z = X[r][:, keep]; mu_ = np.asarray(Z.mean(0)).ravel(); return np.maximum(np.asarray(Z.multiply(Z).mean(0)).ravel() - mu_ ** 2, 0) / max(len(r), 1)
    vn = {p: vmean(role["train"][p]) for p in singles + ["ctrl"]}
    noise_tr_g = lambda t: vmean(sub[key(*t)] if pair_frac is not None else rows_of[key(*t)]) + vn[t[0]] + vn[t[1]] + vn["ctrl"]   # per-gene, training target
    noise_tr = lambda t: float(np.mean(noise_tr_g(t)))                                              # sampling variance of R_train
    vt = {p: vmean(role["test"][p]) for p in singles + ["ctrl"]}
    noise_te_g = lambda t: vmean(rows_of[key(*t)]) + vt[t[0]] + vt[t[1]] + vt["ctrl"]               # per-gene, test target
    hubs = np.array([idx[p] for p in singles if p.startswith("tram@")], int)
    allp = sorted({key(*t): t for tr, te in folds for t in tr + te}.items()); allt = [t for _, t in allp]
    full_ep = dict(D=Dm, mu=mu, pairs=np.array([(idx[x], idx[y]) for x, y in allt]), R_tr=np.stack([Rg(t, "train") for t in allt]),
                   R_te=np.stack([Rg(t, "test") for t in allt]))
    out = []
    for k, (tr, te) in enumerate(folds):
        out.append(dict(k=k, tr=tr, te=te, singles=singles, D=Dm, mu=mu, hubs=hubs, keep=keep, genes=np.asarray(full.var_names)[keep],
                        pc=np.array([(idx[x], idx[y]) for x, y in tr]), pq=np.array([(idx[x], idx[y]) for x, y in te]),
                        R={r: np.stack([Rg(t, r) for t in tr]) for r in ("fit", "val", "train", "test", "f3", "v3", "s3")},
                        R_q_test=np.stack([Rg(t, "test") for t in te]),
                        n_single=np.array([ncell(p) for p in singles]), n_pair_c=np.array([ncell(key(*t)) for t in tr]),
                        n_pair_q=np.array([ncell(key(*t)) for t in te]), n_ctrl=ncell("ctrl"), key=key, full=full_ep, sd_ctrl=sd_ctrl,
                        noise_tr=np.array([noise_tr(t) for t in tr]), noise_te_g=np.stack([noise_te_g(t) for t in te]),
                        noise_tr_g=np.stack([noise_tr_g(t) for t in tr])))
    return out
