"""Cycle-2 diagnosis, part 2 (CPU): D5 nearest-neighbour similarity of a pair's additive profile s = Δ_A + Δ_B (and of
its interaction R^train) to the training pairs — test pairs vs training pairs left out one at a time; D6 split-half
reproducibility of R (pair-specific cosine between R^fit and R^val of the same training pair), and of R^test vs R^train
of the screen's mean; D7 interaction norm of test vs training pairs.   python x71b_diag.py"""
import numpy as np, pandas as pd
from common import RESULTS
from grnpfn_real import real_folds

def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

rows = []
for ds in ("norman", "wessels", "sp4_A172", "sp4_T98G"):
    for fd in real_folds(ds, "pair", 301, 0):
        ix = {g: i for i, g in enumerate(fd["singles"])}; P = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps])
        ptr, pte = P(fd["tr"]), P(fd["te"]); D = fd["D"]
        Str, Ste = unit(D[ptr[:, 0]] + D[ptr[:, 1]]), unit(D[pte[:, 0]] + D[pte[:, 1]])
        Ytr = fd["R"]["train"]; mu = Ytr.mean(0)
        C = Ste @ Str.T; Ctr = Str @ Str.T; np.fill_diagonal(Ctr, -9)
        rf, rv = fd["R"]["fit"] - fd["R"]["fit"].mean(0), fd["R"]["val"] - fd["R"]["val"].mean(0)
        rep = np.sum(unit(rf) * unit(rv), 1)
        for i in range(len(pte)):
            rows.append(dict(dataset=ds, fold=fd["k"], role="test", nn_s=C[i].max(), R_norm=np.linalg.norm(fd["R_q_test"][i] - mu), rep=np.nan))
        for i in range(len(ptr)):
            rows.append(dict(dataset=ds, fold=fd["k"], role="train-LOO", nn_s=Ctr[i].max(), R_norm=np.linalg.norm(Ytr[i] - mu), rep=rep[i]))
d = pd.DataFrame(rows); d.to_csv(RESULTS / "x71_diag" / "similarity.csv", index=False)
print(d.groupby(["dataset", "role"]).agg(nn_s=("nn_s", "mean"), nn_s_med=("nn_s", "median"), R_norm=("R_norm", "mean"), rep=("rep", "mean")).round(3).to_string())
