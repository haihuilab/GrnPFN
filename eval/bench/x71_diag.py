"""Cycle-2 diagnosis (manuscript/notes/autonomous_log.md; dev 301; CPU only).
D1 support: shared-member support of test pairs vs of inner-validation-like pairs (training pairs, leave-one-out).
D2 TTT gain (U3) by test-pair support class.
D3 member-additive structure: ridge R_AB ≈ c + m_A + m_B fitted on training pairs, scored on test pairs (diagnostic).
D4 design selections: member coverage of test pairs, norm of chosen interactions, distinct members.
python x71_diag.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS
from grnpfn_real import real_folds
from phase3 import score

SCREENS = ("norman", "wessels", "sp4_A172", "sp4_T98G")
OUT = RESULTS / "x71_diag"


def degree(pairs):
    return pd.Series([x for p in pairs for x in p]).value_counts()


def member_additive(tr, Y, te, lam=1.0):
    mem = sorted({x for p in tr + te for x in p}); ix = {m: i for i, m in enumerate(mem)}
    X = np.zeros((len(tr), len(mem) + 1)); X[:, -1] = 1
    for i, (a, b) in enumerate(tr):
        X[i, ix[a]] += 1; X[i, ix[b]] += 1
    reg = np.eye(X.shape[1]) * lam; reg[-1, -1] = 1e-6
    W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    Xt = np.zeros((len(te), len(mem) + 1)); Xt[:, -1] = 1
    for i, (a, b) in enumerate(te):
        Xt[i, ix[a]] += 1; Xt[i, ix[b]] += 1
    return Xt @ W


def main():
    OUT.mkdir(parents=True, exist_ok=True); rows, sup = [], []
    for ds in SCREENS:
        for fd in real_folds(ds, "pair", 301, 0):
            k, tr, te = fd["k"], fd["tr"], fd["te"]; dg = degree(tr)
            for t in te:
                da, db = int(dg.get(t[0], 0)), int(dg.get(t[1], 0))
                sup.append(dict(dataset=ds, fold=k, role="test", AB=fd["key"](*t), n_share=da + db, min_deg=min(da, db)))
            for i, t in enumerate(tr):                      # leave-one-out support of a training pair (as an inner-validation pair)
                da, db = int(dg.get(t[0], 0)) - 1, int(dg.get(t[1], 0)) - 1
                sup.append(dict(dataset=ds, fold=k, role="train-LOO", AB=fd["key"](*t), n_share=da + db, min_deg=min(da, db)))
            # D3: member-additive ridge vs mean-only, on test pairs (pair-specific cosine)
            Y = fd["R"]["train"]
            for lam in (0.3, 1.0, 3.0):
                P = member_additive(list(tr), Y, list(te), lam); sp, _ = score(P, fd["R_q_test"], Y.mean(0))
                rows += [dict(dataset=ds, fold=k, lam=lam, AB=fd["key"](*t), cos_pair=x) for t, x in zip(te, sp)]
            print(ds, k, flush=True)
    s = pd.DataFrame(sup); s.to_csv(OUT / "support.csv", index=False); m = pd.DataFrame(rows); m.to_csv(OUT / "member_additive.csv", index=False)
    print("\nD1 support (mean n_share; fraction with min_deg = 0)")
    print(s.groupby(["dataset", "role"]).agg(n_share=("n_share", "mean"), unsupported=("min_deg", lambda x: np.mean(x == 0)), n=("AB", "size")).round(3).to_string())
    print("\nD3 member-additive ridge on test pairs (all training pairs), pair-specific cosine")
    print(m.groupby(["dataset", "lam"]).cos_pair.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
