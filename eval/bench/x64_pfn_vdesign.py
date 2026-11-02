"""P4-H5 (manuscript/notes/autonomous_log.md): uncertainty-aware PFN look-ahead design. Identical to the P4-1/P4-2
look-ahead (x60_pfn_design.pfn_design) except for the outcome perturbation and normalisation: each candidate's
hypothetical outcome is drawn from the PFN's own posterior predictive N(F_j, V_j) given the pairs chosen so far, and the
score is the expected relative reduction of predictive variance over the targets (law of total variance:
E||E[f_t | y_j] − E[f_t]||² = tr Var f_t − E tr Var(f_t | y_j)), estimated with K antithetic draws.
  python x64_pfn_vdesign.py --dataset sp4_T98G --model grnpfn_rel_var   → x60_pfn_design/{ds}_s301_{model}.csv"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from grnpfn.model import load_grnpfn
from grnpfn_real import real_folds
from phase3 import SPENT, score
from x60_pfn_design import DEV, OUT, pfn_batch


def pfn_vdesign(net, fd, pool_ix, cand_ix, n_pool, n_cand, b, rng, K=2):
    """greedy expected-variance-reduction design with the PFN's posterior predictive; returns pool indices in order"""
    G = fd["D"].shape[1]; empty = np.zeros((0, 2), int); S, Ys = [], []
    for _ in range(b):
        ctx = pool_ix[S] if S else empty; Yc = np.array(Ys).reshape(len(S), G)
        Fcur, Vcur = pfn_batch(net, fd, [(ctx, Yc, cand_ix, n_pool[S], n_cand)], return_var=True)[0]; den = Vcur.sum(1) + 1e-12
        cands = [j for j in range(len(pool_ix)) if j not in S]; Z = rng.normal(size=(K, G)); eps = []
        for j in cands:
            c2 = np.vstack([ctx, pool_ix[j:j + 1]]); nc = np.r_[n_pool[S], n_pool[j]]
            for z in Z:
                d = z * np.sqrt(Vcur[j])
                eps += [(c2, np.vstack([Yc, Fcur[j] + d]), cand_ix, nc, n_cand), (c2, np.vstack([Yc, Fcur[j] - d]), cand_ix, nc, n_cand)]
        P = pfn_batch(net, fd, eps)
        sc = [np.mean([np.sum(np.sum((P[2 * (K * i + r)] - P[2 * (K * i + r) + 1]) ** 2, 1) / 4 / den) for r in range(K)]) for i in range(len(cands))]
        j = cands[int(np.argmax(sc))]; S.append(j); Ys.append(Fcur[j])
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--model", required=True); a = ap.parse_args()
    assert 301 not in SPENT
    if DEV == "cuda" and os.environ.get("X60_MEMFRAC"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["X60_MEMFRAC"]))
    torch.set_num_threads(4); net = load_grnpfn(str(RESULTS / "grnpfn" / f"{a.model}.pt"), DEV); recs = []; P = f"PFN[{a.model}]"
    for fd in real_folds(a.dataset, "pair", 301, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
        pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te); cand_ix = np.vstack([pool_ix, te_ix])
        n_pool, n_te = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float); n_cand = np.r_[n_pool, n_te]
        order = pfn_vdesign(net, fd, pool_ix, cand_ix, n_pool, n_cand, max(10, int(round(0.4 * len(pool)))), np.random.default_rng(k))
        for g in (0.15, 0.25, 0.4):
            b = max(10, int(round(g * len(pool))))
            sels = {("pfn-vdesign", 0): sorted(order[:b])}
            for d in range(5):
                sels[("random", d)] = sorted(np.random.default_rng(10_000 * k + 100 * d + int(100 * g)).choice(len(pool), b, replace=False).tolist())
            for (scheme, d), sel in sels.items():
                sel = np.asarray(sel); Yt = fd["R"]["train"][sel]
                Fq = pfn_batch(net, fd, [(pool_ix[sel], Yt, te_ix, n_pool[sel], n_te)])[0]; sp, sr = score(Fq, fd["R_q_test"], Yt.mean(0))
                recs += [dict(dataset=a.dataset, seed=301, fold=k, g=g, scheme=scheme, draw=d, AB=fd["key"](*t), method=P, cos_pair=x, cos_raw=y, sel=" ".join(map(str, sel.tolist())) if scheme != "random" else "")
                         for t, x, y in zip(te, sp, sr)]
            print(a.dataset, "fold", k, "g", g, flush=True)
    d = pd.DataFrame(recs); OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / f"{a.dataset}_s301_{a.model}.csv", index=False)
    print(d.groupby(["g", "scheme"]).cos_pair.mean().unstack().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
