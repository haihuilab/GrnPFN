"""C4 H-ADAPT (manuscript/notes/autonomous_log.md): two-stage adaptive PFN design vs one-shot vdesign and random
at identical budgets; every selection scored with the PFN (grnpfn_rel). Stage 1 = first max(10, b // 2) a-priori vdesign picks; stage 2 continues the same greedy expected-
variance-reduction selection with the stage-1 pairs' MEASURED R^train as context.
  python x76_adaptive_design.py --dataset norman   → results/bench/x76_adaptive/{ds}.csv"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from grnpfn.model import load_grnpfn
from grnpfn_real import real_folds
from phase3 import score
from x60_pfn_design import DEV, pfn_batch

OUT = RESULTS / "x76_adaptive"


def vdesign_from(net, fd, pool_ix, cand_ix, n_pool, n_cand, b, rng, S0=(), Y0=None, K=2):
    """x64 pfn_vdesign (same greedy, same draws per step) started from measured pairs S0 with values Y0"""
    G = fd["D"].shape[1]; S = list(S0); Ys = [y for y in Y0] if Y0 is not None else []
    while len(S) < b:
        ctx = pool_ix[S] if S else np.zeros((0, 2), int); Yc = np.array(Ys).reshape(len(S), G)
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
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); a = ap.parse_args()
    torch.set_num_threads(4); dnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel_var.pt"), DEV); pnet = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel.pt"), DEV)
    v = pd.read_csv(RESULTS / "x60_pfn_design" / f"{a.dataset}_s301_grnpfn_rel_var.csv"); recs, props = [], []
    for fd in real_folds(a.dataset, "pair", 301, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
        pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te); cand_ix = np.vstack([pool_ix, te_ix])
        n_pool, n_te = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float); n_cand = np.r_[n_pool, n_te]
        Rn = np.linalg.norm(fd["R"]["train"] - fd["R"]["train"].mean(0), axis=1); top = Rn >= np.quantile(Rn, 0.8)
        for g in (0.25, 0.4):
            b = max(10, int(round(g * len(pool)))); b1 = max(10, b // 2); sv = v[(v.fold == k) & (v.g == g)]
            first = vdesign_from(dnet, fd, pool_ix, cand_ix, n_pool, n_cand, b1, np.random.default_rng(k))      # = x64's first b1 picks
            ada = vdesign_from(dnet, fd, pool_ix, cand_ix, n_pool, n_cand, b, np.random.default_rng(1000 + k), S0=first, Y0=fd["R"]["train"][first])
            sels = {(s, 0): np.array(list(map(int, sv[sv.scheme == s].sel.iloc[0].split()))) for s in ("pfn-vdesign",)}
            sels[("pfn-adaptive", 0)] = np.sort(ada)
            ov = len(set(first) & set(sels[("pfn-vdesign", 0)].tolist())) / b1                                   # stage 1 vs the saved one-shot set
            for d in range(5):
                sels[("random", d)] = np.array(sorted(np.random.default_rng(10_000 * k + 100 * d + int(100 * g)).choice(len(pool), b, replace=False).tolist()))
            for (scheme, d), sel in sels.items():
                assert len(sel) == b, (scheme, len(sel), b)
                Yt = fd["R"]["train"][sel]
                preds = {"PFN": pfn_batch(pnet, fd, [(pool_ix[sel], Yt, te_ix, n_pool[sel], n_te)])[0]}
                for m, pr in preds.items():
                    sp, _ = score(pr, fd["R_q_test"], Yt.mean(0))
                    recs += [dict(dataset=a.dataset, fold=k, g=g, scheme=scheme, draw=d, AB=fd["key"](*t), method=m, cos_pair=x) for t, x in zip(te, sp)]
                props.append(dict(dataset=a.dataset, fold=k, g=g, scheme=scheme, draw=d, R_norm=Rn[sel].mean() / Rn.mean(), top20=top[sel].sum() / top.sum(),
                                  sel=" ".join(map(str, sel.tolist()))))
            print(a.dataset, "fold", k, "g", g, f"stage-1 overlap with saved one-shot {ov:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); d = pd.DataFrame(recs); d.to_csv(OUT / f"{a.dataset}.csv", index=False); pd.DataFrame(props).to_csv(OUT / f"{a.dataset}_props.csv", index=False)
    print(d.groupby(["g", "method", "scheme"]).cos_pair.mean().unstack().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
