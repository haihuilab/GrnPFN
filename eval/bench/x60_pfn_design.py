"""P4-1 (manuscript/notes/autonomous_log.md): PFN-native a-priori design of which pairs to measure, and PFN prediction.
No kernel ridge in the method.
  python x60_pfn_design.py --dataset norman      → eval/results/bench/x60_pfn_design/{dataset}_s{seed}.csv"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from grnpfn.model import build, collate, load_grnpfn, partners, relations
from grnpfn_real import real_folds
from phase3 import SPENT, score

OUT = RESULTS / "x60_pfn_design"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def pfn_batch(net, fd, episodes, bs=int(os.environ.get("X60_BS", 12)), return_var=False):
    """episodes: list of (ctx_idx (k,2), Yctx (k,G), q_idx (m,2), n_ctx (k,), n_q (m,)) → list of (m, G) predictions"""
    out = []
    for i in range(0, len(episodes), bs):
        ep = episodes[i:i + bs]; use_rel = getattr(net, "rel", None) is not None; use_pa = getattr(net, "partner_attn", False)
        items = [build(fd["D"], fd["mu"], c, Y, q, fd["n_single"], nc, nq, fd["n_ctrl"]) for c, Y, q, nc, nq in ep]
        b = collate([(f, k, qs, s, None, 1.0) + ((relations(len(fd["D"]), e[0], e[2]),) if use_rel else ()) + ((partners(len(fd["D"]), e[0], e[2]),) if use_pa else ())
                     for (f, k, qs, s), e in zip(items, ep)])
        kw = dict(rel=b["rel"].to(DEV)) if use_rel else {}
        if use_pa:
            kw["part"] = b["part"].to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            if return_var:                                                                  # (mean, predictive variance) in target units
                o, lv = net(b["feats"].to(DEV), b["keymask"].to(DEV), b["qsel"].to(DEV), return_var=True, **kw)
                o, v = o.float().cpu().numpy(), np.exp(lv.float().cpu().numpy())
                out += [(o[j][: len(qs)][qs] * s, v[j][: len(qs)][qs] * s ** 2) for j, (f, k, qs, s) in enumerate(items)]
                continue
            o = net(b["feats"].to(DEV), b["keymask"].to(DEV), b["qsel"].to(DEV), **kw).float().cpu().numpy()
        out += [o[j][: len(qs)][qs] * s for j, (f, k, qs, s) in enumerate(items)]           # outputs padded to the batch max
    return out


def pfn_design(net, fd, pool_ix, cand_ix, n_pool, n_cand, b, rng):
    """greedy PFN look-ahead design; pool_ix (n,2) measurable pairs (first n of cand_ix), cand_ix (m,2) all candidates"""
    G = fd["D"].shape[1]; empty = np.zeros((0, 2), int)
    F0 = pfn_batch(net, fd, [(empty, np.zeros((0, G)), cand_ix, np.zeros(0), n_cand)])[0]; sig = F0.std(0) + 1e-8
    S, Ys = [], []
    for _ in range(b):
        ctx = pool_ix[S] if S else empty; Yc = np.array(Ys).reshape(len(S), G)
        Fcur = pfn_batch(net, fd, [(ctx, Yc, cand_ix, n_pool[S], n_cand)])[0]; den = np.sum(Fcur ** 2, 1) + 1e-12
        cands = [j for j in range(len(pool_ix)) if j not in S]; delta = rng.normal(size=G) * sig; eps = []
        for j in cands:
            c2 = np.vstack([ctx, pool_ix[j:j + 1]]); nc = np.r_[n_pool[S], n_pool[j]]
            eps += [(c2, np.vstack([Yc, Fcur[j] + delta]), cand_ix, nc, n_cand), (c2, np.vstack([Yc, Fcur[j] - delta]), cand_ix, nc, n_cand)]
        P = pfn_batch(net, fd, eps); sc = [np.sum(np.sum((P[2 * i] - P[2 * i + 1]) ** 2, 1) / den) for i in range(len(cands))]
        j = cands[int(np.argmax(sc))]; S.append(j); Ys.append(Fcur[j])
    return S                                                                            # in selection order (prefixes = smaller budgets)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--split-seed", type=int, default=301)
    ap.add_argument("--model", default="grnpfn_v1"); a = ap.parse_args()
    assert a.split_seed not in SPENT
    if DEV == "cuda" and os.environ.get("X60_MEMFRAC"):                                     # share the GPU with a training run safely
        torch.cuda.set_per_process_memory_fraction(float(os.environ["X60_MEMFRAC"]))
    torch.set_num_threads(4); net = load_grnpfn(str(RESULTS / "grnpfn" / f"{a.model}.pt"), DEV); recs = []
    for fd in real_folds(a.dataset, "pair", a.split_seed, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; ix = {gn: i for i, gn in enumerate(fd["singles"])}
        pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te); cand_ix = np.vstack([pool_ix, te_ix])
        n_pool, n_te = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float); n_cand = np.r_[n_pool, n_te]
        order = pfn_design(net, fd, pool_ix, cand_ix, n_pool, n_cand, max(10, int(round(0.4 * len(pool)))), np.random.default_rng(k))
        for g in (0.15, 0.25, 0.4):
            b = max(10, int(round(g * len(pool))))
            sels = {("pfn-design", 0): sorted(order[:b])}
            for d in range(5):
                sels[("random", d)] = sorted(np.random.default_rng(10_000 * k + 100 * d + int(100 * g)).choice(len(pool), b, replace=False).tolist())
            for (scheme, d), sel in sels.items():
                sel = np.asarray(sel); Yt = fd["R"]["train"][sel]
                Fq = pfn_batch(net, fd, [(pool_ix[sel], Yt, te_ix, n_pool[sel], n_te)])[0]
                preds = {"PFN": Fq}
                for m, pr in preds.items():
                    sp, sr = score(pr, fd["R_q_test"], Yt.mean(0))
                    recs += [dict(dataset=a.dataset, seed=a.split_seed, fold=k, g=g, scheme=scheme, draw=d, AB=fd["key"](*t), method=m if m != "PFN" or a.model == "grnpfn_v1" else f"PFN[{a.model}]", cos_pair=x, cos_raw=y) for t, x, y in zip(te, sp, sr)]
            print(a.dataset, "fold", k, "g", g, flush=True)
    d = pd.DataFrame(recs); OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / (f"{a.dataset}_s{a.split_seed}.csv" if a.model == "grnpfn_v1" else f"{a.dataset}_s{a.split_seed}_{a.model}.csv"), index=False)
    print(d.groupby(["g", "method", "scheme"]).cos_pair.mean().unstack().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
