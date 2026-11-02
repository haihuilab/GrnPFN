"""Cycle-3 Phase 0 (manuscript/notes/c3_plan.md): learnability profile of real screens vs simulated prior screens.
TRAINING pairs only (real: fold-0 training pairs, train-third cells; no test-third cell is read for any statistic).
Context version / target version of R: real R^fit / R^val (disjoint single/control halves; pair cells shared),
simulated R_tr / R_te (independent noise). Statistics:
  inner curve : 5 random 75/25 splits of the pairs; context = g ∈ {.15, 1} of the 75% (R^ctx), pseudo-targets = the 25%
                (R^tgt); pair-specific cosine of PFNs
  rank        : signal spectrum = eigenvalues of the symmetrised cross-covariance of centred R^ctx and R^tgt;
                participation ratio of its positive part; variance share of the top 1 / 3 programs
  transfer    : cross-pair signal cosine c_ij = cos(R^ctx_i − mean, R^tgt_j − mean), i ≠ j: mean |c|, mean c for pairs
                sharing a member vs not, Spearman(c_ij, cos of the additive profiles Δ_A + Δ_B)
  python x74_learnability.py --real norman wessels ... --sim 48 --models grnpfn_rel  → results/bench/x74_learn/"""
from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import RESULTS
from grnpfn.model import load_grnpfn, predict
from grnpfn.prior import bilinear_response, member_signature, screen_specific
from grnpfn.sim import sample_grn_episode
from phase3 import score

OUT = RESULTS / "x74_learn"; unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
KIND = {"norman": "CRISPRa", "wessels": "knockdown", "frangieh_pairs": "knockdown"}


def profile(D, mu, pairs, Rc, Rt, ns, npair, nctrl, nets, rng, n_split=5):
    M = len(pairs); out = {}
    # rank
    A, B = Rc - Rc.mean(0), Rt - Rt.mean(0); C = (A.T @ B + B.T @ A) / (2 * M)
    ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None); out["pr"] = ev.sum() ** 2 / (ev ** 2).sum(); out["top1"] = ev[0] / ev.sum(); out["top3"] = ev[:3].sum() / ev.sum()
    # transfer
    Cc = unit(A) @ unit(B).T; off = ~np.eye(M, dtype=bool)
    share = (pairs[:, None, 0] == pairs[None, :, 0]) | (pairs[:, None, 0] == pairs[None, :, 1]) | (pairs[:, None, 1] == pairs[None, :, 0]) | (pairs[:, None, 1] == pairs[None, :, 1])
    S = unit(D[pairs[:, 0]] + D[pairs[:, 1]]); Sc = S @ S.T
    out["abs_c"] = np.abs(Cc[off]).mean(); out["c_share"] = Cc[off & share].mean() if (off & share).any() else np.nan
    out["c_noshare"] = Cc[off & ~share].mean() if (off & ~share).any() else np.nan; out["rho_add"] = spearmanr(Cc[off], Sc[off])[0]
    out["frac_share"] = (off & share).sum() / off.sum()
    # inner curve
    acc = {}
    for s in range(n_split):
        perm = rng.permutation(M); nq = max(5, M // 4); q, pool = perm[:nq], perm[nq:]
        for g in (0.15, 1.0):
            sel = pool if g == 1.0 else np.sort(rng.choice(pool, max(8, int(round(g * len(pool)))), replace=False))
            Y, obs = Rc[sel], Rt[q]; sc = lambda pr: float(np.mean(score(pr, obs, Y.mean(0))[0]))
            for m, net in nets.items():
                acc.setdefault((m, g), []).append(sc(predict(net, D, mu, pairs[sel], Y, pairs[q], ns, npair[sel], npair[q], nctrl)))
    for (m, g), v in acc.items():
        out[f"{m}@{g}"] = float(np.mean(v))
    for m in nets:
        out[f"{m} gain"] = out[f"{m}@1.0"] - out[f"{m}@0.15"]
    return out


def sim_one(args):
    i, models, extra = args
    torch.set_num_threads(1); nets = {m: load_grnpfn(str(RESULTS / "grnpfn" / f"{m}.pt"), "cpu") for m in models}
    rng = np.random.default_rng(9_000_000 + i); ep = sample_grn_episode(rng); sp = screen_specific(ep, rng)
    if "member" in extra:                                                                 # C3 candidate component (own stream)
        sp = sp + member_signature(ep, np.random.default_rng(19_000_000 + i), **extra["member"])
    if "bilinear" in extra:
        sp = sp + bilinear_response(ep, np.random.default_rng(29_000_000 + i), **extra["bilinear"])
    o = profile(ep["D"], ep["mu_ctrl"], ep["pairs"], ep["R_tr"] + sp, ep["R_te"] + sp, ep["n_single"], ep["n_pair"], ep["n_ctrl"], nets, rng)
    return dict(source="sim", screen=i, kind=ep["kind"], M=len(ep["pairs"]), P=len(ep["D"]), **o)


def semi_one(ds, models, rep):
    """semi-synthetic: the screen's REAL single profiles, design and per-pair noise level (training side), synthetic
    interactions from the member + bilinear components (scaled to the real R^fit power) + Gaussian noise with the real
    per-gene sampling variance, drawn independently for context and target versions"""
    from grnpfn_real import real_folds
    nets = {m: load_grnpfn(str(RESULTS / "grnpfn" / f"{m}.pt"), "cpu") for m in models}
    fd = real_folds(ds, "pair", 301, 0)[0]; out = []
    for i in range(rep):
        rng = np.random.default_rng(39_000_000 + 1000 * i + sum(map(ord, ds)))
        ep = dict(D=fd["D"], pairs=fd["pc"], R_tr=fd["R"]["fit"])
        sig = member_signature(ep, rng, p0=0.0) + bilinear_response(ep, rng, p0=0.0)
        sig = sig * np.sqrt((fd["R"]["fit"] ** 2).sum(1).mean() / ((sig ** 2).sum(1).mean() + 1e-12)) * 0.5   # signal = half the real R power
        sd = np.sqrt(fd["noise_tr_g"]); Rc, Rt = sig + rng.normal(size=sig.shape) * sd, sig + rng.normal(size=sig.shape) * sd
        o = profile(fd["D"], fd["mu"], fd["pc"], Rc, Rt, fd["n_single"], fd["n_pair_c"], fd["n_ctrl"], nets, rng, n_split=3)
        out.append(dict(source="semi", screen=ds, kind=KIND.get(ds, "drug"), rep=i, M=len(fd["pc"]), P=len(fd["D"]), **o))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--real", nargs="*", default=[]); ap.add_argument("--sim", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["grnpfn_rel"]); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--tag", default="base")
    ap.add_argument("--member", default=None, help="json kwargs of member_signature, e.g. '{}'")
    ap.add_argument("--bilinear", default=None, help="json kwargs of bilinear_response")
    ap.add_argument("--semi", nargs="*", default=[]); ap.add_argument("--semi-rep", type=int, default=8)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); rows = []
    if a.real:
        from grnpfn_real import real_folds
        nets = {m: load_grnpfn(str(RESULTS / "grnpfn" / f"{m}.pt"), "cpu") for m in a.models}
        for ds in a.real:
            fd = real_folds(ds, "pair", 301, 0)[0]; rng = np.random.default_rng(301)
            o = profile(fd["D"], fd["mu"], fd["pc"], fd["R"]["fit"], fd["R"]["val"], fd["n_single"], fd["n_pair_c"], fd["n_ctrl"], nets, rng)
            rows.append(dict(source="real", screen=ds, kind=KIND.get(ds, "drug"), M=len(fd["pc"]), P=len(fd["D"]), **o)); print(ds, "done", flush=True)
    if a.sim:
        with Pool(a.procs) as p:
            import json
            ex = {"member": json.loads(a.member)} if a.member is not None else {}
            if a.bilinear is not None:
                ex["bilinear"] = json.loads(a.bilinear)
            rows += p.map(sim_one, [(i, a.models, ex) for i in range(a.sim)])
    if a.semi:
        with Pool(min(a.procs, len(a.semi))) as p:
            rows += [r for rs in p.starmap(semi_one, [(ds, a.models, a.semi_rep) for ds in a.semi]) for r in rs]
    d = pd.DataFrame(rows); d.to_csv(OUT / f"{a.tag}.csv", index=False)
    if a.semi:
        print(d[d.source == "semi"].groupby("screen")[[c for c in d.columns if "@" in c or "gain" in c]].mean().round(3).to_string())
    pd.set_option("display.width", 250); cols = [c for c in d.columns if c not in ("source", "screen", "kind")]
    print(d[d.source == "real"].set_index("screen")[cols].round(3).to_string())
    if a.sim:
        print(d[d.source == "sim"].groupby("kind")[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
