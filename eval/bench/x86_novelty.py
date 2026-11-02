"""R3 (pre-registered 2026-10-28 in manuscript/notes/autonomous_log.md; descriptive).
  python x86_novelty.py --part novelty      R3-A: GrnPFN vs POLY by whether a test pair's perturbations occur among the measured pairs
  python x86_novelty.py --part confidence   R3-B: zero-shot agreement by GrnPFN's relative predictive uncertainty (GPU)
→ results/bench/x86_novelty/{novelty,confidence}.csv"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from grnpfn.model import load_grnpfn
from grnpfn_real import real_folds
from phase3 import SPENT, score
from x60_pfn_design import DEV, pfn_batch
from x77_zeroshot import DS

OUT = RESULTS / "x86_novelty"; SEEDS = (431, 439, 449, 457)


def perts(pair, drug):
    """perturbations that can be 'seen': genetic members; drug screens: the compound (any dose), anchor ignored"""
    return {x.split("@")[0] for x in pair if not x.startswith("tram")} if drug else set(pair)


def novelty():
    rows = []
    for ds, _ in DS:
        drug = ds.startswith("sp4"); folds = real_folds(ds, "pair", SEEDS[0], 0)
        pf = pd.concat([pd.read_csv(RESULTS / "x79_confirm" / f"curve_s{s}.csv") for s in SEEDS]); pf = pf[(pf.dataset == ds) & (pf.method == "PFN")]
        bl = pd.concat([pd.read_csv(RESULTS / "x80_baselines" / f"curve_{ds}_s{s}.csv") for s in SEEDS]); bl = bl[bl.method == "POLY"]
        for fd in folds:
            k = fd["k"]; pool, te = fd["tr"], fd["te"]; n = len(pool); key = {fd["key"](*t): t for t in te}
            for g in (0.15, 0.25, 0.4):
                for d in range(3):
                    sel = np.sort(np.random.default_rng([k, d, int(100 * g)]).choice(n, max(10, int(round(g * n))), replace=False))
                    seen = set().union(*[perts(pool[i], drug) for i in sel])
                    for ab, t in key.items():
                        p = perts(t, drug); ns = len(p & seen)
                        rows.append(dict(dataset=ds, fold=k, g=g, draw=d, AB=ab, n_perts=len(p), n_seen=ns, cls=("none seen" if ns == 0 else "seen")))
        lab = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        for m, df in (("GrnPFN", pf), ("POLY", bl)):
            x = df[df.g.isin([0.15, 0.25, 0.4])].merge(lab, on=["dataset", "fold", "g", "draw", "AB"], how="left")
            assert x.cls.notna().all()
            x.assign(method=m)[["dataset", "seed", "fold", "g", "draw", "AB", "method", "cos_pair", "cos_pair_testc", "n_perts", "n_seen", "cls"]].to_csv(
                OUT / f"novelty_{ds}_{m}.csv", index=False)
        print("novelty", ds, flush=True)


def confidence():
    net = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_rel_var.pt"), DEV); rows = []
    for s in SEEDS:
        for ds, _ in DS:
            for fd in real_folds(ds, "pair", s, 0):
                ix = {gn: i for i, gn in enumerate(fd["singles"])}; te_ix = np.array([(ix[a], ix[b]) for a, b in fd["te"]]); G = fd["D"].shape[1]
                F, V = pfn_batch(net, fd, [(np.zeros((0, 2), int), np.zeros((0, G)), te_ix, np.zeros(0), np.asarray(fd["n_pair_q"], float))], return_var=True)[0]
                obs = fd["R_q_test"]; ct = score(F, obs, obs.mean(0))[0]; f = F - F.mean(0)
                for i, t in enumerate(fd["te"]):
                    rows.append(dict(dataset=ds, seed=s, fold=fd["k"], AB=fd["key"](*t), cos_testc=ct[i], trV=float(V[i].sum()), sig=float(f[i] @ f[i]), rel_unc=float(V[i].sum() / (f[i] @ f[i] + 1e-12))))
            print("confidence", s, ds, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "confidence.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["novelty", "confidence"]); a = ap.parse_args()
    assert not set(SEEDS) & set(SPENT); OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    novelty() if a.part == "novelty" else confidence()


if __name__ == "__main__":
    main()
