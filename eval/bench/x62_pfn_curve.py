"""P4-H1 evaluation (manuscript/notes/autonomous_log.md): learning curves (random pairs, 3 draws, g ∈ {.15, .25, .40, 1})
and context use (all pairs: real − shuffled context values) for PFN checkpoints on identical subsets.
Dev split 301, pair setting.   python x62_pfn_curve.py --dataset norman --models grnpfn_v1 grnpfn_hybrid grnpfn_cont"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from grnpfn.model import load_grnpfn, predict
from grnpfn_real import real_folds
from phase3 import score

OUT = RESULTS / "x62_pfn_curve"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; nets = {m: load_grnpfn(str(RESULTS / "grnpfn" / f"{m}.pt"), dev) for m in a.models}; recs = []
    for fd in real_folds(a.dataset, "pair", 301, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; ix = {g: i for i, g in enumerate(fd["singles"])}
        pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te)
        npl, nte = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float)
        for g in (0.15, 0.25, 0.4, 1.0):
            for d in range(1 if g == 1.0 else 3):
                sel = np.arange(len(pool)) if g == 1.0 else np.sort(np.random.default_rng([k, d, int(100 * g)]).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
                Y = fd["R"]["train"][sel]; preds = {}
                for m, net in nets.items():
                    preds[m] = predict(net, fd["D"], fd["mu"], pool_ix[sel], Y, te_ix, fd["n_single"], npl[sel], nte, fd["n_ctrl"], device=dev)
                    if g == 1.0:
                        preds[m + " [shuffled ctx]"] = predict(net, fd["D"], fd["mu"], pool_ix[sel], Y[np.random.default_rng(k).permutation(len(Y))], te_ix, fd["n_single"], npl[sel], nte, fd["n_ctrl"], device=dev)
                        preds[m + " [empty ctx]"] = predict(net, fd["D"], fd["mu"], np.zeros((0, 2), int), np.zeros((0, Y.shape[1])), te_ix, fd["n_single"], np.zeros(0), nte, fd["n_ctrl"], device=dev)
                for m, pr in preds.items():
                    sp, sr = score(pr, fd["R_q_test"], Y.mean(0))
                    recs += [dict(dataset=a.dataset, fold=k, g=g, draw=d, AB=fd["key"](*t), method=m, cos_pair=x, cos_raw=y) for t, x, y in zip(te, sp, sr)]
        print(a.dataset, "fold", k, flush=True)
    d = pd.DataFrame(recs); OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / f"{a.dataset}_{'_'.join(a.models)}.csv", index=False)
    print(d.groupby(["g", "method"]).cos_pair.mean().unstack().round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
