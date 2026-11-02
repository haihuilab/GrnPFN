"""P4 diagnosis (manuscript/notes/autonomous_log.md): does GRN-PFN v1 use the measured values of its context pairs?
Real screens (dev split 301, pair setting, all training pairs as context) and 60 simulated screens. Predictions of the
query pairs under: real context; context values shuffled across context pairs; context values set to zero; empty
context. Context utilisation = accuracy(real) − accuracy(shuffled).   python x61_pfn_context.py"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from grnpfn.model import load_grnpfn, predict
from grnpfn.sim import sample_grn_episode
from grnpfn_real import real_folds
from phase3 import score
from sklearn.model_selection import KFold


def variants(net, D, mu, pc, Y, pq, ns, npc, npq, nc, rng):
    P = lambda c, y: predict(net, D, mu, c, y, pq, ns, npc if len(c) else np.zeros(0), npq, nc)
    return {"real": P(pc, Y), "shuffled": P(pc, Y[rng.permutation(len(Y))]), "zero": P(pc, np.zeros_like(Y)), "empty": P(np.zeros((0, 2), int), np.zeros((0, Y.shape[1])))}


def real_one(ds):
    torch.set_num_threads(4); net = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_v1.pt"), "cpu"); rows = []
    for fd in real_folds(ds, "pair", 301, 0):
        ix = {g: i for i, g in enumerate(fd["singles"])}; pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps])
        Y = fd["R"]["train"]; rng = np.random.default_rng(fd["k"])
        V = variants(net, fd["D"], fd["mu"], pidx(fd["tr"]), Y, pidx(fd["te"]), fd["n_single"], np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float), fd["n_ctrl"], rng)
        for m, pr in V.items():
            rows.append(dict(source=ds, fold=fd["k"], variant=m, cos_pair=float(np.mean(score(pr, fd["R_q_test"], Y.mean(0))[0]))))
    return rows


def sim_one(e):
    torch.set_num_threads(1); net = load_grnpfn(str(RESULTS / "grnpfn" / "grnpfn_v1.pt"), "cpu"); rng = np.random.default_rng(8_950_000 + e)
    ep = sample_grn_episode(rng, kind=["crispra", "knockdown", "drug"][e % 3])
    itr, ite = next(iter(KFold(5, shuffle=True, random_state=e).split(ep["pairs"]))); pc, pq = ep["pairs"][itr], ep["pairs"][ite]; Y = ep["R_tr"][itr].astype(float)
    V = variants(net, ep["D"], ep["mu_ctrl"], pc, Y, pq, ep["n_single"], ep["n_pair"][itr], ep["n_pair"][ite], ep["n_ctrl"], rng)
    return [dict(source=f"sim-{ep['kind']}", fold=e, variant=m, cos_pair=float(np.mean(score(pr, ep["R_te"][ite].astype(float), Y.mean(0))[0]))) for m, pr in V.items()]


if __name__ == "__main__":
    with ProcessPoolExecutor(12) as ex:
        sims = ex.map(sim_one, range(60)); reals = ex.map(real_one, ["norman", "wessels", "sp4_A172", "sp4_T98G"])
        rows = [r for o in reals for r in o] + [r for o in sims for r in o]
    d = pd.DataFrame(rows); d.to_csv(RESULTS / "x61_pfn_context.csv", index=False)
    t = d.groupby(["source", "variant"]).cos_pair.mean().unstack(); t["PFN context use (real−shuffled)"] = t["real"] - t["shuffled"]
    pd.set_option("display.width", 220); print(t.round(3).to_string())
