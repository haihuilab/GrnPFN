"""C3 real episode pool (pre-registered, manuscript/notes/grnpfn_log.md): every screen × build seeds 401–405 ×
rotations 0–2, all pairs, built as Gate 3, stored in the simulation-bank episode format.
  python x13_real_pool.py
Writes data/grnpfn_real_pool/{dataset}.pkl (list of episodes)"""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DATA
from grnpfn_real import real_folds

SCREENS = {"norman": "crispra", "wessels": "knockdown", "sp4_A172": "drug", "sp4_T98G": "drug", "sp4_U87MG": "drug"}
OUT = DATA / "grnpfn_real_pool"


def one(args):
    ds, seed, rot = args
    fd = real_folds(ds, "pair", seed, rot)[0]; f = fd["full"]
    n_pair = {fd["key"](*t): n for t, n in zip(fd["tr"], fd["n_pair_c"])} | {fd["key"](*t): n for t, n in zip(fd["te"], fd["n_pair_q"])}
    names = fd["singles"]
    return dict(D=f["D"].astype(np.float32), mu_ctrl=f["mu"].astype(np.float32), pairs=f["pairs"], R_tr=f["R_tr"].astype(np.float32),
                R_te=f["R_te"].astype(np.float32), n_single=fd["n_single"], n_pair=np.array([n_pair[fd["key"](names[a], names[b])] for a, b in f["pairs"]]),
                n_ctrl=fd["n_ctrl"], kind=SCREENS[ds], hubs=fd["hubs"], screen=ds, seed=seed, rotation=rot)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, s, r) for ds in SCREENS for s in (401, 402, 403, 404, 405) for r in range(3)]
    with ProcessPoolExecutor(15) as ex:
        eps = list(ex.map(one, jobs))
    for ds in SCREENS:
        e = [x for x in eps if x["screen"] == ds]; pickle.dump(e, open(OUT / f"{ds}.pkl", "wb"))
        print(ds, len(e), "episodes;", e[0]["D"].shape, "singles×genes;", len(e[0]["pairs"]), "pairs", flush=True)
