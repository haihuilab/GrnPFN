"""Pre-generate a bank of GRN-simulated episodes (synthetic only) in shards.
  python -m grnpfn.bank --out ../data/grnpfn_bank --n 20000 --procs 14 --seed 1000000"""
from __future__ import annotations

import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

KEYS = ("D", "mu_ctrl", "pairs", "R_tr", "R_te", "n_single", "n_pair", "n_ctrl", "kind", "hubs")


def shard(args):
    out, s, start, n, design = args
    f = Path(out) / f"shard_{s:04d}.pkl"
    if f.exists():
        return
    from grnpfn.sim import sample_grn_episode
    eps = []
    for i in range(start, start + n):
        ep = sample_grn_episode(np.random.default_rng(i), design=design)
        eps.append({k: (ep[k].astype(np.float16) if isinstance(ep[k], np.ndarray) and ep[k].dtype == np.float32 else ep[k]) for k in KEYS})
    tmp = f.with_suffix(".tmp"); pickle.dump(eps, open(tmp, "wb")); tmp.rename(f)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=14); ap.add_argument("--seed", type=int, default=1_000_000); ap.add_argument("--per", type=int, default=250)
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--design", default="random", choices=["random", "dense"])                # P4-U1 dense combinatorial designs
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(a.out, s, a.seed + s * a.per, a.per, a.design) for s in range(a.start_shard, a.start_shard + a.n // a.per)]
    with Pool(a.procs) as p:
        for i, _ in enumerate(p.imap_unordered(shard, jobs)):
            print(f"shard {i + 1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
