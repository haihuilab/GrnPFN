"""C3 (pre-registered, manuscript/notes/grnpfn_log.md): continue training GRN-PFN v1 on real episodes from the four
screens other than the target (50%) mixed with simulation-bank episodes (50%); synthetic split rule and loss; lr 1e-4,
batch 4, 3,000 steps. The target screen is never loaded.
  python x13_realmeta.py --target norman
Writes eval/results/bench/grnpfn/grnpfn_meta_{target}.pt"""
from __future__ import annotations

import argparse
import glob
import pickle
import time

import numpy as np
import torch

from common import DATA, RESULTS
from grnpfn.model import GRNPFN, build, collate
from grnpfn.train import split

SCREENS = ("norman", "wessels", "sp4_A172", "sp4_T98G", "sp4_U87MG")


def item(ep, rng):
    ep = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float16 else v) for k, v in ep.items()}
    c, q = split(ep, rng); pr = ep["pairs"]
    f, k, qs, s = build(ep["D"], ep["mu_ctrl"], pr[c], ep["R_tr"][c], pr[q], ep["n_single"], ep["n_pair"][c], ep["n_pair"][q], ep["n_ctrl"])
    Yc = ep["R_tr"][c] / s
    return f, k, qs, s, (ep["R_te"][q] / s).astype(np.float32), float(1.0 / ((Yc ** 2).sum(1).mean() + 1e-6))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=11); a = ap.parse_args()
    real = [e for ds in SCREENS if ds != a.target for e in pickle.load(open(DATA / "grnpfn_real_pool" / f"{ds}.pkl", "rb"))]
    assert all(e["screen"] != a.target for e in real)
    rng = np.random.default_rng(a.seed); shards = sorted(glob.glob(str(DATA / "grnpfn_bank" / "shard_*.pkl")))
    sim = [e for f in rng.choice(shards, 8, replace=False) for e in pickle.load(open(f, "rb"))]
    print(f"target {a.target}: {len(real)} real episodes from {sorted({e['screen'] for e in real})}, {len(sim)} simulated", flush=True)
    dev = "cuda"; torch.manual_seed(a.seed); ck = torch.load(RESULTS / "grnpfn" / "grnpfn_v1.pt", map_location=dev)
    model = GRNPFN(**ck["cfg"]).to(dev); model.load_state_dict(ck["state_dict"]); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); t0 = time.time(); run = 0.0
    for step in range(a.steps):
        items = [item(real[rng.integers(len(real))] if rng.random() < 0.5 else sim[rng.integers(len(sim))], rng) for _ in range(4)]
        b = {k: v.to(dev) for k, v in collate(items).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(b["feats"], b["keymask"], b["qsel"])
        err = ((out.float() - b["target"]) ** 2).sum(-1); loss = ((err * b["qsel"]).sum(1) / b["qsel"].sum(1) * b["w"]).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 250 == 0:
            print(f"step {step} loss {run:.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = RESULTS / "grnpfn" / f"grnpfn_meta_{a.target}.pt"
    torch.save(dict(state_dict=model.state_dict(), cfg=ck["cfg"], step=a.steps, base="grnpfn_v1.pt", target_excluded=a.target), out); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
