"""Train GRN-PFN on the pre-generated simulation bank (synthetic only; no real screen is loaded).
Each draw: an episode, a random context/query split (pair setting, or with probability 0.35 zero-shot: queries are the
pairs touching held-out perturbations), random subsampling of context pairs. Loss: squared error of R̂ against the
independent test-third interaction R_te of the queries, normalized by the mean squared norm of the context targets.
  python -m grnpfn.train --bank ../data/grnpfn_bank --out ../eval/results/bench/grnpfn/grnpfn_v1.pt"""
from __future__ import annotations

import argparse
import glob
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .model import GRNPFN, LoopedGRNPFN, build, collate, partners, relations


def split(ep, rng, max_ctx=160, max_q=40):
    pairs = ep["pairs"]; M = len(pairs); P = len(ep["D"])
    if rng.random() < 0.35 and P > 8:
        hold = rng.choice(P, max(1, int(P * rng.uniform(0.1, 0.25))), replace=False)
        if len(ep["hubs"]):
            hold = np.setdiff1d(hold, ep["hubs"])
        hit = np.isin(pairs, hold).any(1); q = np.flatnonzero(hit); c = np.flatnonzero(~hit)
        if len(q) == 0 or len(c) < 8:
            q, c = None, None
    else:
        q = c = None
    if q is None:
        perm = rng.permutation(M); nq = max(1, int(M * rng.uniform(0.1, 0.3))); q, c = perm[:nq], perm[nq:]
    q = rng.permutation(q)[:max_q]
    c = rng.permutation(c)[: min(len(c), int(rng.integers(max(4, len(c) // 3), max_ctx + 1)))]
    return c, q


class Bank(torch.utils.data.IterableDataset):
    def __init__(self, files, seed, hybrid=False, rel=False, pa=False, extra=None, extra_p=0.0, ge=False, member=False):
        self.files, self.seed, self.hybrid, self.rel, self.pa, self.ge, self.member = files, seed, hybrid, rel, pa, ge, member
        self.extra, self.extra_p = extra, extra_p                                          # P4-U1: second bank (dense designs) mixed in with prob extra_p

    def __iter__(self):
        info = torch.utils.data.get_worker_info(); wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + 7919 * wid)
        nw = info.num_workers if info else 1
        while True:                                                                      # re-scan: shards generated during training join
            src = self.extra if (self.extra and rng.random() < self.extra_p) else self.files
            allf = sorted(glob.glob(f"{src}/shard_*.pkl"))
            files = [f for i, f in enumerate(allf) if i % nw == wid] if len(allf) >= nw else allf
            eps = pickle.load(open(files[rng.integers(len(files))], "rb"))
            for i in rng.permutation(len(eps)):
                ep = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float16 else v) for k, v in eps[i].items()}
                if self.hybrid:                                                          # P4-H1: screen-specific component (prior2)
                    from .prior import screen_specific
                    sp = screen_specific(ep, rng).astype(np.float32); ep["R_tr"] = ep["R_tr"] + sp; ep["R_te"] = ep["R_te"] + sp
                if self.ge:                                                              # P4-U2: global-epistasis component
                    from .prior import global_epistasis
                    ge = global_epistasis(ep, rng).astype(np.float32); ep["R_tr"] = ep["R_tr"] + ge; ep["R_te"] = ep["R_te"] + ge
                if self.member:                                                          # C3: member-additive interaction signatures
                    from .prior import member_signature
                    ms = member_signature(ep, rng).astype(np.float32); ep["R_tr"] = ep["R_tr"] + ms; ep["R_te"] = ep["R_te"] + ms
                c, q = split(ep, rng)
                pr = ep["pairs"]
                f, k, qs, s = build(ep["D"], ep["mu_ctrl"], pr[c], ep["R_tr"][c], pr[q], ep["n_single"], ep["n_pair"][c], ep["n_pair"][q], ep["n_ctrl"])
                Yc = ep["R_tr"][c] / s
                item = (f, k, qs, s, (ep["R_te"][q] / s).astype(np.float32), float(1.0 / ((Yc ** 2).sum(1).mean() + 1e-6)))
                yield item + ((relations(len(ep["D"]), pr[c], pr[q]),) if self.rel else ()) + ((partners(len(ep["D"]), pr[c], pr[q]),) if self.pa else ())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--hybrid", action="store_true"); ap.add_argument("--init", default=None); ap.add_argument("--rel", action="store_true")
    ap.add_argument("--var-head", action="store_true")                                  # P4-H5: posterior-predictive variance head (Gaussian NLL on detached residuals)
    ap.add_argument("--freeze-trunk", action="store_true")                              # train only the variance head
    ap.add_argument("--pa", action="store_true")
    ap.add_argument("--ge", action="store_true")                                        # P4-U2 global-epistasis prior
    ap.add_argument("--member", action="store_true")                                    # C3 member-signature prior
    ap.add_argument("--extra-bank", default=None); ap.add_argument("--extra-p", type=float, default=0.0)   # P4-U1 design-family prior                                        # P4-H6 partner-aligned attention (requires --rel)
    ap.add_argument("--ot-iters", type=int, default=0)
    ap.add_argument("--looped", action="store_true"); ap.add_argument("--tmin", type=int, default=4); ap.add_argument("--tmax", type=int, default=12)
    a = ap.parse_args()
    print(f"{len(glob.glob(f'{a.bank}/shard_*.pkl'))} shards at start (re-scanned during training)", flush=True)
    dev = "cuda"; torch.manual_seed(a.seed)
    if a.looped:
        cfg = dict(d=a.d, loops=a.tmax, looped=True); model = LoopedGRNPFN(d=a.d, loops=a.tmax).to(dev)
    else:
        cfg = dict(d=a.d, layers=a.layers, rel_bias=a.rel, ot_iters=a.ot_iters, var_head=a.var_head, partner_attn=a.pa); model = GRNPFN(**cfg).to(dev)
    assert not a.pa or a.rel, "--pa needs --rel (partner indices follow the relation codes in each item)"
    if a.init:                                                                          # continue from a pretrained checkpoint (new membership biases stay zero)
        miss = model.load_state_dict(torch.load(a.init, map_location=dev)["state_dict"], strict=False); print("init from", a.init, miss, flush=True)
    trng = np.random.default_rng(a.seed + 99)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    if a.freeze_trunk:                                                                  # mean predictions stay exactly those of --init
        for n, p_ in model.named_parameters():
            p_.requires_grad_(n.startswith("var."))
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03)
    dl = torch.utils.data.DataLoader(Bank(a.bank, a.seed, a.hybrid, a.rel, a.pa, a.extra_bank, a.extra_p, a.ge, a.member), batch_size=a.batch, collate_fn=collate, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    t0 = time.time(); run = 0.0; hist = []
    for step, b in enumerate(dl):
        if step >= a.steps:
            break
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if a.var_head:
                out, lv = model(b["feats"], b["keymask"], b["qsel"], rel=b.get("rel"), return_var=True)
            else:
                out = model(b["feats"], b["keymask"], b["qsel"], loops=int(trng.integers(a.tmin, a.tmax + 1))) if a.looped else model(b["feats"], b["keymask"], b["qsel"], rel=b.get("rel"), part=b.get("part"))
        r2 = (out.float() - b["target"]) ** 2
        err = r2.sum(-1)                                                                   # (B, Np)
        if a.var_head:                                                                     # Gaussian NLL of the residual of the (detached) mean
            lv = lv.float().clamp(-15, 5); err = err + 0.5 * (r2.detach() * torch.exp(-lv) + lv).mean(-1)
        loss = ((err * b["qsel"]).sum(1) / b["qsel"].sum(1) * b["w"]).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 250 == 0:
            hist.append((step, run)); print(f"step {step} loss {run:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step and step % 5000 == 0:
            torch.save(dict(state_dict=model.state_dict(), cfg=cfg, step=step, hist=hist), a.out)
    torch.save(dict(state_dict=model.state_dict(), cfg=cfg, step=a.steps, hist=hist), a.out); print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
