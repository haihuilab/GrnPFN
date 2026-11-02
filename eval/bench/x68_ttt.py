"""U3 (manuscript/notes/autonomous_log.md): test-time training of the PFN on the measured pairs of the screen itself.
Episodes are drawn from the context pairs only (random pair split, 20% pseudo-queries). Pseudo-context interactions come
from one half of the training-target cells (R^fit) and pseudo-query targets from the other half (R^val): interactions of
different pairs share the single and control cells, so R^train for both would reward fitting that shared noise. The number of steps is chosen by inner stopping:
TTT on 80% of the context, scored on the held-out 20% (pair-specific cosine, context = the 80%), steps ∈ {0,10,..,S};
then TTT from the base weights on all context pairs for s* steps and predict the test pairs. s* = 0 ⇒ base model.
Same folds and random draws as x62. --sham permutes the context interactions among the pairs inside every TTT episode
(mechanism control).   python x68_ttt.py --dataset sp4_T98G --model grnpfn_rel"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS
from grnpfn.model import build, collate, load_grnpfn, partners, predict, relations
from grnpfn_real import real_folds
from phase3 import score

OUT = RESULTS / "x68_ttt"


def item(net, fd, pool_ix, npl, Yall, ctx, q, rng, sham):
    Yf, Yv = fd["R"]["fit"], fd["R"]["val"]; Yc = Yf[ctx][rng.permutation(len(ctx))] if sham else Yf[ctx]
    f, k, qs, s = build(fd["D"], fd["mu"], pool_ix[ctx], Yc, pool_ix[q], fd["n_single"], npl[ctx], npl[q], fd["n_ctrl"])
    it = (f, k, qs, s, (Yv[q] / s).astype(np.float32), 1.0)
    if getattr(net, "rel", None) is not None:
        it += (relations(len(fd["D"]), pool_ix[ctx], pool_ix[q]),)
    if getattr(net, "partner_attn", False):
        it += (partners(len(fd["D"]), pool_ix[ctx], pool_ix[q]),)
    return it


def ttt(base, fd, pool_ix, npl, Yall, C, steps, rng, a, dev):
    """fine-tune a copy of base on episodes drawn from context indices C (into pool)"""
    net = copy.deepcopy(base).train()
    if steps == 0:
        return net.eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    for _ in range(steps):
        its = []
        for _ in range(a.bs):
            p = rng.permutation(C); nq = max(2, int(round(0.2 * len(C)))); its.append(item(net, fd, pool_ix, npl, Yall, p[nq:], p[:nq], rng, a.sham))
        b = {k: v.to(dev) for k, v in collate(its).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
            out = net(b["feats"], b["keymask"], b["qsel"], rel=b.get("rel"), part=b.get("part"))
        err = ((out.float() - b["target"]) ** 2).sum(-1)
        loss = ((err * b["qsel"]).sum(1) / b["qsel"].sum(1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return net.eval()


def pred(net, fd, pool_ix, npl, Yall, ctx, q_ix, nq, dev):
    return predict(net, fd["D"], fd["mu"], pool_ix[ctx], Yall[ctx], q_ix, fd["n_single"], npl[ctx], nq, fd["n_ctrl"], device=dev)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--model", default="grnpfn_rel")
    ap.add_argument("--S", type=int, default=60); ap.add_argument("--every", type=int, default=10); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=int(os.environ.get("X60_BS", 2))); ap.add_argument("--sham", action="store_true")
    ap.add_argument("--sham-pairs", action="store_true")    # post-hoc control: TTT data (R^fit, R^val) reassigned to the wrong pairs by one permutation
    ap.add_argument("--split3", action="store_true")    # U5: independent cell thirds — TTT context R^f3, TTT targets R^v3, inner-stopping targets R^s3
    ap.add_argument("--cv", type=int, default=0)    # C2 H-CVS: s* from the inner curve averaged over K inner folds (0 = U3 single 80/20 split)
    ap.add_argument("--deploy-ctx", default="train", choices=["train", "fit"])    # diagnosis: context interactions used at prediction time
    ap.add_argument("--gs", type=float, nargs="+", default=[0.15, 0.25, 0.4, 1.0]); ap.add_argument("--folds", type=int, default=99); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and os.environ.get("X60_MEMFRAC"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["X60_MEMFRAC"]))
    base = load_grnpfn(str(RESULTS / "grnpfn" / f"{a.model}.pt"), dev); recs, stops = [], []; t0 = time.time()
    tag = f"{a.model}{'_sham' if a.sham else ''}{'_shampairs' if a.sham_pairs else ''}{'_split3' if a.split3 else ''}{'_dfit' if a.deploy_ctx == 'fit' else ''}{f'_cv{a.cv}' if a.cv else ''}"
    for fd in real_folds(a.dataset, "pair", 301, 0):
        k = fd["k"]
        if k >= a.folds:
            break
        pool, te = fd["tr"], fd["te"]; ix = {g: i for i, g in enumerate(fd["singles"])}
        pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te)
        npl, nte = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float); Yall = fd["R"]["train"]
        for g in a.gs:
            for d in range(1 if g == 1.0 else 3):
                sel = np.arange(len(pool)) if g == 1.0 else np.sort(np.random.default_rng([k, d, int(100 * g)]).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
                rng = np.random.default_rng([k, d, int(100 * g), 68]); p = rng.permutation(sel); nv = max(2, int(round(0.2 * len(sel))))
                V, C1 = np.sort(p[:nv]), np.sort(p[nv:])
                fdt = fd
                if a.split3:
                    fdt = dict(fd); fdt["R"] = dict(fd["R"], fit=fd["R"]["f3"], val=fd["R"]["v3"], stop=fd["R"]["s3"])
                if a.sham_pairs:                                                     # interactions of the drawn pairs shuffled among them (fit and val alike)
                    pi = rng.permutation(sel); fdt = dict(fdt); fdt["R"] = {r: v.copy() for r, v in fdt["R"].items()}
                    for r in ("fit", "val"):
                        fdt["R"][r][sel] = fdt["R"][r][pi]
                # inner stopping: TTT trajectory on C1, scored on V every `every` steps (U3: one 80/20 split; --cv K: curve averaged over K folds)
                splits = [(V, C1)] if not a.cv else [(np.sort(f), np.sort(np.setdiff1d(sel, f))) for f in np.array_split(p, a.cv)]
                curves = []
                for V, C1 in splits:
                    net = copy.deepcopy(base); c = {}
                    for s in range(0, a.S + 1, a.every):
                        if s:
                            net = ttt(net, fdt, pool_ix, npl, Yall, C1, a.every, rng, a, dev)
                        pv = pred(net, fdt, pool_ix, npl, fdt["R"]["fit"], C1, pool_ix[V], npl[V], dev)
                        Ys = fdt["R"].get("stop", fdt["R"]["val"]); c[s] = float(np.mean(score(pv, Ys[V], Ys[C1].mean(0))[0]))
                    curves.append(c)
                sc = {s: float(np.mean([c[s] for c in curves])) for s in curves[0]}
                sstar = max(sc, key=lambda s: (round(sc[s], 4), -s))
                stops.append(dict(dataset=a.dataset, fold=k, g=g, draw=d, sstar=sstar, **{f"v{s}": v for s, v in sc.items()}))
                Yd = fd["R"][a.deploy_ctx]
                preds = {f"{a.model}{'_dfit' if a.deploy_ctx == 'fit' else ''} [base]": pred(base, fd, pool_ix, npl, Yd, sel, te_ix, nte, dev)}
                net = ttt(base, fdt, pool_ix, npl, Yall, sel, sstar, rng, a, dev)
                preds[f"{tag} [TTT]"] = pred(net, fd, pool_ix, npl, Yd, sel, te_ix, nte, dev)
                for m, pr in preds.items():
                    sp, sr = score(pr, fd["R_q_test"], Yall[sel].mean(0))
                    recs += [dict(dataset=a.dataset, fold=k, g=g, draw=d, AB=fd["key"](*t), method=m, cos_pair=x, cos_raw=y) for t, x, y in zip(te, sp, sr)]
                print(a.dataset, "fold", k, "g", g, "draw", d, "s*", sstar, {s: round(v, 3) for s, v in sc.items()}, f"{time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(recs); d.to_csv(OUT / f"{a.dataset}_{tag}.csv", index=False); pd.DataFrame(stops).to_csv(OUT / f"{a.dataset}_{tag}_stops.csv", index=False)
    print(d.groupby(["g", "method"]).cos_pair.mean().unstack().round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
