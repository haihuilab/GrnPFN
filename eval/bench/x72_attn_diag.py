"""Cycle-2 diagnosis, part 4: is the PFN's population attention coherent across genes?
(a) Coherence: for query pairs, per layer/head, split the variance of the attention logits over keys into the
    gene-averaged part (between keys) and the gene-specific deviations; report the share that is gene-specific, and the
    mean total-variation distance between each gene's attention distribution and the gene-averaged distribution.
(b) Intervention (no retraining): population-attention logits replaced by their gene average (one attention matrix per
    head shared by all genes), membership biases unchanged; accuracy at g = .15 and 1 vs the unmodified model.
Dev 301, grnpfn_rel.   python x72_attn_diag.py --dataset norman"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import RESULTS
from grnpfn import model as M
from grnpfn.model import load_grnpfn, predict
from grnpfn_real import real_folds
from phase3 import score

STATS = []; MODE = {"pool": False, "stats": False}


def mha_forward(self, x, y, mask=None, bias=None):
    *b, Lq, d = x.shape; Lk = y.shape[-2]; h = self.h; dh = d // h
    q = self.q(x).view(*b, Lq, h, dh).transpose(-2, -3)
    k, v = self.kv(y).view(*b, Lk, 2, h, dh).unbind(-3); k, v = k.transpose(-2, -3), v.transpose(-2, -3)
    pop = x.dim() == 4 and x.shape[-2] == y.shape[-2] and getattr(self, "_is_pop", False)
    if not pop or not (MODE["pool"] or MODE["stats"]):
        am = None if mask is None else mask[..., None, None, :].expand(*b, h, Lq, Lk)
        if bias is not None:
            am = torch.zeros(am.shape, dtype=q.dtype, device=q.device).masked_fill(~am, float("-inf")) + bias.to(q.dtype)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(o.transpose(-2, -3).reshape(*b, Lq, d))
    # population attention: x (B, G, Np, d); q,k,v (B, G, h, Np, dh); mask (B, G, Np); bias (B, 1, h, Np, Np)
    qf, kf = q.float(), k.float(); km = mask[:, 0]                                          # (B, Np) valid keys
    bb = 0 if bias is None else bias.float()[:, 0]                                         # (B, h, Np, Np)
    Lbar = torch.einsum("bghqd,bghkd->bhqk", qf, kf) / (qf.shape[1] * dh ** 0.5) + bb      # gene-averaged logits
    if MODE["stats"]:
        qs = getattr(self, "_qsel")                                                       # (Np,) bool query rows
        L = torch.einsum("bghqd,bghkd->bghqk", qf[:, :, :, qs], kf) / dh ** 0.5 + (bb[:, None, :, qs] if torch.is_tensor(bb) else 0)
        L = L.masked_fill(~km[:, None, None, None, :], float("nan"))
        Lm = torch.nanmean(L, 1, keepdim=True)
        between = torch.nanmean((Lm - torch.nanmean(Lm, -1, keepdim=True)) ** 2, -1).mean().item()
        within = torch.nanmean((L - Lm) ** 2, -1).mean().item()
        A = torch.softmax(L.nan_to_num(float("-inf")), -1); Abar = torch.softmax(Lm[:, 0].nan_to_num(float("-inf")), -1)[:, None]
        tv = 0.5 * (A - Abar).abs().sum(-1).mean().item()
        STATS.append(dict(layer=self._layer, between=between, within=within, gene_share=within / (within + between), tv=tv))
    if MODE["pool"]:
        Lb = Lbar.masked_fill(~km[:, None, None, :], float("-inf")); A = torch.softmax(Lb, -1).to(v.dtype)   # (B, h, Np, Np)
        o = torch.einsum("bhqk,bghkd->bghqd", A, v)
    else:
        am = mask[..., None, None, :].expand(*b, h, Lq, Lk)
        if bias is not None:
            am = torch.zeros(am.shape, dtype=q.dtype, device=q.device).masked_fill(~am, float("-inf")) + bias.to(q.dtype)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
    return self.o(o.transpose(-2, -3).reshape(*b, Lq, d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--model", default="grnpfn_rel"); a = ap.parse_args()
    M.MHA.forward = mha_forward; dev = "cuda"; net = load_grnpfn(str(RESULTS / "grnpfn" / f"{a.model}.pt"), dev)
    for i, L in enumerate(net.layers):
        L.p._is_pop = True; L.p._layer = i
    recs = []
    for fd in real_folds(a.dataset, "pair", 301, 0):
        k = fd["k"]; pool, te = fd["tr"], fd["te"]; ix = {g: i for i, g in enumerate(fd["singles"])}
        pidx = lambda ps: np.array([(ix[x], ix[y]) for x, y in ps]); pool_ix, te_ix = pidx(pool), pidx(te)
        npl, nte = np.asarray(fd["n_pair_c"], float), np.asarray(fd["n_pair_q"], float)
        for g in (0.15, 1.0):
            sel = np.arange(len(pool)) if g == 1.0 else np.sort(np.random.default_rng([k, 0, int(100 * g)]).choice(len(pool), max(10, int(round(g * len(pool)))), replace=False))
            Y = fd["R"]["train"][sel]; nq = len(fd["singles"]) + len(sel)
            qs = torch.zeros(nq + len(te), dtype=torch.bool, device=dev); qs[nq:] = True
            for L in net.layers:
                L.p._qsel = qs
            for mode in ("softmax", "gene-pooled"):
                MODE["pool"] = mode == "gene-pooled"; MODE["stats"] = mode == "softmax" and k == 0
                P = predict(net, fd["D"], fd["mu"], pool_ix[sel], Y, te_ix, fd["n_single"], npl[sel], nte, fd["n_ctrl"], device=dev)
                sp, _ = score(P, fd["R_q_test"], Y.mean(0))
                recs += [dict(dataset=a.dataset, fold=k, g=g, method=mode, AB=fd["key"](*t), cos_pair=x) for t, x in zip(te, sp)]
                if MODE["stats"]:
                    for s in STATS[-len(net.layers):]:
                        s.update(dataset=a.dataset, g=g)
        print(a.dataset, k, flush=True)
    out = RESULTS / "x71_diag"; d = pd.DataFrame(recs); d.to_csv(out / f"attn_pool_{a.dataset}.csv", index=False)
    st = pd.DataFrame(STATS); st.to_csv(out / f"attn_stats_{a.dataset}.csv", index=False)
    print(st.round(3).to_string(index=False)); print(d.groupby(["g", "method"]).cos_pair.mean().unstack().round(4).to_string())


if __name__ == "__main__":
    main()
