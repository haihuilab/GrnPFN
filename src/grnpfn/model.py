"""GRN-PFN: an in-context learner for perturbation-pair interactions, trained on gene-regulatory-network simulations.

Tokens form a grid populations × genes. Populations are the singles (Δ_p, input third), the context pairs (with their
observed interaction R_n) and the query pairs (no interaction). Token features at gene g: for a single, Δ_pg; for a
pair (A, B), Δ_Ag + Δ_Bg, Δ_Ag·Δ_Bg, |Δ_Ag − Δ_Bg| and (context only) R_ng; for all, the control level μ_g, the token
type and log cell numbers. Each layer applies (i) gene-axis induced attention within each population (Set Transformer
ISAB, m inducing points): every token learns its population's whole profile; (ii) population-axis attention within
each gene, keys restricted to singles and context pairs: a query pair reads the singles and the context pairs at the
same gene, with weights computed from whole-profile states; (iii) a feed-forward layer. A linear head reads R̂ at
query tokens. The model is equivariant to permutations of genes and of populations, symmetric in (A, B), has no gene
identities, and can output any direction (unlike coPFN, whose prediction is a weighted sum of context interactions).
Values are divided by the RMS single effect of the episode; μ stays on the log1p-CP10k scale."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NF = 11


class MHA(nn.Module):
    def __init__(self, d, h, ot_iters=0):
        super().__init__(); self.h = h; self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.o = nn.Linear(d, d)
        self.ot_iters = ot_iters                                                         # > 0: entropic-OT (Sinkhorn) coupling instead of softmax (P4-H3)

    def forward(self, x, y, mask=None, bias=None):
        """x (..., Lq, d) attends to y (..., Lk, d); mask (..., Lk) bool of valid keys; bias: additive logits
        broadcastable to (..., h, Lq, Lk) (membership relations, P4-H2)"""
        *b, Lq, d = x.shape; Lk = y.shape[-2]; h = self.h
        q = self.q(x).view(*b, Lq, h, d // h).transpose(-2, -3)
        k, v = self.kv(y).view(*b, Lk, 2, h, d // h).unbind(-3)
        k, v = k.transpose(-2, -3), v.transpose(-2, -3)
        if mask is None:
            am = None
        elif mask.dim() == x.dim():                                                         # (..., Lq, Lk) query×key mask (OT path)
            am = mask[..., None, :, :].expand(*b, h, Lq, Lk)
        else:
            am = mask[..., None, None, :].expand(*b, h, Lq, Lk)
        if bias is not None:
            am = torch.zeros(am.shape, dtype=q.dtype, device=q.device).masked_fill(~am, float("-inf")) + bias.to(q.dtype)
        if self.ot_iters > 0:
            o = sinkhorn_attention(q, k, v, am, self.ot_iters)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(o.transpose(-2, -3).reshape(*b, Lq, d))


def sinkhorn_attention(q, k, v, am, iters):
    """Entropic-OT coupling between queries and valid keys (log-domain Sinkhorn): rows sum to 1 (each query reads a
    full distribution, as in softmax attention) and every valid key receives the same total mass n_q/n_k, so context
    tokens are used evenly instead of a few hubs dominating. iters = 0 would be softmax attention."""
    S = (q @ k.transpose(-1, -2)).float() / np.sqrt(q.shape[-1]); NEG = -1e4               # finite "−inf": keeps gradients finite
    if am is None:
        valid = torch.ones_like(S, dtype=torch.bool)
    elif am.dtype == torch.bool:
        valid = am
    else:
        valid = torch.isfinite(am); S = S + torch.where(valid, am.float(), torch.zeros_like(S))
    S = S.masked_fill(~valid, NEG)
    nk = valid.any(-2, keepdim=True).sum(-1, keepdim=True).clamp(min=1).float()            # valid keys per (…, h)
    nq = valid.any(-1, keepdim=True).sum(-2, keepdim=True).clamp(min=1).float()            # real query rows
    S = (S - torch.logsumexp(S, -1, keepdim=True)).masked_fill(~valid, NEG)
    for _ in range(iters):
        S = (S - torch.logsumexp(S, -2, keepdim=True) + torch.log(nq / nk)).masked_fill(~valid, NEG)   # columns: n_q/n_k
        S = (S - torch.logsumexp(S, -1, keepdim=True)).masked_fill(~valid, NEG)                         # rows: 1
    P = torch.exp(S) * valid
    return (P.to(v.dtype) @ v)


class Layer(nn.Module):
    def __init__(self, d, h, m, ot_iters=0, pa=False):
        super().__init__()
        self.h = h
        if pa:                                                                            # P4-H6 partner-aligned attention (zero-gated)
            self.pa_q, self.pa_k = nn.Linear(d, d), nn.Linear(d, d); self.pa_a = nn.Parameter(torch.zeros(h))
        self.ind = nn.Parameter(torch.randn(m, d) / np.sqrt(d))
        self.g1, self.g2, self.p = MHA(d, h), MHA(d, h), MHA(d, h, ot_iters)
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, keymask, bias=None, popmask=None, part=None):
        """x (B, Npop, G, d); keymask (B, Npop) valid keys for population attention (singles + context pairs);
        bias (B, h, Npop, Npop) additive membership logits or None"""
        B, Np, G, d = x.shape
        z = self.n1(x)
        H = self.g1(self.ind.expand(B, Np, -1, -1), z)                                     # (B,Np,m,d) inducing summary per population
        x = x + self.g2(z, H)                                                             # genes read their population summary
        z = self.n2(x).transpose(1, 2)                                                    # (B,G,Np,d)
        km = keymask[:, None, :].expand(B, G, Np)
        if self.p.ot_iters > 0 and popmask is not None:                                     # OT coupling: only real populations are rows
            km = (popmask[:, :, None] & keymask[:, None, :])[:, None].expand(B, G, Np, Np)
        bb = None if bias is None else bias[:, None]
        if part is not None and hasattr(self, "pa_a") and int(part.max()) >= 0:
            pa = self.partner_logits(z, part); bb = pa if bb is None else bb + pa
        if self.p.ot_iters > 0 and torch.is_grad_enabled():                                 # OT: recompute in backward (memory)
            att = checkpoint(self.p, z, z, km, bb, use_reentrant=False)
        else:
            att = self.p(z, z, km, bb)
        x = x + att.transpose(1, 2)                                                       # each gene: populations attend to singles + context
        return x + self.ff(self.n3(x))

    def partner_logits(self, z, part):
        """z (B, G, Np, d) population states per gene; part (B, Np, Np, 2) single indices of the NON-shared partners for
        population pairs (t, s) sharing exactly one member, −1 elsewhere → additive logits (B, G, h, Np, Np):
        α_h ⟨W_q z_{p(t)}, W_k z_{p(s)}⟩ / √d_h — attention between related pairs compares the partners, not the pairs"""
        B, G, Np, d = z.shape; h = self.h; dh = d // h; Ps = int(part.max()) + 1
        zs = z[:, :, :Ps]
        qa = self.pa_q(zs).view(B, G, Ps, h, dh).transpose(2, 3); ka = self.pa_k(zs).view(B, G, Ps, h, dh).transpose(2, 3)
        S = (qa @ ka.transpose(-1, -2)) / np.sqrt(dh) * self.pa_a.view(1, 1, h, 1, 1)            # (B, G, h, Ps, Ps)
        valid = part[..., 0] >= 0
        idx = (part[..., 0].clamp(min=0).long() * Ps + part[..., 1].clamp(min=0).long()).view(B, 1, 1, Np * Np).expand(B, G, h, Np * Np)
        out = S.reshape(B, G, h, Ps * Ps).gather(-1, idx).view(B, G, h, Np, Np)
        return out * valid[:, None, None].to(out.dtype)


N_REL = 4                                                                                # 0 none, 1 pair→member, 2 pair↔pair sharing a member, 3 single→containing pair


class GRNPFN(nn.Module):
    def __init__(self, d=64, heads=4, layers=4, m=16, rel_bias=False, ot_iters=0, var_head=False, partner_attn=False):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([Layer(d, heads, m, ot_iters, partner_attn) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.heads = heads; self.partner_attn = partner_attn
        self.rel = nn.Parameter(torch.zeros(layers, N_REL, heads)) if rel_bias else None     # zero-init: starts as the plain model
        # P4-H5 posterior-predictive variance head: log σ² per query population and gene, read from DETACHED trunk features,
        # so the trunk and mean head are trained by the unchanged MSE objective (the mean stays the posterior-mean estimator)
        self.var = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)) if var_head else None
        if var_head:
            nn.init.zeros_(self.var[3].weight); nn.init.constant_(self.var[3].bias, -5.6)        # ≈ log of the typical per-gene squared error

    def forward(self, feats, keymask, qsel, rel=None, return_var=False, part=None):
        """feats (B, Npop, G, NF); keymask (B, Npop); qsel (B, Npop) query tokens; rel (B, Npop, Npop) int relation codes
        (used when the model has membership biases) → R̂ (B, Npop, G) (valid at qsel)"""
        x = self.embed(feats); popmask = keymask | qsel                                      # real (non-padding) populations
        for i, L in enumerate(self.layers):
            bias = None if (self.rel is None or rel is None) else self.rel[i][rel.long()].permute(0, 3, 1, 2)   # (B, h, Np, Np)
            x = L(x, keymask, bias, popmask, part)
        if return_var:
            return self.out(x)[..., 0], self.var(x.detach())[..., 0]                    # mean, log σ² (scaled units)
        return self.out(x)[..., 0]


class LoopedGRNPFN(nn.Module):
    """Weight-tied variant: one Layer applied T times. Before every pass the state is mixed with the input embedding,
    u_t = W[x_t, e] (input injection without drift of the residual stream), x_{t+1} = Layer(u_t). Training draws T per
    step; at test time T is free (more passes = more refinement). Each pass is gradient-checkpointed."""
    def __init__(self, d=64, heads=4, m=16, loops=12):
        super().__init__()
        self.loops = loops
        self.embed = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
        self.mix = nn.Linear(2 * d, d); self.block = Layer(d, heads, m)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def _step(self, x, e, keymask):
        return self.block(self.mix(torch.cat([x, e], -1)), keymask)

    def forward(self, feats, keymask, qsel, loops=None):
        e = self.embed(feats); x = e
        for _ in range(loops or self.loops):
            x = checkpoint(self._step, x, e, keymask, use_reentrant=False) if self.training else self._step(x, e, keymask)
        return self.out(x)[..., 0]


def build(D, mu, pairs_ctx, Y, pairs_q, n_single, n_pair_ctx, n_pair_q, n_ctrl):
    """numpy episode → (feats (Np, G, NF), keymask (Np,), qsel (Np,), scale). Populations ordered singles, context, queries."""
    s = float(np.sqrt(np.mean(D.astype(np.float64) ** 2))) + 1e-8
    D = D / s; Y = Y / s; P, G = D.shape
    ls = np.log(np.asarray(n_single, float)) - np.log(300.0)
    def pair_feats(pr, R, npair, flag):
        a, b = D[pr[:, 0]], D[pr[:, 1]]
        f = np.zeros((len(pr), G, NF), np.float32)
        f[..., 0] = a + b; f[..., 1] = a * b; f[..., 2] = np.abs(a - b); f[..., 3] = 0 if R is None else R
        f[..., 5 + flag] = 1
        f[..., 8] = (np.log(np.asarray(npair, float)) - np.log(300.0))[:, None]
        f[..., 9] = np.minimum(ls[pr[:, 0]], ls[pr[:, 1]])[:, None]; f[..., 10] = np.maximum(ls[pr[:, 0]], ls[pr[:, 1]])[:, None]
        return f
    fs = np.zeros((P, G, NF), np.float32); fs[..., 0] = D; fs[..., 5] = 1; fs[..., 8] = ls[:, None]
    feats = np.concatenate([fs, pair_feats(pairs_ctx, Y, n_pair_ctx, 1), pair_feats(pairs_q, None, n_pair_q, 2)])
    feats[..., 4] = mu[None, :]
    feats[..., 9:] = np.where(np.isfinite(feats[..., 9:]), feats[..., 9:], 0)
    Np = len(feats); keymask = np.zeros(Np, bool); keymask[: P + len(pairs_ctx)] = True
    qsel = np.zeros(Np, bool); qsel[P + len(pairs_ctx):] = True
    feats[..., 8] = feats[..., 8] + 0 * np.log(n_ctrl)
    return feats, keymask, qsel, s


def relations(n_single, pairs_ctx, pairs_q):
    """(Np, Np) int8 membership relation codes for populations ordered singles, context pairs, query pairs"""
    P = n_single; pr = np.vstack([np.asarray(pairs_ctx, int).reshape(-1, 2), np.asarray(pairs_q, int).reshape(-1, 2)]); M = len(pr); Np = P + M
    R = np.zeros((Np, Np), np.int8); idx = np.arange(M)
    R[P + idx, pr[:, 0]] = 1; R[P + idx, pr[:, 1]] = 1                                     # pair → its members
    R[pr[:, 0], P + idx] = 3; R[pr[:, 1], P + idx] = 3                                     # single → pairs containing it
    share = (pr[:, None, 0] == pr[None, :, 0]) | (pr[:, None, 0] == pr[None, :, 1]) | (pr[:, None, 1] == pr[None, :, 0]) | (pr[:, None, 1] == pr[None, :, 1])
    np.fill_diagonal(share, False); R[P:, P:][share] = 2
    return R


def partners(n_single, pairs_ctx, pairs_q):
    """(Np, Np, 2) int16: for population pairs (t, s) — both pairs, sharing exactly one member — the single indices of
    t's and s's non-shared partners; −1 elsewhere (P4-H6)"""
    P = n_single; pr = np.vstack([np.asarray(pairs_ctx, int).reshape(-1, 2), np.asarray(pairs_q, int).reshape(-1, 2)]); M = len(pr); Np = P + M
    out = -np.ones((Np, Np, 2), np.int16)
    for i in range(2):
        for j in range(2):                                                                # member i of t equals member j of s
            eq = pr[:, None, i] == pr[None, :, j]; other_t, other_s = pr[:, 1 - i][:, None], pr[:, 1 - j][None, :]
            ok = eq & (other_t != other_s)                                                # exactly one shared member
            T, S_ = np.nonzero(ok); out[P + T, P + S_, 0] = other_t[T, 0]; out[P + T, P + S_, 1] = other_s[0, S_]
    return out


def collate(items):
    Np = max(f.shape[0] for f, *_ in items); G = items[0][0].shape[1]; B = len(items)
    feats = torch.zeros(B, Np, G, NF); km = torch.zeros(B, Np, dtype=torch.bool); qs = torch.zeros(B, Np, dtype=torch.bool)
    tgt = torch.zeros(B, Np, G); w = torch.zeros(B)
    for i, (f, k, q, s, t, ww, *_) in enumerate(items):
        n = f.shape[0]; feats[i, :n] = torch.from_numpy(f); km[i, :n] = torch.from_numpy(k); qs[i, :n] = torch.from_numpy(q)
        if t is not None:
            tgt[i, n - t.shape[0]: n] = torch.from_numpy(t)
        w[i] = ww
    out = dict(feats=feats, keymask=km, qsel=qs, target=tgt, w=w)
    if all(len(it) > 6 for it in items):                                                   # membership relations attached
        rel = torch.zeros(B, Np, Np, dtype=torch.int8)
        for i, it in enumerate(items):
            n = it[0].shape[0]; rel[i, :n, :n] = torch.from_numpy(it[6])
        out["rel"] = rel
    if all(len(it) > 7 for it in items):                                                   # partner indices attached (P4-H6)
        part = -torch.ones(B, Np, Np, 2, dtype=torch.int16)
        for i, it in enumerate(items):
            n = it[0].shape[0]; part[i, :n, :n] = torch.from_numpy(it[7])
        out["part"] = part
    return out


def predict(model, D, mu, pairs_ctx, Y, pairs_q, n_single, n_pair_ctx, n_pair_q, n_ctrl, device="cpu", loops=None):
    f, k, q, s = build(D, mu, pairs_ctx, Y, pairs_q, n_single, n_pair_ctx, n_pair_q, n_ctrl)
    use_rel = getattr(model, "rel", None) is not None; use_pa = getattr(model, "partner_attn", False)
    b = collate([(f, k, q, s, None, 1.0) + ((relations(len(D), pairs_ctx, pairs_q),) if use_rel else ()) + ((partners(len(D), pairs_ctx, pairs_q),) if use_pa else ())])
    with torch.no_grad(), torch.autocast(device_type="cuda" if device != "cpu" else "cpu", dtype=torch.bfloat16, enabled=device != "cpu"):
        kw = dict(loops=loops) if isinstance(model, LoopedGRNPFN) else {}
        if use_rel:
            kw["rel"] = b["rel"].to(device)
        if use_pa:
            kw["part"] = b["part"].to(device)
        out = model(b["feats"].to(device), b["keymask"].to(device), b["qsel"].to(device), **kw)
    return out[0, q].float().cpu().numpy() * s


def load_grnpfn(path, device="cpu"):
    ck = torch.load(path, map_location=device); cfg = dict(ck["cfg"]); looped = cfg.pop("looped", False)
    m = (LoopedGRNPFN if looped else GRNPFN)(**cfg); m.load_state_dict(ck["state_dict"]); return m.to(device).eval()
