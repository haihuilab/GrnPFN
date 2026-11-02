"""Gene-regulatory-network simulator of combinatorial perturbation screens (prior of GRN-PFN; synthetic only).

A screen is a random network of hidden pathway nodes (unmeasured; drug targets), transcription factors and target
genes, evaluated in topological order with saturating Hill regulation:
    x_j = b_j · ε_ij · (base_j + Σ_r a_jr · H_jr(x_r)),   H = x^n/(K^n + x^n) (activation) or K^n/(K^n + x^n) (repression),
with K_jr at the regulator's control median (regulators operate near their midpoint), per-cell intrinsic noise ε_ij
(log-normal) and per-cell state factors s_i that modulate root nodes. A perturbation multiplies its node during the
pass (CRISPRa: × (1 + (f − 1)u_i); knockdown or inhibitor: × (1 − e·u_i)), with per-cell strength u_i and penetrance,
so effects propagate and combine through the network's nonlinearity. Pairs apply both. Measured genes (TFs and targets)
are sampled as Poisson counts with per-cell library size and normalized as the real data (log1p of counts per 10⁴).

Populations and cell numbers follow the real-screen templates (prior v3); cells of singles and control are split into
input / train-target / test-target thirds as in the benchmark. Genes kept: the G_keep measured genes with the largest
max_p |Δ_p,g| on the input third (the rule applied to the real screens)."""
from __future__ import annotations

import numpy as np

from .prior import PriorConfig


def _draw_counts(rng, cfg, P, M):
    tp = cfg.templates[rng.integers(len(cfg.templates))]; jl = np.log(cfg.template_jitter); js = np.exp(rng.uniform(-jl, jl)); tl = np.log(cfg.template_tail)
    knots = lambda q: np.r_[np.log(q[0]) - tl, np.log(q), np.log(q[-1]) + tl]
    draw = lambda q, n: np.exp(np.interp(rng.random(n), [0, .05, .25, .5, .75, .95, 1], knots(q)))
    n_s = np.maximum(30, draw(tp["single"], P) * js).astype(int); n_p = np.maximum(20, draw(tp["pair"], M) * js).astype(int)
    n_c = int(max(450, tp["ctrl"] * np.exp(rng.uniform(-jl, jl))))
    return n_s, n_p, n_c


class Network:
    """node j: x_j = b_j · ε_ij · (base_j + V_j · σ(Σ_r w_jr · log(x_r / K_r) + c_j)); σ logistic (saturating integration,
    multiplicative in the regulators), K_r at the regulator's control median; regulators chosen with preferential
    attachment (hub TFs)"""
    def __init__(self, rng, n_hidden, n_tf, n_target, n_state=2, n_prog=None):
        """layers in topological order: hidden pathway nodes (drug targets) → perturbable TFs → program nodes (few,
        saturating, convergent) → target genes; measured genes are TFs, programs and targets"""
        self.rng = rng
        n_prog = int(rng.integers(1, 7)) if n_prog is None else n_prog
        self.nh, self.ntf, self.np_, self.ntg = n_hidden, n_tf, n_prog, n_target - n_prog
        H = np.arange(n_hidden); T = n_hidden + np.arange(n_tf); Pg = n_hidden + n_tf + np.arange(n_prog)
        self.N = n_hidden + n_tf + n_target; Tg = np.arange(Pg[-1] + 1, self.N)
        self.reg, self.w = [], []
        pick = lambda pool, k: rng.choice(pool, min(k, len(pool)), replace=False) if k and len(pool) else np.array([], int)
        prog_in = rng.uniform(0.3, 0.9)                                                      # fraction of TFs feeding each program (convergence)
        pw = rng.uniform(2.0, 6.0)                                                            # program input strength (saturation)
        glob = rng.random() < 0.85; gsign = 1 if rng.random() < 0.5 else -1                   # global program (growth / differentiation axis)
        p_glob = rng.uniform(0.3, 0.8)
        for j in range(self.N):
            if j in H:
                r = pick(H[H < j], int(rng.integers(0, 2))); w = np.abs(rng.normal(1.5, 0.5, len(r)))
            elif j in T:
                r = np.r_[pick(H, int(rng.integers(0, 3))), pick(T[T < j], int(rng.integers(0, 2)))]
                w = np.where(rng.random(len(r)) < 0.7, 1, -1) * np.abs(rng.normal(1.5, 0.6, len(r)))
            elif j == Pg[0] and glob:
                r = np.r_[T, pick(H, int(rng.integers(0, 3)))]
                w = gsign * np.abs(rng.normal(1.0, 0.3, len(r))) * pw / np.sqrt(len(r))
            elif j in Pg:
                k = max(2, int(prog_in * n_tf * rng.uniform(0.7, 1.0)))
                r = np.r_[pick(T, k), pick(H, int(rng.integers(0, 3)))]
                w = np.where(rng.random(len(r)) < 0.75, 1, -1) * np.abs(rng.normal(1.0, 0.4, len(r))) * pw / np.sqrt(max(len(r), 1))
            else:
                r = np.r_[pick(Pg, int(rng.integers(1, 4))), pick(T, int(rng.integers(0, 3)))]
                if glob and Pg[0] not in r and rng.random() < p_glob:
                    r = np.r_[Pg[0], r]
                w = np.where(rng.random(len(r)) < 0.65, 1, -1) * np.abs(rng.normal(2.5, 0.8, len(r))) / np.sqrt(max(len(r), 1))
            self.reg.append(np.asarray(r, int)); self.w.append(w)
        self.steep = float(np.exp(rng.uniform(np.log(4.0), np.log(30.0))))                          # regulatory steepness (episode level)
        for j in np.r_[Pg, Tg]:
            self.w[j] = self.w[j] * self.steep
        outdeg = np.zeros(self.N)
        for r in self.reg:
            outdeg[r] += 1
        self.c = rng.normal(0, 1.0, self.N); self.V = np.exp(rng.normal(0.5, 0.5, self.N)); self.base = rng.uniform(0.02, 0.3, self.N)
        self.b = np.exp(rng.normal(0, 0.5, self.N))
        self.load = rng.normal(0, 0.2, (self.N, n_state)) * (rng.random((self.N, n_state)) < 0.25)
        self.noise_sd = rng.uniform(0.05, 0.25); self.n_state = n_state
        self.K = np.ones(self.N); self.outdeg = outdeg
        self.K = np.median(self.run(400, {}), 0)                                         # pilot pass: K at control medians

    def run(self, n, pert):
        """pert: {node: per-cell multiplier array (n,)} → node values (n, N)"""
        rng = self.rng; x = np.zeros((n, self.N)); lx = np.zeros((n, self.N))
        s = rng.normal(size=(n, self.n_state))
        eps = np.exp(rng.normal(0, self.noise_sd, (n, self.N)) + s @ self.load.T)
        for j in range(self.N):
            r = self.reg[j]
            z = self.c[j] + (lx[:, r] @ self.w[j] if len(r) else 0.0)
            v = self.b[j] * eps[:, j] * (self.base[j] + self.V[j] / (1 + np.exp(-z)))
            if j in pert:
                v = v * pert[j]
            x[:, j] = v; lx[:, j] = np.log(np.maximum(v, 1e-9) / self.K[j])
        return x


def sample_grn_episode(rng: np.random.Generator, cfg: PriorConfig | None = None, G_keep=256, kind=None, max_cells=90_000, truth_cells=0, depth=1.0,
                       design="random"):
    """design: "random" (sparse random pairs for genetic screens, anchor design for drugs — the original prior) or
    "dense" (P4-U1: all-by-all combinatorial design among 10–30 genetic perturbations, 60–100% of all pairs, ≤ 300)"""
    cfg = cfg or PriorConfig(version=3)
    if design == "dense":
        kind = kind or rng.choice(["crispra", "knockdown"], p=[0.57, 0.43])
    kind = kind or rng.choice(["crispra", "knockdown", "drug"], p=[0.4, 0.3, 0.3])
    n_hidden = int(rng.integers(4, 16)); n_tf = int(rng.integers(40, 100)); n_target = int(rng.integers(G_keep + 64, G_keep + 192) - n_tf)
    net = Network(rng, n_hidden, n_tf, n_target)
    meas = np.arange(n_hidden, net.N)
    # perturbations: (node, mode, strength) — genes are TFs (and a few targets); drugs are hidden nodes at dose levels
    if kind == "drug":
        n_comp = int(rng.integers(5, 23)); doses = int(rng.integers(2, 5))
        wh = (net.outdeg[:n_hidden] + 1.0); comp_nodes = rng.choice(n_hidden, n_comp, replace=True, p=wh / wh.sum())
        comp_eff = rng.uniform(0.3, 0.98, n_comp); comp_ic = rng.uniform(0.3, 3.0, n_comp); comp_act = rng.random(n_comp) < 0.15
        lv = np.logspace(-1, 1, doses)
        singles = [(comp_nodes[c], "act" if comp_act[c] else "inh", comp_eff[c] * l / (l + comp_ic[c]) * (5 if comp_act[c] else 1)) for c in range(n_comp) for l in lv]
        hub_node = int(np.argmax(net.outdeg[:n_hidden])); hub_e = rng.uniform(0.5, 0.98); hub_ic = rng.uniform(0.3, 3.0)
        hubs = [(hub_node, "inh", hub_e * l / (l + hub_ic)) for l in lv]
        P_list = hubs + singles
        pairs = np.array([(doses + c * doses + li, li) for c in range(n_comp) for li in range(doses)])  # compound@dose + hub@same dose
    else:
        P = int(rng.integers(10, 31)) if design == "dense" else int(rng.integers(20, min(90, n_tf)))
        tfs = np.arange(n_hidden, n_hidden + n_tf); wsel = (net.outdeg[tfs] + 1.0) ** rng.uniform(0.5, 1.5)
        nodes = rng.choice(tfs, P, replace=False, p=wsel / wsel.sum())                # screens target influential regulators
        mode = "act" if kind == "crispra" else "inh"
        st = np.exp(rng.uniform(np.log(3), np.log(50), len(nodes))) if kind == "crispra" else rng.uniform(0.4, 0.95, len(nodes))
        P_list = [(n, mode, s) for n, s in zip(nodes, st)]
        allp = np.array([(i, j) for i in range(len(P_list)) for j in range(i + 1, len(P_list))])
        if design == "dense":
            pairs = allp[rng.permutation(len(allp))[: min(300, int(round(len(allp) * rng.uniform(0.6, 1.0))))]]
        else:
            pairs = allp[rng.permutation(len(allp))[: int(rng.integers(30, 151))]]
    P, M = len(P_list), len(pairs)
    n_s, n_p, n_c = _draw_counts(rng, cfg, P, M)
    if depth != 1.0:                                                                    # validation only: deeper screens
        n_s, n_p, n_c = (n_s * depth).astype(int), (n_p * depth).astype(int), int(n_c * depth)
    tot = n_s.sum() + n_p.sum() + n_c
    if tot > max_cells:
        f = max_cells / tot; n_s = np.maximum(30, (n_s * f).astype(int)); n_p = np.maximum(20, (n_p * f).astype(int)); n_c = max(450, int(n_c * f))
    pen = rng.uniform(0.6, 1.0); u_ab = rng.uniform(1.5, 6.0)

    # effector sharing: in double-perturbed cells each perturbation is weaker (limited dCas9 activator / Cas13 / uptake)
    dil = rng.uniform(0.4, 1.0) if kind != "drug" else rng.uniform(0.7, 1.0)

    def mult(spec, n, d=1.0):
        node, mode, s = spec
        u = d * rng.beta(u_ab, 1.5, n) * (rng.random(n) < pen)
        return (1 + (s - 1) * u) if mode == "act" else (1 - s * u)

    gmean = np.exp(rng.normal(np.log(rng.uniform(0.15, 1.5)), 1.2, len(meas)))                         # control expected counts per gene at 5,000 UMI
    xc_pilot = net.run(2000, {})[:, meas].mean(0)
    scale = gmean / np.maximum(xc_pilot, 1e-9)
    lib_mu = np.log(rng.uniform(2000, 12000))

    def cells(n, specs):
        pert = {}
        for sp in specs:
            m = mult(sp, n, dil if len(specs) > 1 else 1.0); pert[sp[0]] = pert.get(sp[0], 1.0) * m
        x = net.run(n, pert)[:, meas]
        lib = np.exp(rng.normal(lib_mu, 0.4, n))
        cnt = rng.poisson(x * scale[None] * (lib / 5000.0)[:, None])
        return np.log1p(cnt * (1e4 / lib)[:, None]).astype(np.float32)

    thirds = lambda Z: np.array_split(Z[rng.permutation(len(Z))], 3)
    ctrl = thirds(cells(n_c, []))
    S = [thirds(cells(n, [P_list[p]])) for p, n in enumerate(n_s)]
    m = lambda Z: Z.mean(0)
    PAc = [cells(n, [P_list[a], P_list[b]]) for (a, b), n in zip(pairs, n_p)]
    PA = [m(Z) for Z in PAc]
    PH = [(m(Z[: len(Z) // 2]), m(Z[len(Z) // 2:])) for Z in PAc]
    D = np.stack([m(S[p][0]) - m(ctrl[0]) for p in range(P)])
    keep = np.sort(np.argsort(-np.abs(D).max(0))[:G_keep])
    R = {}
    for r, role in ((1, "tr"), (2, "te")):
        R[role] = np.stack([PA[i] - m(S[a][r]) - m(S[b][r]) + m(ctrl[r]) for i, (a, b) in enumerate(pairs)])[:, keep]
    # independent half-split interactions (pair halves; singles and control: train vs test third) for the prior check
    Rh = [np.stack([PH[i][h] - m(S[a][1 + h]) - m(S[b][1 + h]) + m(ctrl[1 + h]) for i, (a, b) in enumerate(pairs)])[:, keep] for h in (0, 1)]
    D1 = np.stack([m(S[p][1]) - m(ctrl[1]) for p in range(P)])[:, keep]
    vn = lambda Z: Z[:, keep].var(0) / max(len(Z), 1)                                   # sampling variance of a mean
    noise_tr = np.array([float(np.mean(vn(PAc[i]) + vn(S[a][1]) + vn(S[b][1]) + vn(ctrl[1]))) for i, (a, b) in enumerate(pairs)])
    extra = {}
    if truth_cells:                                                                     # per-gene sampling variance of the two halves
        extra["noise_h_g"] = np.stack([np.stack([vn(PAc[i][: len(PAc[i]) // 2] if h == 0 else PAc[i][len(PAc[i]) // 2:]) + vn(S[a][1 + h]) + vn(S[b][1 + h]) + vn(ctrl[1 + h])
                                                 for i, (a, b) in enumerate(pairs)]) for h in (0, 1)]).astype(np.float32)
    if truth_cells:                                                                     # near noise-free interactions (validation only)
        mt = lambda specs: cells(truth_cells, specs)[:, keep].mean(0); c0 = mt([]); sm = [mt([P_list[p]]) for p in range(P)]
        extra["R_true"] = np.stack([mt([P_list[a], P_list[b]]) - sm[a] - sm[b] + c0 for a, b in pairs]).astype(np.float32)
    return dict(**extra, D=D[:, keep].astype(np.float32), R_h1=Rh[0].astype(np.float32), R_h2=Rh[1].astype(np.float32), D1=D1.astype(np.float32), mu_ctrl=m(ctrl[0])[keep].astype(np.float32), pairs=pairs, R_tr=R["tr"].astype(np.float32),
                R_te=R["te"].astype(np.float32), n_single=n_s.astype(float), n_pair=n_p.astype(float), n_ctrl=float(n_c), kind=kind, noise_tr=noise_tr.astype(np.float32),
                hubs=np.arange(len(hubs)) if kind == "drug" else np.array([], int))
