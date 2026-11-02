"""Prior of the GrnPFN simulator.

1. ``PriorConfig``: screen templates (cell numbers, per-cell variance) read from eval/results/bench/copfn/prior_stats.json,
   used by ``sim.sample_grn_episode`` (formerly the ``copfn`` package; its episode sampler was removed on 2026-10-28).
2. Hybrid-prior components added to simulated screens during training (formerly ``prior2.py``): ``screen_specific``
   (the hybrid prior of the manuscript), ``global_epistasis``, ``member_signature`` and ``bilinear_response``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STATS = Path(__file__).resolve().parents[2] / "eval" / "results" / "bench" / "copfn" / "prior_stats.json"


@dataclass
class PriorConfig:
    d: int = 50
    P_range: tuple[int, int] = (20, 120)
    pairs_range: tuple[int, int] = (40, 300)
    n_query: int = 32
    rho_log_range: tuple[float, float] = (np.log(0.02), np.log(0.35))   # ‖R‖²/‖Δ_A+Δ_B‖² (empirical 10–90%: 0.02–0.34)
    uniform_frac_range: tuple[float, float] = (0.15, 0.7)      # fraction of interaction power in the uniform term γ(Δ_A+Δ_B)
    share_H_range: tuple[float, float] = (0.05, 0.6)           # coupling share of the non-uniform power
    share_U_range: tuple[float, float] = (0.0, 0.3)            # gene-modulation share of the non-uniform power
    gamma_range: tuple[float, float] = (-0.5, -0.02)           # implied range of γ = −sqrt(uniform_frac · ρ)
    rank_H: tuple[int, int] = (2, 12)
    rank_G: tuple[int, int] = (1, 6)
    frac_modulated: tuple[float, float] = (0.0, 0.8)           # fraction of perturbations with a U_p
    cells_single_log: tuple[float, float] = (np.log(40), np.log(1200))
    cells_pair_log: tuple[float, float] = (np.log(30), np.log(900))
    cells_ctrl_log: tuple[float, float] = (np.log(400), np.log(12000))
    cell_var_range: tuple[float, float] = (1.0, 20.0)          # per-cell latent variance / mean squared effect norm (empirical 3–16)
    homoscedastic: bool = False                                # Stage 1: equal cell numbers, Gaussian, no misspec
    version: int = 1                                           # 2: changes below, from the prior predictive check (copfn_ppc.py);
                                                               # 3: v2 with the overrides in __post_init__ (copfn_ppc_structure.py)
    # version 2 only:
    template_jitter: float = 2.5                               # cell numbers and per-cell variance from a real screen, × e^U(±log 2.5)
    template_tail: float = 1.5                                 # cell-number range extended beyond the real 5–95% quantiles by this factor
    inert_frac_range: tuple[float, float] = (0.0, 0.35)        # fraction of perturbations with no effect (R = 0 for their pairs)
    inert_zero_prob: float = 0.35                              # probability that a screen has no inert perturbation at all
    pop_var_sd: float = 0.5                                    # log-sd of per-population per-cell variance around the screen's value
    null_screen_prob: float = 0.08                             # whole screens without any effect
    hubs_range: tuple[int, int] = (1, 5)                       # star graphs: 1–4 hubs
    hub_dominance_log: tuple[float, float] = (0.0, float(np.log(40.0)))   # ‖Δ_hub‖² / median ‖Δ_spoke‖²: log-uniform 1–40
    norm_sd_range: tuple[float, float] = (0.3, 0.8)            # log-normal spread of effect norms
    noise_scale_log: tuple[float, float] = (0.0, 0.0)          # per-cell variance × e^U(lo, hi) (version 3: below 1)
    cells_from_quantiles: bool = False                         # cell numbers by inverse CDF through the template's quantiles
    misspec: float = 0.0                                       # fraction of misspecified episodes
    leaky: bool = False
    stats_path: Path = STATS
    spectra: list[np.ndarray] = field(default_factory=list)
    templates: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.version >= 3:                                       # version 3 (2026-10-20): structural + joint-bias PPC
            self.uniform_frac_range, self.rank_H = (0.3, 0.95), (1, 6)
            self.noise_scale_log = (float(np.log(0.6)), 0.0)
            self.inert_frac_range, self.inert_zero_prob = (0.0, 0.2), 0.6
            self.cells_from_quantiles = True
        st = json.load(open(self.stats_path))
        if not self.templates:
            for v in st.values():
                self.templates.append(dict(single=v["cells_single"], pair=v["cells_pair"], ctrl=v["cells_ctrl"],
                                           var=v["cell_var_over_scale"]["single"]))
        if not self.spectra:
            for v in st.values():                                   # pad screens with fewer perturbations than d
                sp = np.asarray(v["spectrum"], float)[: self.d]
                if len(sp) < self.d:
                    sp = np.concatenate([sp, np.full(self.d - len(sp), max(sp.min(), 1e-6))])
                self.spectra.append(sp / sp.sum())


# ── hybrid-prior components ────────────────────────────────────────────────────
# Hybrid prior for in-context learning from measured pairs (manuscript/notes/autonomous_log.md, P4-H1).
# 
# Diagnosis: in the GRN simulator, interactions are almost fully determined by the single profiles through shared network
# mechanics (an empty-context GRN-PFN beats kernel ridge with all pairs on simulated screens), so pretraining never rewards
# learning from measured pairs; on real screens kernel ridge's accuracy comes almost entirely from them. This prior keeps
# the mechanistic interaction of each simulated screen and adds a SCREEN-SPECIFIC component, drawn fresh per screen, that
# is identifiable only from the screen's own measured pairs:
#   R_AB = R_GRN,AB + ρ·(f(z_A, z_B) P + (m_A ⊙ m_B) Q) + noise (the simulator's),
#   z_p = normalised random low-rank projection of the observed single profile Δ_p (rank r ~ U{2..12});
#   f = random symmetric 2-layer network of (z_A + z_B, z_A ⊙ z_B, |z_A − z_B|) → K coefficients;
#   m_p ~ N(0, I_K) iid per perturbation (member-specific effects: learnable only from pairs sharing a member);
#   P, Q = K output programs mixing the singles' own gene directions and random sparse gene programs;
#   ρ = ratio of the specific to the mechanistic power, log-uniform in [0.3, 3]; with probability p0 ρ = 0.
# The component is added identically to R_tr and R_te (it is signal; the sampling noise is the simulator's).


def screen_specific(ep, rng, p0=0.25, p_low=0.0):
    """p_low (P4-U2): probability that the coefficient function is low-order (linear + bilinear in the projected singles)
    instead of the random two-layer network; 0 = the P4-H1 prior exactly (same random draws)"""
    D = np.asarray(ep["D"], float); P_, G = D.shape; pairs = ep["pairs"]; R = np.asarray(ep["R_tr"], float)
    if rng.random() < p0:
        return np.zeros_like(R)
    r = int(rng.integers(2, 13)); K = int(rng.integers(2, 9)); h = 32
    Z = D @ rng.normal(size=(G, r)); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8) * np.sqrt(r)
    a, b = Z[pairs[:, 0]], Z[pairs[:, 1]]; X = np.concatenate([a + b, a * b, np.abs(a - b)], 1)
    act = [np.tanh, lambda x: np.maximum(x, 0), np.sin][int(rng.integers(3))]
    W1 = rng.normal(size=(X.shape[1], h)) / np.sqrt(X.shape[1]); W2 = rng.normal(size=(h, K)) / np.sqrt(h)
    C = act(X @ W1 + rng.normal(size=h) * 0.5) @ W2
    if p_low > 0 and rng.random() < p_low:                                               # low-order: smooth, transferable across pairs
        XL = X[:, : 2 * r]; C = XL @ (rng.normal(size=(2 * r, K)) / np.sqrt(2 * r))
    w_mod = rng.uniform(0, 1); M = rng.normal(size=(P_, K)); Cm = M[pairs[:, 0]] * M[pairs[:, 1]]
    def programs():
        own = D[rng.integers(P_, size=K)]; own = own / (np.linalg.norm(own, axis=1, keepdims=True) + 1e-8)
        sp = rng.normal(size=(K, G)) * (rng.random((K, G)) < rng.uniform(0.02, 0.2)); sp = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
        m = rng.uniform(0, 1, (K, 1)); return m * own + (1 - m) * sp
    S = C @ programs(); S = S / (np.sqrt((S ** 2).sum(1).mean()) + 1e-12)
    Sm = Cm @ programs(); Sm = Sm / (np.sqrt((Sm ** 2).sum(1).mean()) + 1e-12)
    spec = (1 - w_mod) * S + w_mod * Sm
    base = np.sqrt((R ** 2).sum(1).mean()) + 1e-12; rho = np.exp(rng.uniform(np.log(0.3), np.log(3.0)))
    return rho * base * spec / (np.sqrt((spec ** 2).sum(1).mean()) + 1e-12)


def global_epistasis(ep, rng, p_on=0.75):
    """P4-U2 global-epistasis component (manuscript/notes/autonomous_log.md): interactions along the pair's own additive
    profile, R_AB += γ_AB (Δ_A + Δ_B), the signature of a saturating response on a shared phenotype axis. Per screen:
    γ_AB = γ0 − γ1 · rank_z(‖Δ_A + Δ_B‖) + a_A + a_B + e_AB with γ0 ~ U(−0.35, 0.10) (mostly buffering), saturation
    γ1 ~ U(0, 0.15), member effects a_p ~ N(0, σ_a²), σ_a ~ U(0, 0.12), pair residual e ~ N(0, σ_e²), σ_e ~ U(0, 0.10);
    off with probability 1 − p_on. Added identically to R_tr and R_te (signal)."""
    D = np.asarray(ep["D"], float); pairs = ep["pairs"]; S = D[pairs[:, 0]] + D[pairs[:, 1]]
    if rng.random() >= p_on:
        return np.zeros_like(S)
    n = np.linalg.norm(S, axis=1); rz = (np.argsort(np.argsort(n)) + 0.5) / len(n) * 2 - 1           # magnitude rank in [−1, 1]
    a = rng.normal(size=len(D)) * rng.uniform(0, 0.12)
    g = rng.uniform(-0.35, 0.10) - rng.uniform(0, 0.15) * rz + a[pairs[:, 0]] + a[pairs[:, 1]] + rng.normal(size=len(pairs)) * rng.uniform(0, 0.10)
    return g[:, None] * S


def member_signature(ep, rng, p0=0.25, tau=(0.3, 3.0)):
    """C3 candidate (manuscript/notes/c3_plan.md; Phase-0 E0a/E0b): member-additive interaction signatures.
    Real screens show strong cross-pair transfer between pairs sharing a member (signal cosine .06–.20 vs .02–.04 in the
    hybrid prior). Component: R_AB += τ·(u_A + u_B), u_p = Σ_k s_k c_pk π_k, with K ~ U{1..6} programs π_k (built like the
    screen-specific programs), decaying weights s_k = k^(−γ), γ ~ U(0, 2), and coefficients
    c_p = β·(z_p W) + sqrt(1 − β²)·ε_p: β ~ U(0, 1) sets how much of a perturbation's signature is predictable from its own
    single profile (z_p = normalised random projection of Δ_p, as in screen_specific) vs idiosyncratic (ε_p ~ N(0, I),
    learnable only from pairs sharing p). τ = power relative to the mechanistic R, log-uniform in `tau`; off w.p. p0.
    Added identically to R_tr and R_te (signal)."""
    D = np.asarray(ep["D"], float); P_, G = D.shape; pairs = ep["pairs"]; R = np.asarray(ep["R_tr"], float)
    if rng.random() < p0:
        return np.zeros_like(R)
    K = int(rng.integers(1, 7)); r = int(rng.integers(2, 13)); beta = rng.uniform(0, 1)
    Z = D @ rng.normal(size=(G, r)); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8) * np.sqrt(r)
    Cz = Z @ (rng.normal(size=(r, K)) / np.sqrt(r)); Cz = Cz / (Cz.std(0, keepdims=True) + 1e-8)
    C = beta * Cz + np.sqrt(1 - beta ** 2) * rng.normal(size=(P_, K))
    own = D[rng.integers(P_, size=K)]; own = own / (np.linalg.norm(own, axis=1, keepdims=True) + 1e-8)
    sp = rng.normal(size=(K, G)) * (rng.random((K, G)) < rng.uniform(0.02, 0.2)); sp = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
    m = rng.uniform(0, 1, (K, 1)); prog = m * own + (1 - m) * sp
    U = (C * np.arange(1, K + 1) ** (-rng.uniform(0, 2))) @ prog; S = U[pairs[:, 0]] + U[pairs[:, 1]]
    base = np.sqrt((R ** 2).sum(1).mean()) + 1e-12; t = np.exp(rng.uniform(np.log(tau[0]), np.log(tau[1])))
    return t * base * S / (np.sqrt((S ** 2).sum(1).mean()) + 1e-12)


def bilinear_response(ep, rng, p0=0.25, tau=(0.3, 3.0)):
    """C3 candidate 2 (manuscript/notes/autonomous_log.md, C3 quick iteration 1 diagnostics): interactions
    BILINEAR in the actual gene-space single profiles. R_AB += τ·Σ_k s_k u_k [a_Ak (v_k·Δ_B) + a_Bk (v_k·Δ_A)], with K ~ U{1..8} terms, decaying weights
    s_k = k^(−γ), γ ~ U(0, 2); input directions v_k and output programs u_k in gene space, each a random mix of a
    perturbation's own profile direction, a top principal direction of the screen's singles and a sparse random program;
    member gains a_pk = β·(w_k·Δ_p)/sd + sqrt(1 − β²)·ε_pk, β ~ U(0, 1): β → 1 gives profile × profile coupling (a
    symmetric bilinear form, ½H[Δ_A, Δ_B]), β → 0 member-specific modulation of the partner's profile (learnable only
    from pairs sharing the member). τ = power relative to the mechanistic R, log-uniform in `tau`; off w.p. p0."""
    D = np.asarray(ep["D"], float); P_, G = D.shape; pairs = ep["pairs"]; R = np.asarray(ep["R_tr"], float)
    if rng.random() < p0:
        return np.zeros_like(R)
    K = int(rng.integers(1, 9)); beta = rng.uniform(0, 1)
    Dc = D - D.mean(0); pcs = np.linalg.svd(Dc, full_matrices=False)[2][: min(10, P_)]
    unitv = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
    def dirs():
        own = unitv(D[rng.integers(P_, size=K)]); pc = pcs[rng.integers(len(pcs), size=K)] * rng.choice([-1, 1], (K, 1))
        sp = unitv(rng.normal(size=(K, G)) * (rng.random((K, G)) < rng.uniform(0.02, 0.2)))
        w = rng.dirichlet(np.ones(3), K); return unitv(w[:, :1] * own + w[:, 1:2] * pc + w[:, 2:] * sp)
    V, U, W = dirs(), dirs(), dirs()
    proj = D @ W.T; A = beta * proj / (proj.std(0, keepdims=True) + 1e-8) + np.sqrt(1 - beta ** 2) * rng.normal(size=(P_, K))
    s = np.arange(1, K + 1) ** (-rng.uniform(0, 2)); X = D @ V.T                                     # (P, K) partner projections
    a, b = pairs[:, 0], pairs[:, 1]; Cf = (A[a] * X[b] + A[b] * X[a]) * s
    S = Cf @ U; base = np.sqrt((R ** 2).sum(1).mean()) + 1e-12; t = np.exp(rng.uniform(np.log(tau[0]), np.log(tau[1])))
    return t * base * S / (np.sqrt((S ** 2).sum(1).mean()) + 1e-12)
