"""Benchmark folds and the models fitted to measured pairs (ridge, kernel ridge), shared by the GrnPFN pipeline
(grnpfn_real.py, x80_baselines.py)."""
from __future__ import annotations

import json

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from common import DATA

DATASETS = ["norman", "wessels", "sp4_A172", "sp4_T98G", "sp4_U87MG"]
N_PC, TOP, INNER_REPS, MIN_CELLS = 50, 100, 4, 20


# ── splits ────────────────────────────────────────────────────────────────────
def unit_of(g: str) -> str | None:
    """zero-shot unit: a gene, or a compound with all its doses; the fixed trametinib partner is never held out"""
    return None if g.startswith("tram@") else g.split("@")[0]


def unit_kfold(pairs, seed, n=5):
    """gene-level K-fold over pairs: validation pairs contain ≥ 1 held-out unit, training pairs contain none"""
    units = sorted({u for t in pairs for g in t if (u := unit_of(g))})
    out = []
    for _, te in KFold(n, shuffle=True, random_state=seed).split(units):
        T = {units[i] for i in te}
        hit = np.array([any(unit_of(g) in T for g in t) for t in pairs])
        out.append((np.flatnonzero(~hit), np.flatnonzero(hit)))
    return out


def dose_of(t) -> str:
    """compound dose level of a sci-Plex 4 pair (compound@dose, tram@dose)"""
    return t[0].split("@")[1]


def dose_folds(pairs):
    """leave-one-dose-level-out folds over pairs, ordered by dose"""
    levels = sorted({dose_of(t) for t in pairs}, key=float)
    return [(np.flatnonzero([dose_of(t) != d for t in pairs]), np.flatnonzero([dose_of(t) == d for t in pairs])) for d in levels]


def inner_splits(pairs, k, zero_shot, dose=False):
    if dose:                                                     # one repeat: leave each training dose level out
        return [dose_folds(pairs)]
    seeds = [k] + [1000 * r + k for r in range(1, INNER_REPS)]
    return [unit_kfold(pairs, s) if zero_shot else list(KFold(5, shuffle=True, random_state=s).split(pairs)) for s in seeds]


def dose_scaled(D, singles):
    """replace every compound's dose profiles by their projection on the compound's dominant direction (sign aligned
    with the mean profile), so that dose enters only as a scalar magnitude s_{X,d}: D'[X@d] = (u_X · D[X@d]) u_X"""
    out = {}
    by = {}
    for p in singles:
        by.setdefault(p.split("@")[0], []).append(p)
    for comp, ps in by.items():
        A = np.stack([D[p] for p in ps])
        u = np.linalg.svd(A, full_matrices=False)[2][0]
        u *= np.sign(u @ A.mean(0)) or 1.0
        for p in ps:
            out[p] = (D[p] @ u) * u
    return out


def outer_folds(dataset, rows_of, zero_shot, dose=False):
    """five folds of (train pairs, test pairs) over the pairs whose pair and singles each have ≥ 20 cells;
    dose=True: one fold per held-out compound dose level (sci-Plex 4)"""
    if dataset == "norman":                                     # folds of the earlier K562 benchmark (deep baselines)
        folds = json.loads((DATA / "bench_cache" / "norman_folds.json").read_text())
        valid = sorted({c for f in folds.values() for c in f["test_combos"]})
        split = [([valid.index(c) for c in folds[str(k)]["train_combos"]], [valid.index(c) for c in folds[str(k)]["test_combos"]])
                 for k in range(5)]
    else:
        ok = lambda p: len(rows_of.get(p, [])) >= MIN_CELLS
        valid = sorted(p for p in rows_of if "+" in p and ok(p) and all(ok(g) for g in p.split("+")))
        split = list(KFold(5, shuffle=True, random_state=0).split(valid))
    pairs = [tuple(c.split("+")) for c in valid]
    if dose:
        split = dose_folds(pairs)
    elif zero_shot:
        split = unit_kfold(pairs, 0)
    return [([pairs[i] for i in tr], [pairs[i] for i in te]) for tr, te in split]


def fit_ridge(D, trp, T, alpha, product):
    X_ = lambda ps: np.stack([np.r_[D[a] + D[b], D[a] * D[b]] if product else D[a] + D[b] for a, b in ps])
    m = Ridge(alpha).fit(X_(trp), np.stack([T[t] for t in trp]))
    return lambda ps: m.predict(X_(ps))


def fit_krr(D, trp, T, alpha, gamma):
    X_ = lambda ps: np.stack([D[a] + D[b] for a, b in ps])
    m = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(X_(trp), np.stack([T[t] for t in trp]))
    return lambda ps: m.predict(X_(ps))


def fit_poly(D, trp, T, alpha, gamma, coef0=1.0):
    """unstructured second-order control: kernel ridge regression with a degree-2 polynomial kernel on the concatenated
    single profiles [D_A, D_B], symmetrized over the order of the two members. Its feature space contains every constant,
    first-order and second-order monomial of (D_A, D_B), including D_A ⊗ D_A, D_B ⊗ D_B and D_A ⊗ D_B, with no structure
    imposed."""
    S = lambda ps: (np.stack([D[a] for a, _ in ps]), np.stack([D[b] for _, b in ps]))

    def G(ps, qs):
        (A, B), (Cc, Dd) = S(ps), S(qs)
        return 0.5 * ((gamma * (A @ Cc.T + B @ Dd.T) + coef0) ** 2 + (gamma * (A @ Dd.T + B @ Cc.T) + coef0) ** 2)
    m = KernelRidge(kernel="precomputed", alpha=alpha).fit(G(trp, trp), np.stack([T[t] for t in trp]))
    return lambda ps: m.predict(G(ps, trp))


def fit_shared_operator(D, trp, T, alpha):
    """one linear map J_0 fitted to both members of every training pair: R_AB ≈ J_0 Δ_q (q ∈ {A, B}); predicts ½J_0(Δ_A+Δ_B)"""
    Xq = np.stack([D[q] for t in trp for q in t]); Yq = np.stack([T[t] for t in trp for _ in t])
    A = Xq.T @ Xq / len(Xq); s = np.trace(A) / A.shape[0]
    J0 = (Yq.T @ Xq / len(Xq)) @ np.linalg.inv(A + alpha * s * np.eye(A.shape[0]))
    return lambda ps: np.stack([0.5 * J0 @ (D[a] + D[b]) for a, b in ps])
