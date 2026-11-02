"""Shared data loading, expression statistics and metrics for the benchmarks.

Every method is scored on the same statistic: per-gene mean of
log1p(CP10k) over the backbone's genes, where CP10k uses the all-gene library.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "eval" / "results" / "bench"
sys.path.insert(0, str(ROOT / "src"))


def log(msg: str, fh=None) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


# ═══════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════

def load_kang() -> ad.AnnData:
    """Kang et al. 2018 IFN-β PBMC (pertpy copy): 24,673 cells, 8 donors, 8 cell types."""
    a = ad.read_h5ad(DATA / "kang2018" / "kang2018.h5ad")
    a.X = sp.csr_matrix(a.X)
    a.layers["counts"] = a.X.copy()
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a.obs["label"] = a.obs["label"].astype(str)
    a.obs["batch"] = a.obs["replicate"].astype(str)
    return a


def load_norman(min_cells: int = 20) -> tuple[ad.AnnData, pd.DataFrame]:
    """Norman et al. 2019 K562 CRISPRa (GSE133344).

    Returns the AnnData (obs['pert'] ∈ {'ctrl', gene, 'A+B'} with combos in
    sorted gene order, obs['batch'] = gemgroup) and a table of valid
    (A, B, A+B) triples where all three conditions have ≥ ``min_cells`` cells.
    """
    d = DATA / "norman2019"
    with gzip.open(d / "GSE133344_filtered_matrix.mtx.gz") as f:
        X = scipy.io.mmread(f).T.tocsr().astype(np.float32)
    with gzip.open(d / "GSE133344_filtered_barcodes.tsv.gz", "rt") as f:
        barcodes = [l.strip() for l in f]
    with gzip.open(d / "GSE133344_filtered_genes.tsv.gz", "rt") as f:
        genes = [l.rstrip("\n").split("\t") for l in f]
    meta = pd.read_csv(d / "GSE133344_filtered_cell_identities.csv.gz", index_col=0)
    meta = meta.reindex(barcodes)

    a = ad.AnnData(X=X)
    a.obs_names = [f"{b}_{i}" for i, b in enumerate(barcodes)]
    a.var_names = [g[1] if len(g) > 1 else g[0] for g in genes]
    a.var_names_make_unique()
    keep = meta["guide_identity"].notna().to_numpy()
    meta.index = a.obs_names
    a.obs = meta
    a = a[keep].copy()

    pair = a.obs["guide_identity"].str.split("__").str[0]
    g1, g2 = pair.str.split("_").str[0], pair.str.split("_").str[1]

    def classify(x: str, y: str) -> str:
        cx, cy = "NegCtrl" in x, "NegCtrl" in y
        if cx and cy:
            return "ctrl"
        if cy:
            return x
        if cx:
            return y
        return "+".join(sorted((x, y)))

    a.obs["pert"] = [classify(x, y) for x, y in zip(g1, g2)]
    a.obs["batch"] = a.obs["gemgroup"].astype(int).astype(str)
    a.layers["counts"] = a.X.copy()

    counts = a.obs["pert"].value_counts()
    rows = []
    for combo, n_ab in counts.items():
        if "+" not in combo:
            continue
        A, B = combo.split("+")
        if counts.get(A, 0) >= min_cells and counts.get(B, 0) >= min_cells and n_ab >= min_cells:
            rows.append(dict(A=A, B=B, AB=combo, n_A=int(counts[A]),
                             n_B=int(counts[B]), n_AB=int(n_ab)))
    triples = pd.DataFrame(rows).sort_values("n_AB", ascending=False).reset_index(drop=True)
    return a, triples


# ═══════════════════════════════════════════════════════════════════════════
# Expression statistics
# ═══════════════════════════════════════════════════════════════════════════

def log1p_cp10k(adata: ad.AnnData, features: list[str], rows: npt.NDArray | None = None,
                input_key: str = "counts") -> npt.NDArray[np.float32]:
    """Dense log1p(1e4 · c / L_total) on ``features`` for the selected rows."""
    raw = adata.layers[input_key]
    if rows is not None:
        raw = raw[rows]
    lib = np.asarray(raw.sum(axis=1)).ravel().astype(np.float32)
    sub = raw[:, adata.var_names.get_indexer(features)]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return np.log1p(1e4 * sub / np.maximum(lib, 1.0)[:, None]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def pearson(a: npt.NDArray, b: npt.NDArray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else float("nan")


def cosine(a: npt.NDArray, b: npt.NDArray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else float("nan")


TOP = 100


def tc(p: npt.NDArray, o: npt.NDArray) -> float:
    """cosine over the TOP genes with the largest absolute observed value; 0 if undefined"""
    top = np.argsort(-np.abs(o))[:TOP]; c = cosine(p[top], o[top]); return 0.0 if np.isnan(c) else c


def delta_metrics(pred: npt.NDArray, true: npt.NDArray, prefix: str = "") -> dict[str, float]:
    """Mean-profile metrics on predicted vs observed Δ (perturbed − control).

    top-k DE genes are chosen by |true Δ| (standard in GEARS/scGen benchmarks).
    """
    out: dict[str, float] = {}
    out[f"{prefix}pearson_delta_all"] = pearson(pred, true)
    order = np.argsort(-np.abs(true))
    for k in (20, 100):
        top = order[:k]
        out[f"{prefix}pearson_delta_de{k}"] = pearson(pred[top], true[top])
        out[f"{prefix}mse_de{k}"] = float(np.mean((pred[top] - true[top]) ** 2))
        out[f"{prefix}sign_acc_de{k}"] = float(np.mean(np.sign(pred[top]) == np.sign(true[top])))
    out[f"{prefix}mse_all"] = float(np.mean((pred - true) ** 2))
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
