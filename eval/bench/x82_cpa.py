"""R1-C (pre-registered 2026-10-28, manuscript/notes/autonomous_log.md): the K562 recipe of p5_cpa_norman.py applied
unchanged to another screen (--dataset wessels|sp4_A172|sp4_T98G|sp4_U87MG); writes p82_external/.

Run with the CPA environment:  ../../.venv-cpa/bin/python p5_cpa_norman.py --fold 0 [--epochs 200]
Model and training hyperparameters follow the CPA Norman tutorial, with a Gaussian likelihood because the
cache holds log1p(CP10k). Test-combination cells are registered (split 'ood') but never trained on.
Counterfactuals: the cells of each queried condition get control-cell expression in X (CPA's documented
recipe), so CPA encodes a control basal state and adds the queried perturbation. Writes
cpa_fold{k}.npz (raw) and cpares_fold{k}.npz (measured A + B plus CPA's own interaction) like CellFlow.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "bench_cache"
OUT = ROOT / "eval" / "results" / "bench" / "p82_external"

MODEL = dict(n_latent=32, recon_loss="gauss", doser_type="linear",
             n_hidden_encoder=256, n_layers_encoder=4, n_hidden_decoder=256, n_layers_decoder=2,
             use_batch_norm_encoder=True, use_layer_norm_encoder=False,
             use_batch_norm_decoder=False, use_layer_norm_decoder=False,
             dropout_rate_encoder=0.2, dropout_rate_decoder=0.0, variational=False)
TRAIN = dict(n_epochs_kl_warmup=None, n_epochs_adv_warmup=50, n_epochs_mixup_warmup=10,
             n_epochs_pretrain_ae=10, mixup_alpha=0.1, lr=1e-4, wd=3.2e-6, adv_steps=3,
             reg_adv=10.0, pen_adv=20.0, adv_lr=1e-4, adv_wd=7.1e-6, n_layers_adv=2,
             n_hidden_adv=128, use_batch_norm_adv=True, use_layer_norm_adv=False,
             dropout_rate_adv=0.3, step_size_lr=25, do_clip_grad=False,
             gradient_clip_value=1.0, adv_loss="cce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-pred-ctrl", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import cpa

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = json.loads((CACHE / f"{args.dataset}_folds.json").read_text())[str(args.fold)]
    a = ad.read_h5ad(CACHE / f"{args.dataset}_hvg3000.h5ad")
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    pert = a.obs["pert"].astype(str).to_numpy()
    a.obs = a.obs[[]].copy()
    # label-only fix (R1-C): CPA 0.8 looks components up by substring (str.contains), so "X@1" would match "X@10"; a
    # terminal "_" on every component makes the tokens prefix-free. Rows are still selected by the original labels.
    tok = lambda p: p if p == "ctrl" else "+".join(c + "_" for c in p.split("+"))
    a.obs["condition"] = [tok(p) for p in pert]
    a.obs["cell_type"] = args.dataset
    a.obs["smiles"] = [c.replace("+", "..") for c in a.obs["condition"]]  # placeholder: CPA 0.8 looks up a SMILES per perturbation
    rng = np.random.default_rng(args.seed + args.fold)
    split = np.where(rng.random(a.n_obs) < 0.1, "valid", "train").astype(object)
    split[np.isin(pert, f["test_combos"])] = "ood"
    a.obs["split"] = split
    ctrl_rows = np.flatnonzero(pert == "ctrl")
    mean = lambda m: np.asarray(a.X[m].mean(0)).ravel()
    ctrl_e = mean(pert == "ctrl")

    cpa.CPA.setup_anndata(a, perturbation_key="condition", control_group="ctrl", dosage_key=None,
                          categorical_covariate_keys=["cell_type"], smiles_key="smiles",
                          is_count_data=False, max_comb_len=2)
    m = cpa.CPA(a, split_key="split", train_split="train", valid_split="valid", test_split="ood",
                seed=args.seed, **MODEL)
    with tempfile.TemporaryDirectory() as tmp:
        m.train(max_epochs=args.epochs, use_gpu=True, batch_size=2048, plan_kwargs=TRAIN,
                early_stopping_patience=5, check_val_every_n_epoch=5, save_path=tmp)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)

    # One prediction batch per queried condition: its registered rows, with control expression in X.
    pick = rng.choice(ctrl_rows, min(args.n_pred_ctrl, len(ctrl_rows)), replace=False)
    Xc = a.X[pick]
    conds = sorted(set(f["test_combos"]) | set(f["test_A"]) | set(f["test_B"]))
    rows, owner = [], []
    for c in conds:
        r = np.flatnonzero(pert == c)
        r = np.resize(r, len(pick))  # repeat the condition's encoding over the control cells
        rows.append(r)
        owner += [c] * len(pick)
    q = a[np.concatenate(rows)].copy()
    q.X = sp.vstack([Xc] * len(conds)).tocsr()
    m.predict(q, batch_size=2048)
    pred = np.asarray(q.obsm["CPA_pred"])
    owner = np.asarray(owner)
    base = mean(pick)  # CPA's prediction is compared with the control cells it started from
    d = {c: pred[owner == c].mean(0) - base for c in conds}
    combos, raw, res = [], [], []
    for A, B in zip(f["test_A"], f["test_B"]):
        ab = "+".join(sorted((A, B)))
        combos.append(ab)
        raw.append(d[ab] + (base - ctrl_e))
        res.append(mean(pert == A) - ctrl_e + mean(pert == B) - ctrl_e + d[ab] - d[A] - d[B])
    for name, dd in (("cpa", raw), ("cpares", res)):
        np.savez(OUT / f"{name}{args.tag}_{args.dataset}_fold{args.fold}.npz", combos=np.array(combos),
                 delta=np.stack(dd).astype(np.float32), genes=np.asarray(a.var_names))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
