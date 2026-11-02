"""R3 report (pre-registered 2026-10-28; descriptive). Run once:  python x86_report.py > ../results/bench/x86_novelty/report.txt"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manuscript" / "scripts"))
from pfn_supp_tables import cluster_pooled, pooled  # noqa: E402

from common import RESULTS  # noqa: E402
from x77_zeroshot import DS  # noqa: E402

R = RESULTS / "x86_novelty"


def novelty():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "novelty_*_*.csv"))])
    print("## R3-A Unseen perturbations (fresh splits)\n")
    print("Share of (pair, draw, split) records with no perturbation seen, by screen and budget:")
    print(d[d.method == "GrnPFN"].assign(none=lambda x: x.cls == "none seen").groupby(["dataset", "g"]).none.mean().unstack().round(3).to_string(), "\n")
    for col in ("cos_pair", "cos_pair_testc"):
        for g in (0.15, 0.25, 0.4):
            for cls in ("none seen", "seen"):
                ser, lev = [], []
                for ds, name in DS:
                    x = d[(d.dataset == ds) & (d.g == g) & (d.cls == cls)]
                    a = x[x.method == "GrnPFN"].groupby("AB")[col].mean(); b = x[x.method == "POLY"].groupby("AB")[col].mean(); df = (a - b).dropna()
                    lev.append(f"{name}: n={len(df)} GrnPFN {a.mean():.3f} POLY {b.mean():.3f} diff {df.mean():+.3f}")
                    if len(df) >= 5:
                        ser.append((ds, df))
                m, lo, hi = pooled([v.values for _, v in ser]); _, clo, chi = cluster_pooled(ser)
                print(f"- [{col}] g={g} {cls}: pooled GrnPFN−POLY {m:+.4f} [{lo:+.4f}, {hi:+.4f}] dep [{clo:+.4f}, {chi:+.4f}] over {len(ser)} screens; " + "; ".join(lev))
    # genetic screens by number of seen members
    print("\nGenetic screens by number of seen members (cos_pair, g = .15):")
    for ds in ("norman", "wessels"):
        x = d[(d.dataset == ds) & (d.g == 0.15)]
        for k in (0, 1, 2):
            y = x[x.n_seen == k]; a = y[y.method == "GrnPFN"].groupby("AB").cos_pair.mean(); b = y[y.method == "POLY"].groupby("AB").cos_pair.mean()
            print(f"- {ds} {k} seen: n={len((a - b).dropna())} GrnPFN {a.mean():.3f} POLY {b.mean():.3f} diff {(a - b).dropna().mean():+.3f}")


def confidence():
    c = pd.read_csv(R / "confidence.csv")
    x83 = pd.concat([pd.read_csv(f) for f in glob.glob(str(RESULTS / "x83_fixedtarget" / "curve_*_s4[3-5]*.csv"))])
    ref = x83[(x83.method == "GrnPFN") & (x83.g == 0.0)].groupby(["dataset", "seed", "fold", "AB"]).cos_testc.mean()
    j = c.set_index(["dataset", "seed", "fold", "AB"]).cos_testc.to_frame("new").join(ref.rename("x83"), how="inner")
    print(f"\n## R3-B Know when to trust it (fresh splits, no measured pairs)\n\nCheck vs x83 zero-context GrnPFN (grnpfn_rel): n={len(j)}, max |Δcos| {np.abs(j.new - j.x83).max():.2e}, mean Δ {(j.new - j.x83).mean():+.1e}\n")
    for score_col, lab in (("rel_unc", "relative uncertainty tr V / ‖f‖² (primary)"), ("trV", "tr V (secondary)")):
        c["terc"] = c.groupby(["dataset", "seed"])[score_col].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["most confident", "middle", "least confident"]))
        print(f"### {lab}")
        t = c.groupby(["dataset", "terc"], observed=True).cos_testc.mean().unstack(); print(t[["most confident", "middle", "least confident"]].round(3).to_string())
        ser = []
        for ds, name in DS:
            y = c[c.dataset == ds]
            hi = y[y.terc == "most confident"].groupby("AB").cos_testc.mean(); lo = y[y.terc == "least confident"].groupby("AB").cos_testc.mean()
            rho = spearmanr(y[score_col], y.cos_testc)[0]
            print(f"- {name}: most − least confident {hi.mean() - lo.mean():+.3f} (pairs {len(hi)}/{len(lo)}); Spearman(score, agreement) {rho:+.3f}")
        # pooled: per pair the tercile is per split; use split-level records averaged per pair within tercile, difference of tercile means per screen
        diffs = []
        for ds, _ in DS:
            y = c[c.dataset == ds]; hi = y[y.terc == "most confident"].groupby("AB").cos_testc.mean().values; lo = y[y.terc == "least confident"].groupby("AB").cos_testc.mean().values
            diffs.append((hi, lo))
        rng = np.random.default_rng(0); bs = np.zeros(2000)
        for hi, lo in diffs:
            bs += hi[rng.integers(0, len(hi), (2000, len(hi)))].mean(1) - lo[rng.integers(0, len(lo), (2000, len(lo)))].mean(1)
        bs /= len(diffs); m = np.mean([h.mean() - l.mean() for h, l in diffs])
        print(f"- pooled most − least confident: {m:+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (independent-groups bootstrap over pairs within screens)\n")


if __name__ == "__main__":
    novelty(); confidence()
