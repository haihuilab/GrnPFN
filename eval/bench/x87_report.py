"""R4 report (pre-registered 2026-10-28; descriptive; run once):  python x87_report.py > ../results/bench/x87_gxd/report.txt
One pair per gene and one drug, so the dependence-aware bootstrap over members equals the bootstrap over pairs."""
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS

R = RESULTS / "x87_gxd"


def boot(x, n=2000):
    x = np.asarray(x); b = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1); return x.mean(), *np.percentile(b, [2.5, 97.5])


def pp(d, m, g, col):
    return d[(d.method == m) & (d.g == g)].groupby("AB")[col].mean()


c = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "curve_gxd_pf_s*.csv"))])
print(f"# A375 CRISPRi × vemurafenib (PerturbFate), held out; seeds {sorted(c.seed.unique())}; pairs {c.AB.nunique()}\n")
print("Levels (per-pair means):")
for col in ("cos_pair", "cos_pair_testc"):
    print(f"[{col}]\n" + c.groupby(["method", "g"])[col].mean().unstack().round(3).to_string())
print("\n## Zero shot (test-pair centering)")
for m in ("C-GBM", "C-ridge", "C-gamma"):
    x = (pp(c, "GrnPFN", 0.0, "cos_pair_testc") - pp(c, m, 0.0, "cos_pair_testc")).dropna(); mu, lo, hi = boot(x.values)
    print(f"- GrnPFN − {m}: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}] n={len(x)}; positive pairs {np.mean(x > 0):.2f}")
x = (pp(c, "GrnPFN", 0.0, "cos_pair_testc") - pp(c, "POLY", 0.15, "cos_pair_testc")).dropna(); mu, lo, hi = boot(x.values)
print(f"- GrnPFN (0 pairs) − POLY (15%): {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")
print("\n## Learning curve")
for col in ("cos_pair", "cos_pair_testc"):
    for g in (0.15, 0.25, 0.4, 1.0):
        for m in ("POLY", "RIDGE"):
            x = (pp(c, "GrnPFN", g, col) - pp(c, m, g, col)).dropna(); mu, lo, hi = boot(x.values)
            print(f"- [{col}] g={g} GrnPFN − {m}: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")
f = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "confidence_gxd_pf_s*.csv"))])
f["terc"] = f.groupby("seed").rel_unc.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["most", "middle", "least"]))
t = {k: f[f.terc == k].groupby("AB").cos_testc.mean() for k in ("most", "middle", "least")}
print("\n## Confidence (relative uncertainty terciles, zero shot)")
print("- " + "; ".join(f"{k} {v.mean():.3f} (n={len(v)})" for k, v in t.items()))
print(f"- Spearman(rel_unc, agreement) {spearmanr(f.rel_unc, f.cos_testc)[0]:+.3f}; Spearman(size, rel_unc) {spearmanr(f.sig, f.rel_unc)[0]:+.3f}")
