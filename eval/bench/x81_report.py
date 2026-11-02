"""R1 reviewer controls report (pre-registered 2026-10-28; secondary, descriptive, no decision criterion). Run once
after all parts finish:  python x81_report.py > ../results/bench/x81_reviewer/report.txt
Per-pair means over draws, folds and splits; per-screen 95% bootstrap intervals over pairs; pooled over screens with equal
weights (x79_report.pooled, 95%)."""
import glob

import numpy as np
import pandas as pd

from common import RESULTS
from x77_zeroshot import DS
from x79_report import pooled

R81, C, B, P82 = RESULTS / "x81_reviewer", RESULTS / "x79_confirm", RESULTS / "x80_baselines", RESULTS / "p82_external"
FRESH = (431, 439, 449, 457); NAME = dict(DS)


def boot(x, n=2000):
    rng = np.random.default_rng(0); return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), (2.5, 97.5))


def pp(d, ds, method, scheme=None, g=None, col="cos_pair"):
    s = d[(d.dataset == ds) & (d.method == method)]
    if scheme is not None:
        s = s[s.scheme == scheme]
    if g is not None:
        s = s[s.g == g]
    return s.groupby("AB")[col].mean()


def contrast(d, a, b, lab, col="cos_pair", screens=DS):
    diffs, parts = [], []
    for ds, name in screens:
        x = (pp(d, ds, *a, col=col) - pp(d, ds, *b, col=col)).dropna()
        if len(x) == 0:
            continue
        lo, hi = boot(x.values); diffs.append(x.values); parts.append(f"{name} {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(x)})")
    if not diffs:
        print(f"- {lab}: no data"); return
    m, lo, hi = pooled(diffs); npos = sum(x.mean() > 0 for x in diffs)
    print(f"- {lab} [{col}]: pooled {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; screens > 0: {npos}/{len(diffs)}\n  - " + "; ".join(parts))


def design_frame(seeds):
    dev = seeds == (301,)
    new = pd.concat([pd.read_csv(f) for s in seeds for f in glob.glob(str(R81 / f"select_*_s{s}.csv"))])
    if dev:
        v = pd.concat([pd.read_csv(f) for f in glob.glob(str(RESULTS / "x60_pfn_design" / "*_s301_grnpfn_rel_var.csv"))])
        v = v[(v.g == 0.25) & v.scheme.isin(["pfn-vdesign", "random"])].assign(method="GrnPFN")
    else:
        v = pd.concat([pd.read_csv(C / f"design_{ds}_s{s}.csv") for s in seeds for ds, _ in DS])
        v = v[(v.method == "PFN") & v.scheme.isin(["pfn-vdesign", "random"])].assign(method="GrnPFN")
    b = pd.concat([pd.read_csv(f) for s in seeds for f in glob.glob(str(B / f"design_*_s{s}.csv"))])
    return pd.concat([new, v, b])


def r1a(seeds):
    d = design_frame(seeds); print(f"\n## R1-A selection controls, g = .25, seeds {list(seeds)}\n")
    print(d[d.method.isin(["GrnPFN", "POLY", "RIDGE"])].groupby(["dataset", "method", "scheme"]).cos_pair.mean().unstack("scheme").round(3).to_string(), "\n")
    for m in ("GrnPFN", "POLY", "RIDGE"):
        print(f"Predictor {m}:")
        contrast(d, (m, "pfn-vdesign"), (m, "pred-norm"), f"{m}: GrnPFN design − largest predicted interaction")
        contrast(d, (m, "pfn-vdesign"), (m, "pred-var"), f"{m}: GrnPFN design − largest predictive variance")
        contrast(d, (m, "pred-norm"), (m, "random"), f"{m}: largest predicted interaction − random")
        contrast(d, (m, "pred-var"), (m, "random"), f"{m}: largest predictive variance − random")
    sp = pd.concat([pd.read_csv(f) for s in seeds for f in glob.glob(str(R81 / f"selprops_*_s{s}.csv"))])
    disc = pd.read_csv(RESULTS.parent.parent.parent / "manuscript" / "figures" / "main_pfn" / ("Fig4_discovery_s301.csv" if seeds == (301,) else "Fig4_discovery_fresh.csv"))
    if "g" in disc:
        disc = disc[disc.g == 0.25]
    disc = disc.groupby(["dataset", "fold", "seed", "scheme"] if "seed" in disc else ["dataset", "fold", "scheme"]).top20.mean().reset_index()
    t = pd.concat([disc.groupby(["dataset", "scheme"]).top20.mean(), sp.groupby(["dataset", "scheme"]).top20.mean()]).unstack("scheme")
    print("\nShare of the strongest 20% of candidate interactions selected (top20):\n" + t.round(3).to_string())
    print("\nR_norm of the new rules:\n" + sp.groupby(["dataset", "scheme"]).R_norm.mean().unstack().round(3).to_string())


def curves(seeds):
    if seeds == (301,):
        pfn = pd.read_csv(C / "smoke_dev301" / "curve_s301.csv")
    else:
        pfn = pd.concat([pd.read_csv(C / f"curve_s{s}.csv") for s in seeds])
    pfn = pfn[pfn.method == "PFN"].assign(method="GrnPFN")
    base = pd.concat([pd.read_csv(f) for s in seeds for f in glob.glob(str(B / f"curve_*_s{s}.csv"))])
    return pd.concat([pfn, base])


def r1b(seeds):
    d = curves(seeds); print(f"\n## R1-B raw cosine (uncentered), seeds {list(seeds)}\n")
    print(d.groupby(["dataset", "method", "g"]).cos_raw.mean().unstack("g").round(3).to_string(), "\n")
    for g in (0.15, 0.25, 0.4, 1.0):
        for m in ("POLY", "RIDGE"):
            contrast(d, ("GrnPFN", None, g), (m, None, g), f"GrnPFN@{g} − {m}@{g}", col="cos_raw")


def r1c(seeds):
    d = curves(seeds); d = d[d.g == 1.0]
    sc = pd.concat([pd.read_csv(P82 / f"score_s{s}.csv") for s in seeds]); d = pd.concat([d, sc])
    q = pd.concat([pd.read_csv(P82 / f"queryable_s{s}.csv") for s in seeds]).drop_duplicates(["dataset", "fold", "method"])
    print(f"\n## R1-C deep models on the other screens (all training pairs, benchmark centering), seeds {list(seeds)}\n")
    print("Queryable test pairs:\n" + q.groupby(["dataset", "method"])[["queryable", "test"]].sum().to_string() + "\n")
    print(d[d.dataset != "norman"].groupby(["dataset", "method"]).cos_pair.mean().unstack("method").round(3).to_string(), "\n")
    oth = [x for x in DS if x[0] != "norman"]
    for m in ("gears", "cellflow", "cpa"):
        contrast(d, ("GrnPFN",), (m,), f"GrnPFN − {m}", screens=oth)
        contrast(d, ("POLY",), (m,), f"POLY − {m}", screens=oth)


def r1d(seeds):
    c = pd.concat([pd.read_csv(R81 / f"calib_s{s}.csv") for s in seeds])
    print(f"\n## R1-D variance-head calibration (fold means; ratio = realized / predicted squared error), seeds {list(seeds)}\n")
    print(c.groupby(["dataset", "g"])[["ratio", "spearman"]].mean().unstack("g").round(3).to_string())


if __name__ == "__main__":
    print("# R1 reviewer controls (pre-registered 2026-10-28; secondary, descriptive, no decision criterion)")
    for seeds, lab in (((301,), "DEVELOPMENT split 301"), (FRESH, "confirmation splits 431/439/449/457 (already used for C5/C6)")):
        print(f"\n# {lab}")
        r1a(seeds); r1b(seeds); r1c(seeds); r1d(seeds)
