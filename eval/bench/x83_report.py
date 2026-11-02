"""R2 report (pre-registered 2026-10-28; descriptive). Run once after x83_fixedtarget.py has finished.
  python x83_report.py > ../results/bench/x83_fixedtarget/report.txt"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manuscript" / "scripts"))
from pfn_supp_tables import boot, cluster_pooled, pooled  # noqa: E402

from common import RESULTS  # noqa: E402
from x77_zeroshot import DS  # noqa: E402

R = RESULTS / "x83_fixedtarget"; FRESH = (431, 439, 449, 457)
METRICS = ("cos_reg", "cos_fix", "cos_testc", "cos256_fix", "nse")
SCHEMES = ("pfn-vdesign", "random", "poly-design", "single-norm", "pred-norm", "pred-var", "pred-norm-div")


def load(part, seeds):
    return pd.concat([pd.read_csv(f) for s in seeds for f in glob.glob(str(R / f"{part}_*_s{s}.csv"))])


def per_pair(d, ds, method, col, **kw):
    s = d[(d.dataset == ds) & (d.method == method)]
    for k, v in kw.items():
        s = s[s[k] == v]
    return s.groupby("AB")[col].mean()


def contrast(d, a, b, col, lab, sel_a=None, sel_b=None, q=(2.5, 97.5)):
    """a, b = (method, filters); nse is flipped so that positive always favours a"""
    sign = -1 if col == "nse" else 1; ser = []
    for ds, _ in DS:
        x = sign * (per_pair(d, ds, a[0], col, **a[1]) - per_pair(d, ds, b[0], col, **b[1])).dropna()
        if len(x):
            ser.append((ds, x))
    m, lo, hi = pooled([x.values for _, x in ser], q); _, clo, chi = cluster_pooled(ser, q); npos = sum(x.mean() > 0 for _, x in ser)
    print(f"- {lab} [{col}]: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] dep [{clo:+.4f}, {chi:+.4f}]; >0 {npos}/{len(ser)}; " + "; ".join(f"{n} {x.mean():+.4f}" for (ds, x), (_, n) in zip(ser, DS)))


def design(seeds, tag):
    d = load("design", seeds); print(f"\n## Design, g = .25, {tag} (seeds {seeds})\n")
    for m in ("GrnPFN", "POLY", "RIDGE"):
        print(f"\n### Levels, scored with {m}")
        for col in METRICS:
            t = d[d.method == m].groupby(["dataset", "scheme", "AB"])[col].mean().groupby(["dataset", "scheme"]).mean().unstack("scheme")[list(SCHEMES)]
            print(f"[{col}]\n" + t.round(3).to_string())
    for m in ("GrnPFN", "POLY", "RIDGE"):
        print(f"\n### Contrasts, scored with {m}")
        for col in METRICS:
            for o in SCHEMES[1:]:
                contrast(d, (m, dict(scheme="pfn-vdesign")), (m, dict(scheme=o)), col, f"{m}: vdesign − {o}")
            for o in ("single-norm", "pred-norm", "pred-var", "pred-norm-div", "poly-design"):
                contrast(d, (m, dict(scheme=o)), (m, dict(scheme="random")), col, f"{m}: {o} − random")
    print("\n### Pipelines")
    for col in METRICS:
        contrast(d, ("GrnPFN", dict(scheme="pfn-vdesign")), ("POLY", dict(scheme="poly-design")), col, "GrnPFN+vdesign − POLY+kernel design")
        contrast(d, ("GrnPFN", dict(scheme="pred-norm")), ("POLY", dict(scheme="poly-design")), col, "GrnPFN+pred-norm − POLY+kernel design")
    p = d[d.method == "POLY"].drop_duplicates(["dataset", "seed", "fold", "scheme", "draw"])
    print("\n### Recall of the strongest 20% of candidate pairs (share of them selected) and R_norm")
    print(p.groupby(["dataset", "scheme"])[["top20", "R_norm"]].mean().unstack("scheme").round(3).to_string())


def curve():
    c = load("curve", FRESH); print("\n## Main endpoints on the fresh splits under every metric (R2-B)\n")
    print(c.groupby(["dataset", "method", "g"])[list(METRICS)].mean().round(3).to_string())
    for col in METRICS[1:]:
        for z in ("C-GBM", "C-ridge", "C-gamma"):
            contrast(c, ("GrnPFN", dict(g=0.0)), (z, dict(g=0.0)), col, f"zero shot GrnPFN − {z}")
        for g in (0.15, 0.25, 0.4, 1.0):
            contrast(c, ("GrnPFN", dict(g=g)), ("POLY", dict(g=g)), col, f"GrnPFN − POLY at {g}")
        for g in (0.15, 1.0):
            contrast(c, ("GrnPFN", dict(g=g)), ("RIDGE", dict(g=g)), col, f"GrnPFN − RIDGE at {g}")
        contrast(c, ("GrnPFN", dict(g=0.0)), ("POLY", dict(g=0.15)), col, "GrnPFN at 0 − POLY at .15")
    contrast(c, ("GrnPFN", dict(g=0.15)), ("POLY", dict(g=0.15)), "cos_reg", "check: E1 under the registered metric")


def checks():
    print("## Reproduction of the registered scores (cos_reg vs earlier files; max |difference|)\n")
    for s in (301,) + FRESH:
        for ds, _ in DS:
            n = pd.read_csv(R / f"design_{ds}_s{s}.csv"); o = pd.read_csv(RESULTS / "x80_baselines" / f"design_{ds}_s{s}.csv"); out = []
            for m in ("POLY", "RIDGE"):
                for sc in ("pfn-vdesign", "random", "poly-design"):
                    a = n[(n.method == m) & (n.scheme == sc)].groupby(["fold", "draw", "AB"]).cos_reg.mean(); b = o[(o.method == m) & (o.scheme == sc)].groupby(["fold", "draw", "AB"]).cos_pair.mean()
                    j = pd.concat([a, b], axis=1).dropna(); out.append(np.abs(j.iloc[:, 0] - j.iloc[:, 1]).max())
            x = pd.read_csv(RESULTS / "x81_reviewer" / f"select_{ds}_s{s}.csv")
            for sc in ("pred-norm", "pred-var"):
                a = n[(n.method == "POLY") & (n.scheme == sc)].groupby(["fold", "AB"]).cos_reg.mean(); b = x[(x.method == "POLY") & (x.scheme == sc)].groupby(["fold", "AB"]).cos_pair.mean()
                j = pd.concat([a, b], axis=1).dropna(); out.append(np.abs(j.iloc[:, 0] - j.iloc[:, 1]).max())
            print(f"- s{s} {ds}: design/R1-A max |Δ| {max(out):.2e}")
        if s in FRESH:
            for ds, _ in DS:
                c = pd.read_csv(R / f"curve_{ds}_s{s}.csv"); o = pd.read_csv(RESULTS / "x80_baselines" / f"curve_{ds}_s{s}.csv"); p = pd.read_csv(RESULTS / "x79_confirm" / f"curve_s{s}.csv")
                a = c[c.method == "POLY"].groupby(["fold", "g", "draw", "AB"]).cos_reg.mean(); b = o[o.method == "POLY"].groupby(["fold", "g", "draw", "AB"]).cos_pair.mean()
                j = pd.concat([a, b], axis=1).dropna(); e1 = np.abs(j.iloc[:, 0] - j.iloc[:, 1]).max()
                a = c[(c.method == "GrnPFN") & (c.g > 0)].groupby(["fold", "g", "draw", "AB"]).cos_reg.mean()
                b = p[(p.dataset == ds) & (p.method == "PFN")].groupby(["fold", "g", "draw", "AB"]).cos_pair.mean()
                j = pd.concat([a, b], axis=1).dropna(); e2 = np.abs(j.iloc[:, 0] - j.iloc[:, 1]).max()
                a = c[(c.method == "C-GBM")].groupby(["fold", "AB"]).cos_testc.mean(); b = p[(p.dataset == ds) & (p.method == "C-GBM")].groupby(["fold", "AB"]).cos_pair_testc.mean()
                j = pd.concat([a, b], axis=1).dropna(); e3 = np.abs(j.iloc[:, 0] - j.iloc[:, 1]).max()
                print(f"  - curve s{s} {ds}: POLY {e1:.2e}, GrnPFN {e2:.2e}, C-GBM {e3:.2e}")


def calib():
    print("\n## Variance-head calibration (R1-D; x81_reviewer/calib_s*.csv), mean over folds (and splits)\n")
    for tag, seeds in (("dev 301", (301,)), ("fresh", FRESH)):
        c = pd.concat([pd.read_csv(RESULTS / "x81_reviewer" / f"calib_s{s}.csv") for s in seeds])
        print(tag + "\n" + c.groupby(["dataset", "g"])[["spearman", "ratio"]].mean().unstack("g").round(3).to_string())


def main():
    checks(); calib(); design((301,), "development split"); design(FRESH, "fresh splits"); curve()


if __name__ == "__main__":
    main()
