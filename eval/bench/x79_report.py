"""C5 report, exactly as pre-registered (manuscript/notes/autonomous_log.md, 2026-10-27; written before any result).
Per test pair: mean over draws, folds and the seeds; pooled over the 5 screens with equal weights; bootstrap over pairs
within screens. The zero-shot endpoint is judged with a 97.5% CI (Bonferroni over two endpoints).
  python x79_report.py [--seeds 431 439 449 457]"""
import argparse
import glob

import numpy as np
import pandas as pd

from x77_zeroshot import DS

R = __import__("common").RESULTS / "x79_confirm"


def pooled(diffs, q=(2.5, 97.5), n=2000, seed=0):
    rng = np.random.default_rng(seed); bs = np.zeros(n)
    for x in diffs:
        bs += x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    bs /= len(diffs); return np.mean([x.mean() for x in diffs]), *np.percentile(bs, q)


def contrast(d, a, b, col, lab, q=(2.5, 97.5), rule=False, g=None):
    diffs = []
    for ds, _ in DS:
        s = d[d.dataset == ds] if g is None else d[(d.dataset == ds) & (d.g == g)]
        pa = s[s.method == a[0]] if a[1] is None else s[(s.method == a[0]) & (s.scheme == a[1])]
        pb = s[s.method == b[0]] if b[1] is None else s[(s.method == b[0]) & (s.scheme == b[1])]
        if a[2] is not None:
            pa = pa[pa.g == a[2]]
        if b[2] is not None:
            pb = pb[pb.g == b[2]]
        diffs.append((pa.groupby("AB")[col].mean() - pb.groupby("AB")[col].mean()).dropna())
    ok = [x for x in diffs if len(x)]
    if not ok:
        return
    m, lo, hi = pooled([x.values for x in ok], q); npos = sum(x.mean() > 0 for x in ok)
    v = f"  → {'CONFIRMED' if (m > 0 and lo > 0 and npos >= 4 and len(ok) == 5) else 'NOT CONFIRMED'}" if rule else ""
    print(f"- {lab} [{col}]: pooled {m:+.4f} [{lo:+.4f}, {hi:+.4f}] ({q[1] - q[0]:.1f}% CI); screens > 0: {npos}/{len(ok)}{v}")
    print("  - " + "; ".join(f"{n} {x.mean():+.4f} (n={len(x)})" for (_, n), x in zip(DS, diffs) if len(x)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, default=[431, 439, 449, 457]); a = ap.parse_args()
    cur = pd.concat([pd.read_csv(f) for s in a.seeds for f in glob.glob(str(R / f"curve_s{s}.csv"))])
    des = [pd.read_csv(f) for s in a.seeds for f in glob.glob(str(R / f"design_*_s{s}.csv"))]
    ext = [pd.read_csv(f) for s in a.seeds for f in glob.glob(str(R / f"external_s{s}.csv"))]
    print(f"# C5 confirmation on fresh seeds {sorted(cur.seed.unique().tolist())} (pre-registered)\n")
    print(cur.groupby(["dataset", "method", "g"]).cos_pair.mean().unstack("g").round(3).to_string(), "\n")
    print("## Zero-shot endpoint (97.5% CI)\n")
    contrast(cur, ("PFN [empty]", None, None), ("C-GBM", None, None), "cos_pair_testc", "E2 zero shot: PFN(empty) − C-GBM", (1.25, 98.75), rule=True)
    print("\n## Secondary (95% CIs, no criterion)\n")
    for c in ("C-ridge", "C-gamma"):
        contrast(cur, ("PFN [empty]", None, None), (c, None, None), "cos_pair_testc", f"PFN(empty) − {c}")
    contrast(cur, ("PFN [empty]", None, None), ("C-GBM", None, None), "cos_raw", "E2 raw: PFN(empty) − C-GBM")
    if des:
        d = pd.concat(des); print(f"\nDesign (g = .25; screens: {sorted(d.dataset.unique())}):")
        contrast(d, ("PFN", "pfn-vdesign", None), ("PFN", "random", None), "cos_pair", "PFN + vdesign − PFN + random")
    if ext:
        e = pd.concat(ext); p = cur[(cur.dataset == "norman") & (cur.method == "PFN") & (cur.g == 1.0)].groupby("AB").cos_pair.mean()
        print("\nK562, all training pairs vs external deep models:")
        for m in ("gears", "cellflow", "cpa"):
            x = (p - e[e.method == m].groupby("AB").cos_pair.mean()).dropna(); mm, lo, hi = pooled([x.values])
            print(f"- PFN − {m}: {mm:+.4f} [{lo:+.4f}, {hi:+.4f}] n={len(x)} ({m} {e[e.method == m].cos_pair.mean():.3f})")


if __name__ == "__main__":
    main()
