"""C4 H-ADAPT report, exactly as pre-registered in manuscript/notes/autonomous_log.md (written before any result existed).
Per test pair: mean over folds (and random draws) at each budget, then the average of the two budgets (g = .25, .40).
Pooled contrast = mean over the 5 screens of the per-screen mean paired difference (equal screen weights); 95% CI by
bootstrap over pairs within screens (2,000 resamples). Contrast of interest: pfn-adaptive − pfn-vdesign (one-shot).
'Adaptivity helps' only if pooled mean > 0, CI excludes 0 and ≥ 4/5 screens > 0.   python x76_report.py"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
R = Path(__file__).resolve().parents[1] / "results" / "bench" / "x76_adaptive"
DS = [("norman", "K562"), ("wessels", "THP-1"), ("sp4_A172", "A172"), ("sp4_T98G", "T98G"), ("sp4_U87MG", "U87MG")]


def per_pair(d, method, scheme):
    s = d[(d.method == method) & (d.scheme == scheme)]
    by_g = s.groupby(["g", "AB"]).cos_pair.mean().unstack("g")                    # mean over folds and draws per budget
    return by_g.dropna().mean(1)                                                   # average of the two budgets per pair


def pooled(diffs, n=2000, seed=0):
    rng = np.random.default_rng(seed); bs = np.zeros(n)
    for x in diffs:
        bs += x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    bs /= len(diffs); return np.mean([x.mean() for x in diffs]), *np.percentile(bs, [2.5, 97.5])


def main():
    data = {ds: pd.read_csv(R / f"{ds}.csv") for ds, _ in DS if (R / f"{ds}.csv").exists()}
    missing = [ds for ds, _ in DS if ds not in data]
    print(f"# C4 H-ADAPT (dev 301){'  — INCOMPLETE, missing: ' + ', '.join(missing) if missing else ''}\n")
    cons = [("pfn-adaptive", "pfn-vdesign", "PFN-adaptive − one-shot vdesign"), ("pfn-adaptive", "random", "PFN-adaptive − random"),
            ("pfn-vdesign", "random", "one-shot vdesign − random")]
    for method in ("PFN",):
        print(f"## Predictor: {method}\n")
        means = pd.DataFrame({name: {s: per_pair(data[ds], method, s).mean() for s in ("pfn-adaptive", "pfn-vdesign", "random")}
                              for ds, name in DS if ds in data}).T
        print(means.round(4).to_string(), "\n")
        for a, b, lab in cons:
            diffs, rows = [], []
            for ds, name in DS:
                if ds not in data:
                    continue
                x = (per_pair(data[ds], method, a) - per_pair(data[ds], method, b)).dropna(); diffs.append(x.values)
                rows.append(f"{name} {x.mean():+.4f} (P={wilcoxon(x).pvalue:.2g}, n={len(x)})")
            m, lo, hi = pooled(diffs); npos = sum(x.mean() > 0 for x in diffs)
            verdict = ""
            if a == "pfn-adaptive" and b == "pfn-vdesign":
                ok = m > 0 and lo > 0 and npos >= 4 and not missing
                verdict = f"  → {'PASS' if ok else 'FAIL'} (needs pooled > 0, CI excluding 0, ≥ 4/5 screens > 0)"
            print(f"- {lab}: pooled {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; screens > 0: {npos}/{len(diffs)}{verdict}")
            print("  - " + "; ".join(rows))
        print()
    print("## Discovery (measured set; means over folds, budgets and draws)\n")
    props = pd.concat([pd.read_csv(R / f"{ds}_props.csv") for ds in data])
    t = props.groupby(["dataset", "scheme"])[["R_norm", "top20"]].mean().unstack("scheme")
    print(t.round(3).to_string())
    print("\nPooled (equal screen weights): " + "; ".join(f"{s}: R_norm {t['R_norm'][s].mean():.3f}, top-20% capture {t['top20'][s].mean():.3f}" for s in t["R_norm"].columns))


if __name__ == "__main__":
    main()
