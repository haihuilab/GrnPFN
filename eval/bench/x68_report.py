"""U3 report: TTT vs base (and sham TTT) per screen and budget. Per-pair means over folds/draws, cluster bootstrap
over pairs (2,000), two-sided Wilcoxon on per-pair differences.
python x68_report.py [model]"""
import glob, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
R = Path(__file__).resolve().parents[1] / "results" / "bench"; mdl = sys.argv[1] if len(sys.argv) > 1 else "grnpfn_rel"
t = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "x68_ttt" / f"*_{mdl}*.csv")) if not f.endswith("_stops.csv")])
df = t.drop_duplicates(["dataset", "fold", "g", "draw", "AB", "method"])
B, T, S, SP = f"{mdl} [base]", f"{mdl} [TTT]", f"{mdl}_sham [TTT]", f"{mdl}_shampairs [TTT]"; rng = np.random.default_rng(0)

def cmp(p, a, b):
    if a not in p or b not in p:
        return None
    d = (p[a] - p[b]).dropna().groupby(level="AB").mean().values
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), wilcoxon(d).pvalue, len(d)

rows = []
for (ds, g), s in df.groupby(["dataset", "g"]):
    p = s.pivot_table(index=["fold", "draw", "AB"], columns="method", values="cos_pair")
    if T not in p:
        continue
    r = dict(dataset=ds, g=g, base=p[B].mean(), ttt=p[T].mean())
    for nm, (a, b) in {"TTT-base": (T, B), "sham-base": (S, B), "shampairs-base": (SP, B)}.items():
        c = cmp(p, a, b)
        if c:
            r[nm] = c[0]; r[nm + " CI"] = f"[{c[1]:+.3f},{c[2]:+.3f}]"; r[nm + " P"] = c[3]
    rows.append(r)
o = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(o.round(4).to_string(index=False))
print("\ncurve gain (g=1 minus g=.15)")
for ds, s in o.groupby("dataset"):
    s = s.set_index("g")
    if 1.0 in s.index and 0.15 in s.index:
        print(f"{ds:9s} base {s.base[1.0] - s.base[.15]:+.3f}  TTT {s.ttt[1.0] - s.ttt[.15]:+.3f}")
st = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "x68_ttt" / f"*_{mdl}_stops.csv"))])
print("\ninner-stopping s* distribution\n", st.groupby(["dataset", "g"]).sstar.describe()[["mean", "min", "max"]].round(1).to_string())
