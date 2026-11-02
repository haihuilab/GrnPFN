"""U5 report: TTT with three-way-split inner stopping (split3) vs U3 TTT and base; inner-vs-test gain calibration.
python x70_u5_report.py"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
R = Path(__file__).resolve().parents[1] / "results" / "bench" / "x68_ttt"; rng = np.random.default_rng(0)
B, U3, U5 = "grnpfn_rel [base]", "grnpfn_rel [TTT]", "grnpfn_rel_split3 [TTT]"; rows, cal = [], []
for ds in ("norman", "wessels", "sp4_A172", "sp4_T98G"):
    if not (R / f"{ds}_grnpfn_rel_split3.csv").exists():
        continue
    t = pd.concat([pd.read_csv(R / f"{ds}_grnpfn_rel.csv"), pd.read_csv(R / f"{ds}_grnpfn_rel_split3.csv")]).drop_duplicates(["fold", "g", "draw", "AB", "method"])
    for g, s in t.groupby("g"):
        p = s.pivot_table(index=["fold", "draw", "AB"], columns="method", values="cos_pair"); r = dict(screen=ds, g=g)
        for lab, a, b in (("U5-base", U5, B), ("U3-base", U3, B), ("U5-U3", U5, U3)):
            d = (p[a] - p[b]).dropna().groupby(level="AB").mean().values; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            r[lab] = d.mean(); r[lab + " CI"] = f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]"; r[lab + " P"] = wilcoxon(d).pvalue
        rows.append(r)
    for tag, m in (("", U3), ("_split3", U5)):
        st = pd.read_csv(R / f"{ds}_grnpfn_rel{tag}_stops.csv"); st["inner"] = st.apply(lambda q: q[f"v{int(q.sstar)}"] - q["v0"], axis=1)
        pv = t.pivot_table(index=["fold", "g", "draw"], columns="method", values="cos_pair", aggfunc="mean"); gain = (pv[m] - pv[B]).rename("test").reset_index()
        x = st.merge(gain, on=["fold", "g", "draw"]); x["dataset"], x["rule"] = ds, "U5" if tag else "U3"; cal.append(x)
o = pd.DataFrame(rows); pd.set_option("display.width", 250); print(o.round(4).to_string(index=False))
c = pd.concat(cal)
print("\ninner vs test gain")
for (ds, rule), x in c.groupby(["dataset", "rule"]):
    print(f"{ds:9s} {rule}: mean inner {x.inner.mean():+.3f} test {x.test.mean():+.3f} |inner-test| {np.abs(x.inner - x.test).mean():.3f} Spearman {spearmanr(x.inner, x.test)[0]:+.2f} mean s* {x.sstar.mean():.1f}")
for rule, x in c.groupby("rule"):
    r = spearmanr(x.inner, x.test); print(f"pooled {rule}: Spearman {r[0]:+.2f} P={r[1]:.2g}")
