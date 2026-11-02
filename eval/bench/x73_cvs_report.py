"""C2 H-CVS report (pre-registered in manuscript/notes/autonomous_log.md): cross-validated TTT stopping (cv5) vs U3 TTT
and base on random pairs (x68). Per-pair means over folds/draws,
two-sided Wilcoxon, bootstrap 95% CI over pairs.   python x73_cvs_report.py"""
import glob
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
R = Path(__file__).resolve().parents[1] / "results" / "bench"; rng = np.random.default_rng(0)
B, U3, CV = "grnpfn_rel [base]", "grnpfn_rel [TTT]", "grnpfn_rel_cv5 [TTT]"

def dd(a, b):
    d = (a - b.reindex(a.index)).dropna(); bs = [d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), wilcoxon(d).pvalue

rows, cal, prim = [], [], []
for ds in ("norman", "wessels", "sp4_A172", "sp4_T98G"):
    f = R / "x68_ttt" / f"{ds}_grnpfn_rel_cv5.csv"
    if not f.exists():
        continue
    t = pd.concat([pd.read_csv(R / "x68_ttt" / f"{ds}_grnpfn_rel.csv"), pd.read_csv(f)]).drop_duplicates(["fold", "g", "draw", "AB", "method"])
    for g, s in t.groupby("g"):
        pp = lambda m: s[s.method == m].groupby("AB").cos_pair.mean()
        r = dict(screen=ds, g=g, base=pp(B).mean(), U3=pp(U3).mean(), CV=pp(CV).mean())
        for lab, a, b in (("CV-base", CV, B), ("U3-base", U3, B), ("CV-U3", CV, U3)):
            m, lo, hi, p = dd(pp(a), pp(b)); r[lab] = f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] P={p:.2g}"
            if lab == "CV-base":
                prim.append(dict(screen=ds, g=g, d=m, P=p))
        rows.append(r)
    for tag, m in (("", U3), ("_cv5", CV)):
        st = pd.read_csv(R / "x68_ttt" / f"{ds}_grnpfn_rel{tag}_stops.csv"); st["inner"] = st.apply(lambda q: q[f"v{int(q.sstar)}"] - q["v0"], axis=1)
        pv = t.pivot_table(index=["fold", "g", "draw"], columns="method", values="cos_pair", aggfunc="mean"); gain = (pv[m] - pv[B]).rename("test").reset_index()
        x = st.merge(gain, on=["fold", "g", "draw"]); cal.append(dict(screen=ds, rule="CV" if tag else "U3", inner=x.inner.mean(), test=x.test.mean(),
                                                                    abs_gap=np.abs(x.inner - x.test).mean(), sstar=x.sstar.mean(), frac_s0=np.mean(x.sstar == 0)))
pd.set_option("display.width", 250)
print("Random pairs (x68):"); print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nInner vs test gain:"); print(pd.DataFrame(cal).round(3).to_string(index=False))
P = pd.DataFrame(prim); bad = P[(P.d < 0) & (P.P < .05)]
print(f"\nPrimary: significant losses of TTT-CV vs base: {len(bad)} {bad.to_dict('records')}")
for ds, g, u3 in (("norman", .15, None), ("wessels", 1.0, None)):
    q = P[(P.screen == ds) & (P.g == g)]
    if len(q):
        print(f"  {ds} g={g}: TTT-CV − base {q.d.iloc[0]:+.4f} (criterion ≥ −.005)")
print("Secondary: share of U3 gain kept (mean over budgets)")
for ds in ("sp4_A172", "sp4_T98G"):
    rr = [r for r in rows if r["screen"] == ds]
    if rr:
        cv = np.mean([r["CV"] - r["base"] for r in rr]); u3 = np.mean([r["U3"] - r["base"] for r in rr]); print(f"  {ds}: {cv:+.4f} / {u3:+.4f} = {cv / u3:.0%}")
