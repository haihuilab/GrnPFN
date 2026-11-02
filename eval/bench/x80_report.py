"""C6 report, as pre-registered (manuscript/notes/autonomous_log.md, 2026-10-27; written before any result).
GrnPFN rows: fresh seeds from x79_confirm (curve, design); dev seed 301 from x79_confirm/smoke_dev301/curve_s301.csv and
x60_pfn_design (x64 vdesign at g = .25). Baselines from x80_baselines.
  python x80_report.py --seeds 431 439 449 457     (primary)        python x80_report.py --seeds 301   (development)"""
import argparse
import glob

import pandas as pd

from common import RESULTS
from x79_report import DS, contrast

B = RESULTS / "x80_baselines"; C = RESULTS / "x79_confirm"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, default=[431, 439, 449, 457]); a = ap.parse_args()
    dev = a.seeds == [301]
    pfn = pd.read_csv(C / "smoke_dev301" / "curve_s301.csv") if dev else pd.concat([pd.read_csv(C / f"curve_s{s}.csv") for s in a.seeds])
    pfn = pfn[pfn.method.isin(["PFN", "PFN [empty]"])].assign(method=lambda x: x.method.str.replace("PFN", "GrnPFN"))
    base = pd.concat([pd.read_csv(f) for s in a.seeds for f in glob.glob(str(B / f"curve_*_s{s}.csv"))])
    cur = pd.concat([pfn, base]); print(f"# C6 standard baselines, seeds {a.seeds}{' (DEVELOPMENT)' if dev else ' (pre-registered primary)'}\n")
    print(cur.groupby(["dataset", "method", "g"]).cos_pair.mean().unstack("g").round(3).to_string(), "\n")
    contrast(cur, ("GrnPFN", None, 0.15), ("POLY", None, 0.15), "cos_pair", "PRIMARY B1: GrnPFN@.15 − POLY@.15 (benchmark centring)", rule=not dev)
    print("\nSecondary:")
    for g in (0.15, 0.25, 0.4, 1.0):
        for m in ("POLY", "RIDGE", "RIDGE-SUM"):
            for col in ("cos_pair", "cos_pair_testc"):
                if g == 0.15 and m == "POLY" and col == "cos_pair":
                    continue
                contrast(cur, ("GrnPFN", None, g), (m, None, g), col, f"GrnPFN@{g} − {m}@{g}")
    contrast(cur, ("GrnPFN [empty]", None, None), ("POLY", None, 0.15), "cos_pair_testc", "GrnPFN(0 pairs) − POLY@.15")
    des = [pd.read_csv(f) for s in a.seeds for f in glob.glob(str(B / f"design_*_s{s}.csv"))]
    if des:
        d = pd.concat(des)
        if dev:
            v = pd.concat([pd.read_csv(RESULTS / "x60_pfn_design" / f"{ds}_s301_grnpfn_rel_var.csv") for ds, _ in DS])
            v = v[(v.g == 0.25) & (v.scheme == "pfn-vdesign")].assign(method="GrnPFN")
        else:
            v = pd.concat([pd.read_csv(f) for s in a.seeds for f in glob.glob(str(C / f"design_*_s{s}.csv"))])
            v = v[(v.scheme == "pfn-vdesign") & (v.method == "PFN")].assign(method="GrnPFN")
        d = pd.concat([d, v]); print(f"\nDesign at g = .25 (screens: {sorted(d[d.method == 'POLY'].dataset.unique())}):")
        contrast(d, ("POLY", "pfn-vdesign", None), ("POLY", "random", None), "cos_pair", "POLY + GrnPFN design − POLY + random")
        contrast(d, ("GrnPFN", "pfn-vdesign", None), ("POLY", "poly-design", None), "cos_pair", "GrnPFN + GrnPFN design − POLY + POLY-design")
        contrast(d, ("POLY", "pfn-vdesign", None), ("POLY", "poly-design", None), "cos_pair", "POLY + GrnPFN design − POLY + POLY-design")
        contrast(d, ("RIDGE", "pfn-vdesign", None), ("RIDGE", "random", None), "cos_pair", "RIDGE + GrnPFN design − RIDGE + random")


if __name__ == "__main__":
    main()
