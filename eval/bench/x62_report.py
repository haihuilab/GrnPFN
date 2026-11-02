"""Summarise x62 learning curves + context use across all checkpoints evaluated so far (dev seed 301)."""
import sys, glob
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "results" / "bench" / "x62_pfn_curve"
df = pd.concat([pd.read_csv(f) for f in glob.glob(str(D / "*.csv"))]).drop_duplicates(["dataset", "fold", "g", "draw", "AB", "method"])
m = df.groupby(["dataset", "g", "method"]).cos_pair.mean().unstack("g")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(m.round(3).to_string())
models = sys.argv[1:]
if models:
    base = models[0]; print(f"\nDifferences vs {base} (real context), per g; ctx_use = real - shuffled at g=1")
    rows = []
    for ds in m.index.get_level_values(0).unique():
        for mod in models:
            r = {"dataset": ds, "model": mod}
            for g in m.columns:
                r[f"g={g}"] = m.loc[(ds, mod), g] - m.loc[(ds, base), g]
            r["ctx_use"] = m.loc[(ds, mod), 1.0] - m.loc[(ds, f"{mod} [shuffled ctx]"), 1.0]
            r["slope(.15->1)"] = m.loc[(ds, mod), 1.0] - m.loc[(ds, mod), 0.15]
            rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
