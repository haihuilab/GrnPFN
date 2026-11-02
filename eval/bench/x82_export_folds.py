"""R1-C (pre-registered 2026-10-28): write the benchmark's pair folds of the other screens for the external tools.
data/bench_cache/{ds}_folds.json = {fold: {train_combos, test_combos, test_A, test_B}} from benchmark.outer_folds,
exactly the folds that real_folds() uses. Run in the main env:  ../../.venv/bin/python x82_export_folds.py"""
import json

import anndata as ad
import numpy as np
import pandas as pd

from benchmark import outer_folds
from common import DATA

for ds in ("wessels", "sp4_A172", "sp4_T98G", "sp4_U87MG"):
    pert = ad.read_h5ad(DATA / "bench_cache" / f"{ds}_hvg3000.h5ad", backed="r").obs["pert"].astype(str).to_numpy()
    rows_of = pd.Series(np.arange(len(pert))).groupby(pert).apply(np.asarray).to_dict()
    out = {}
    for k, (tr, te) in enumerate(outer_folds(ds, rows_of, False)):
        out[k] = dict(train_combos=["+".join(t) for t in tr], test_combos=["+".join(t) for t in te],
                      test_A=[t[0] for t in te], test_B=[t[1] for t in te])
        assert all(c in rows_of for c in out[k]["test_combos"] + out[k]["train_combos"])
    (DATA / "bench_cache" / f"{ds}_folds.json").write_text(json.dumps(out, indent=1))
    print(ds, [len(v["test_combos"]) for v in out.values()], sum(len(v["test_combos"]) for v in out.values()))
