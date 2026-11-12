#!/usr/bin/env python
"""Main effects and interactions for the 2^3 factorial ablation (Reviewer 4, priority experiment 2).

Uses the `factors:` block of the grid config to code each variant, and per_run.csv from summarize.py.

Example:
    python scripts/factorial_analysis.py --grid configs/grid_factorial_transfer.yaml \
        --per-run results/grid_factorial_transfer/tables_target_test/per_run.csv --outcome macro_auroc
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from robotxd.stats import factorial_effects  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--outcome", nargs="+", default=["macro_auroc"])
    args = ap.parse_args()

    g = yaml.safe_load(open(args.grid))
    fac = pd.DataFrame.from_dict(g["factors"], orient="index")
    factors = list(fac.columns)
    df = pd.read_csv(args.per_run).merge(fac, left_on="variant", right_index=True, how="inner")
    missing = set(fac.index) - set(df.variant)
    if missing:
        print(f"WARNING: no runs found for {sorted(missing)}")
    out_dir = os.path.dirname(args.per_run)
    for y in args.outcome:
        if y not in df:
            print(f"skip {y}: not in per_run.csv")
            continue
        res = factorial_effects(df, factors, y)
        path = os.path.join(out_dir, f"factorial_{y}.csv")
        res.to_csv(path, index=False)
        print(f"\n== {y}  (n={len(df)} runs)")
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
