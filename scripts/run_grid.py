#!/usr/bin/env python
"""Run every variant of a grid config for every seed (skips finished runs, resumes partial ones).

A grid config has a `base:` config and a `grid:` mapping variant name -> list of dotted overrides:

    base: covid_robotxdpp.yaml
    seeds: [42, 7, 13, 21, 99]
    grid:
      full: []
      no_decorrelation: [loss.decorrelation=none]

Example:  python scripts/run_grid.py --grid configs/grid_ablation_covid.yaml
"""
import argparse
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", help="run only these variants")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every variant, e.g. data.root=/data/nih")
    args = ap.parse_args()

    with open(args.grid) as f:
        g = yaml.safe_load(f)
    base = os.path.join(os.path.dirname(args.grid), g["base"])
    seeds = args.seeds or g.get("seeds", [42, 7, 13, 21, 99])
    prefix = g.get("prefix", os.path.splitext(os.path.basename(args.grid))[0])

    for variant, overrides in g["grid"].items():
        if args.only and variant not in args.only:
            continue
        name = f"{prefix}/{variant}"
        for s in seeds:
            if os.path.exists(os.path.join(args.out, name, f"seed{s}", "DONE")):
                print(f"skip {name} seed{s} (done)")
                continue
            cmd = [sys.executable, os.path.join(HERE, "train.py"), "--config", base, "--seed", str(s),
                   "--name", name, "--out", args.out]
            if overrides or args.set:
                cmd += ["--set", *args.set, *(overrides or [])]
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
