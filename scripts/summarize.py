#!/usr/bin/env python
"""Single source of truth for every reported table.

Reads saved predictions from ``<root>/<variant>/seed*/pred_<split>.npz`` and writes:
    per_run.csv      one row per (variant, seed): all metrics + diagnostics (+ bootstrap CI if asked)
    summary.csv      mean, SD and t-based 95% CI over seeds for each variant and metric
    paired.csv       paired comparison of every variant with --reference (t-test, exact Wilcoxon, Holm)

Example:
    python scripts/summarize.py --root results/grid_ablation_covid --reference full --split test
    python scripts/summarize.py --root results/transfer_nih_to_chexpert --split target_test --bootstrap 2000
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from robotxd import stats  # noqa: E402
from robotxd.engine import load_pred  # noqa: E402
from robotxd.metrics import compute_metrics, multilabel_metrics, single_label_metrics  # noqa: E402

MAIN_SINGLE = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc"]
MAIN_MULTI = ["macro_auroc", "macro_auprc", "macro_f1"]


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "target_test"])
    ap.add_argument("--reference", default=None)
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples per run (0 = off)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, f"tables_{args.split}")
    os.makedirs(out, exist_ok=True)

    rows, multilabel = [], None
    for run in sorted(glob.glob(os.path.join(args.root, "*", "seed*"))):
        pf = os.path.join(run, f"pred_{args.split}.npz")
        if not os.path.exists(pf):
            continue
        meta = json.load(open(os.path.join(run, "run.json")))
        multilabel, names = meta["multilabel"], meta.get("names")
        pred = load_pred(pf)
        m = compute_metrics(pred, multilabel, names)
        row = {"variant": os.path.basename(os.path.dirname(run)), "seed": int(run.split("seed")[-1]),
               "protocol": meta.get("protocol"), **m}
        dpath = os.path.join(run, "diagnostics.json")
        if os.path.exists(dpath):
            row.update(flatten(json.load(open(dpath))))
        if args.bootstrap:
            idx_csv = os.path.join(run, "target_test_index.csv" if args.split == "target_test" else "test_index.csv")
            groups = None
            if args.split != "val" and os.path.exists(idx_csv):
                pat = pd.read_csv(idx_csv)["patient"]
                groups = pat.to_numpy()[pred["index"]] if pat.notna().all() else None
            key = "macro_auroc"
            if multilabel:
                fn = lambda yt, yp, mk: multilabel_metrics(yt, yp, mk)[key]  # noqa: E731
            else:
                fn = lambda yt, yp, mk: single_label_metrics(yt, yp)[key]  # noqa: E731
            b = stats.bootstrap_ci(fn, pred["y_true"], pred["y_prob"], pred["mask"] if multilabel else None,
                                   groups=groups, n_boot=args.bootstrap)
            row.update({f"{key}_boot_low": b["ci_low"], f"{key}_boot_high": b["ci_high"],
                        "bootstrap_unit": "patient" if groups is not None else "image"})
        rows.append(row)

    if not rows:
        raise SystemExit(f"no pred_{args.split}.npz files under {args.root}")
    per_run = pd.DataFrame(rows).sort_values(["variant", "seed"])
    per_run.to_csv(os.path.join(out, "per_run.csv"), index=False)

    main_metrics = MAIN_MULTI if multilabel else MAIN_SINGLE
    summ = []
    for v, g in per_run.groupby("variant"):
        r = {"variant": v, "n_seeds": len(g)}
        for k in main_metrics:
            c = stats.t_ci(g[k].values)
            r.update({f"{k}_mean": c["mean"], f"{k}_sd": c["sd"], f"{k}_ci_low": c["ci_low"], f"{k}_ci_high": c["ci_high"]})
        summ.append(r)
    pd.DataFrame(summ).to_csv(os.path.join(out, "summary.csv"), index=False)

    if args.reference:
        ref = per_run[per_run.variant == args.reference].set_index("seed")
        prow = []
        for k in main_metrics:
            block = []
            for v, g in per_run.groupby("variant"):
                if v == args.reference:
                    continue
                common = sorted(set(ref.index) & set(g.seed))
                if len(common) < 2:
                    continue
                gg = g.set_index("seed")
                block.append({"metric": k, "reference": args.reference, "variant": v,
                              **stats.paired_compare(ref.loc[common, k], gg.loc[common, k])})
            if block:
                adj = stats.holm([b["p_t"] for b in block])
                for b, a in zip(block, adj):
                    b["p_t_holm"] = a
                prow += block
        pd.DataFrame(prow).to_csv(os.path.join(out, "paired.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(summ)[["variant", "n_seeds"] + [f"{k}_mean" for k in main_metrics]].round(4))
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
