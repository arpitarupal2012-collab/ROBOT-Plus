#!/usr/bin/env python
"""Paper figures, generated ONLY from saved run outputs (predictions, histories, summary tables).

Every figure is written as PNG (600 dpi) and PDF to --out. Nothing is drawn from typed-in numbers.

Sub-commands
------------
roc        ROC curves of one run (single- or multi-label)              --run results/x/seed42
confusion  confusion matrix of one single-label run                     --run results/x/seed42
curves     mean validation curve per variant over seeds                 --root results/grid_x --metric val_macro_f1
bars       variant comparison with t-based 95% CI (from summary.csv)    --summary .../summary.csv --metric macro_f1
transfer   per-label AUROC, mean +- t-CI over seeds, one series/variant --per-run .../per_run.csv
factorial  effect estimates with 95% CI (from factorial_*.csv)          --csv .../factorial_macro_auroc.csv
drift      PCA retained variance: fixed vs refitted basis               --per-run .../per_run.csv
all        everything that the folder tree under --root supports         --root results

Colours: validated categorical palette, assigned in fixed order; series also differ by marker/line
style, and every multi-series figure has a legend (colour is never the only cue).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from robotxd.engine import load_pred  # noqa: E402
from robotxd.stats import t_ci  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "savefig.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".{png,pdf}"))


def run_meta(run):
    return json.load(open(os.path.join(run, "run.json")))


def safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


# ----------------------------------------------------------------------------- per-run figures
def fig_roc(run, split, out):
    meta, pred = run_meta(run), load_pred(os.path.join(run, f"pred_{split}.npz"))
    names, ml = meta["names"], meta["multilabel"]
    y, p = pred["y_true"], pred["y_prob"]
    mask = pred["mask"].astype(bool) if ml else None
    onehot = y if ml else np.eye(p.shape[1])[y]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    idx = list(range(len(names)))
    if len(idx) > 8:  # more than 8 series is unreadable: plot the 8 most prevalent, list the rest in the table
        prev = [(onehot[mask[:, j] if ml else slice(None), j]).mean() for j in idx]
        idx = list(np.argsort(prev)[::-1][:8])
    from sklearn.metrics import roc_auc_score
    for k, j in enumerate(idx):
        m = mask[:, j] if ml else np.ones(len(y), bool)
        if len(np.unique(onehot[m, j])) < 2:
            continue
        fpr, tpr, _ = roc_curve(onehot[m, j], p[m, j])
        auc = roc_auc_score(onehot[m, j], p[m, j])
        ax.plot(fpr, tpr, color=SERIES[k], linestyle=STYLES[k], label=f"{names[j]} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
           title=f"ROC, {split} split")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="Class (AUROC)")
    save(fig, out, f"roc_{safe(os.path.basename(os.path.dirname(run)))}_{os.path.basename(run)}_{split}")


def fig_confusion(run, split, out):
    meta, pred = run_meta(run), load_pred(os.path.join(run, f"pred_{split}.npz"))
    if meta["multilabel"]:
        print("confusion matrix is not defined for multi-label outputs; skipped")
        return
    names = meta["names"]
    cm = confusion_matrix(pred["y_true"], pred["y_prob"].argmax(1), labels=range(len(names)))
    rowpct = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(rowpct, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{100 * rowpct[i, j]:.1f}%", ha="center", va="center", fontsize=7,
                    color="#fcfcfb" if rowpct[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion matrix ({split}; counts and row %)")
    save(fig, out, f"confusion_{safe(os.path.basename(os.path.dirname(run)))}_{os.path.basename(run)}_{split}")


# ----------------------------------------------------------------------------- multi-run figures
def fig_curves(root, metric, out):
    variants = sorted(d for d in os.listdir(root) if glob.glob(os.path.join(root, d, "seed*", "history.csv")))
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, v in enumerate(variants[:8]):
        hs = [pd.read_csv(h) for h in glob.glob(os.path.join(root, v, "seed*", "history.csv"))]
        hs = [h for h in hs if metric in h]
        if not hs:
            continue
        n = min(len(h) for h in hs)
        arr = np.stack([h[metric].to_numpy()[:n] for h in hs])
        ep = np.arange(1, n + 1)
        ax.plot(ep, arr.mean(0), color=SERIES[k], linestyle=STYLES[k], label=f"{v} (n={len(hs)})")
        if len(hs) > 1:
            ax.fill_between(ep, arr.min(0), arr.max(0), color=SERIES[k], alpha=0.12, linewidth=0)
    ax.set(xlabel="Epoch", ylabel=metric.replace("_", " "), title="Validation curves (mean; band = seed range)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, f"curves_{safe(os.path.basename(root.rstrip('/')))}_{metric}")


def fig_bars(summary_csv, metric, out):
    s = pd.read_csv(summary_csv)
    m, lo, hi = f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"
    if m not in s:
        print(f"{metric} not in {summary_csv}")
        return
    s = s.sort_values(m)
    fig, ax = plt.subplots(figsize=(5.2, 0.38 * len(s) + 0.9))
    y = np.arange(len(s))
    ax.barh(y, s[m], color=SERIES[0], height=0.6)
    ax.errorbar(s[m], y, xerr=[s[m] - s[lo], s[hi] - s[m]], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    for yi, (v, h) in enumerate(zip(s[m], s[hi])):
        ax.text(h, yi, f"  {v:.4f}", va="center", fontsize=7, color=INK)
    lo_x = max(0, (s[lo].min()) - 0.02)
    ax.set_xlim(lo_x, min(1.0, s[hi].max() + 0.03))
    ax.set_yticks(y, s["variant"])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=f"{metric.replace('_', ' ')} (mean, t-based 95% CI over seeds)", title="Variant comparison")
    save(fig, out, f"bars_{safe(os.path.basename(os.path.dirname(os.path.dirname(summary_csv))))}_{metric}")


def fig_transfer(per_run_csv, out, variants=None):
    df = pd.read_csv(per_run_csv)
    cols = [c for c in df.columns if c.startswith("auroc/")]
    if not cols:
        return
    labels = [c.split("/", 1)[1] for c in cols]
    vs = variants or sorted(df.variant.unique())[:8]
    fig, ax = plt.subplots(figsize=(max(5, 0.75 * len(labels) + 1.5), 3.4))
    w = 0.7 / max(len(vs), 1)
    x = np.arange(len(labels))
    lows = []
    for k, v in enumerate(vs):
        g = df[df.variant == v]
        means, lo, hi = [], [], []
        for c in cols:
            vals = g[c].dropna().values
            ci = t_ci(vals) if len(vals) > 1 else {"mean": vals.mean() if len(vals) else np.nan,
                                                  "ci_low": np.nan, "ci_high": np.nan}
            means.append(ci["mean"])
            lo.append(ci["mean"] - ci["ci_low"] if not np.isnan(ci["ci_low"]) else 0)
            hi.append(ci["ci_high"] - ci["mean"] if not np.isnan(ci["ci_high"]) else 0)
        means = np.array(means)
        lows.append(np.nanmin(means - np.array(lo)))
        ax.errorbar(x + (k - (len(vs) - 1) / 2) * w, means, yerr=[lo, hi], fmt=MARKERS[k], color=SERIES[k],
                    ecolor=SERIES[k], elinewidth=1.2, capsize=2, markersize=6, label=f"{v} (n={len(g)})")
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")  # chance level
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(max(0.0, min(0.45, min(lows) - 0.05)), 1.0)
    ax.grid(axis="x", visible=False)
    split = os.path.basename(os.path.dirname(per_run_csv)).replace("tables_", "").replace("_", " ")
    ax.set(ylabel="AUROC (mean, t-based 95% CI)")
    ax.set_title(f"Per-class AUROC, {split} split (dotted line = chance)", pad=26)
    ax.legend(frameon=False, ncol=min(len(vs), 3), loc="lower left", bbox_to_anchor=(0, 1.02))
    save(fig, out, f"transfer_auroc_{safe(os.path.basename(os.path.dirname(os.path.dirname(per_run_csv))))}")


def fig_factorial(csv, out):
    r = pd.read_csv(csv)
    r = r[r.term != "intercept"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.0, 0.36 * len(r) + 0.9))
    y = np.arange(len(r))
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.errorbar(r["estimate"], y, xerr=[r["estimate"] - r["ci_low"], r["ci_high"] - r["estimate"]],
                fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=1.5, capsize=2, markersize=5)
    xtext = r["ci_high"].max()
    for yi, p in enumerate(r["p"]):
        ax.text(xtext, yi, f"   p={p:.3f}", va="center", fontsize=7)
    ax.set_yticks(y, r["term"])
    ax.grid(axis="y", visible=False)
    outcome = os.path.basename(csv).replace("factorial_", "").replace(".csv", "")
    ax.set(xlabel=f"coefficient on {outcome} (factors coded -1/+1; 95% CI)", title="2$^3$ factorial: main effects and interactions")
    save(fig, out, f"factorial_{safe(outcome)}")


def fig_drift(per_run_csv, out):
    df = pd.read_csv(per_run_csv)
    a, b = "pca_drift.retained_variance_fixed_basis", "pca_drift.retained_variance_refitted_basis"
    if a not in df:
        return
    g = df.dropna(subset=[a]).groupby("variant")[[a, b]].mean()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(g) + 1.5), 3.2))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g[a], width=0.38, color=SERIES[0], label="fixed basis (fitted before training)")
    ax.bar(x + 0.2, g[b], width=0.38, color=SERIES[1], label="basis refitted at end of training", hatch="//",
           edgecolor="#fcfcfb")
    ax.set_xticks(x, g.index, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="retained variance (final features)")
    ax.set_title("PCA basis drift", pad=34)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    save(fig, out, f"pca_drift_{safe(os.path.basename(os.path.dirname(os.path.dirname(per_run_csv))))}")


def fig_all(root, out):
    """Walk ``root`` and draw every figure the saved outputs support."""
    for run in sorted(glob.glob(os.path.join(root, "**", "seed*"), recursive=True)):
        if not os.path.exists(os.path.join(run, "DONE")):
            continue
        # one representative seed per variant for per-run figures
        if sorted(glob.glob(os.path.join(os.path.dirname(run), "seed*")))[0] != run:
            continue
        sub = os.path.join(out, os.path.relpath(os.path.dirname(run), root))
        for split in ("test", "target_test"):
            if os.path.exists(os.path.join(run, f"pred_{split}.npz")):
                fig_roc(run, split, sub)
                fig_confusion(run, split, sub)
    for parent in sorted({os.path.dirname(os.path.dirname(os.path.dirname(h))) for h in
                          glob.glob(os.path.join(root, "**", "seed*", "history.csv"), recursive=True)}):
        sub = os.path.join(out, os.path.relpath(parent, root))
        for metric in ("val_macro_f1", "val_macro_auroc"):
            fig_curves(parent, metric, sub)
    for summ in glob.glob(os.path.join(root, "**", "tables_*", "summary.csv"), recursive=True):
        sub = os.path.join(out, os.path.relpath(os.path.dirname(summ), root))
        for metric in ("macro_f1", "macro_auroc"):
            fig_bars(summ, metric, sub)
        pr = os.path.join(os.path.dirname(summ), "per_run.csv")
        fig_transfer(pr, sub)
        fig_drift(pr, sub)
        for fcsv in glob.glob(os.path.join(os.path.dirname(summ), "factorial_*.csv")):
            fig_factorial(fcsv, sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["roc", "confusion", "curves", "bars", "transfer", "factorial", "drift", "all"])
    ap.add_argument("--run")
    ap.add_argument("--root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--summary")
    ap.add_argument("--per-run")
    ap.add_argument("--csv")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    {"roc": lambda: fig_roc(a.run, a.split, a.out),
     "confusion": lambda: fig_confusion(a.run, a.split, a.out),
     "curves": lambda: fig_curves(a.root, a.metric, a.out),
     "bars": lambda: fig_bars(a.summary, a.metric, a.out),
     "transfer": lambda: fig_transfer(a.per_run, a.out, a.variants),
     "factorial": lambda: fig_factorial(a.csv, a.out),
     "drift": lambda: fig_drift(a.per_run, a.out),
     "all": lambda: fig_all(a.root, a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
