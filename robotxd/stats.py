"""Statistics used for every reported number (Reviewer 2, comment 5; Reviewer 4, comment 8).

* Seed-level intervals use the t-distribution (df = n - 1), not 1.96.
* Bootstrap intervals resample patients (or images when no patient ID exists).
* Paired comparisons report a paired t-test and the exact Wilcoxon test, with Holm correction.
  With n = 5 pairs the smallest attainable two-sided Wilcoxon p-value is 0.0625.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def t_ci(values, conf: float = 0.95) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean, sd = v.mean(), v.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.5 + conf / 2, df=n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return {"mean": mean, "sd": sd, "n": n, "ci_low": mean - half, "ci_high": mean + half, "t_crit": stats.t.ppf(0.5 + conf / 2, n - 1) if n > 1 else np.nan}


def bootstrap_ci(metric_fn, y_true, y_prob, mask=None, groups=None, n_boot: int = 2000,
                 conf: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap of ``metric_fn(y_true, y_prob, mask)``; resamples whole groups (patients)."""
    rng = np.random.default_rng(seed)
    groups = np.arange(len(y_true)) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    point = metric_fn(y_true, y_prob, mask)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        try:
            vals.append(metric_fn(y_true[idx], y_prob[idx], None if mask is None else mask[idx]))
        except ValueError:  # e.g. a class missing from the resample
            continue
    lo, hi = np.percentile(vals, [50 * (1 - conf), 50 * (1 + conf)])
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot_valid": len(vals)}


def paired_compare(a, b) -> dict:
    """a, b: per-seed metric values matched by seed. Positive diff = a better."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ci = t_ci(d)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b, method="exact")
        wp = w.pvalue
    except ValueError:
        wp = np.nan
    sd = d.std(ddof=1)
    return {"mean_diff": d.mean(), "diff_ci_low": ci["ci_low"], "diff_ci_high": ci["ci_high"],
            "t": t.statistic, "df": len(d) - 1, "p_t": t.pvalue, "p_wilcoxon_exact": wp,
            "cohen_dz": d.mean() / sd if sd > 0 else np.nan, "n_pairs": len(d)}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def factorial_effects(df: pd.DataFrame, factors: list[str], outcome: str) -> pd.DataFrame:
    """OLS on seed-level outcomes with all main effects and interactions of 0/1 factors.

    Factors are coded -1/+1 so each coefficient x 2 is the classical factorial effect.
    Returns estimate, SE, 95% CI, t and p for every term.
    """
    X_cols, names = [np.ones(len(df))], ["intercept"]
    coded = {f: 2 * df[f].astype(float).to_numpy() - 1 for f in factors}
    for r in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, r):
            X_cols.append(np.prod([coded[f] for f in combo], axis=0))
            names.append(":".join(combo))
    X = np.column_stack(X_cols)
    y = df[outcome].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    tval = beta / se
    p = 2 * stats.t.sf(np.abs(tval), dof)
    tc = stats.t.ppf(0.975, dof)
    return pd.DataFrame({"term": names, "estimate": beta, "effect(2*beta)": 2 * beta, "se": se,
                         "ci_low": beta - tc * se, "ci_high": beta + tc * se, "t": tval, "p": p, "df": dof})
