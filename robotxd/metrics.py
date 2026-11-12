"""Metrics for single-label (softmax) and multi-label (sigmoid) predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def single_label_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names=None) -> dict:
    """Macro metrics from class probabilities (argmax decision). Per-class values included."""
    y_pred = y_prob.argmax(1)
    k = y_prob.shape[1]
    onehot = np.eye(k)[y_true]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_auroc": roc_auc_score(onehot, y_prob, average="macro"),
        "micro_auroc": roc_auc_score(onehot, y_prob, average="micro"),
    }
    names = class_names or [str(i) for i in range(k)]
    p = precision_score(y_true, y_pred, average=None, labels=range(k), zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=range(k), zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=range(k), zero_division=0)
    for i, n in enumerate(names):
        out[f"precision/{n}"], out[f"recall/{n}"], out[f"f1/{n}"] = p[i], r[i], f[i]
        out[f"auroc/{n}"] = roc_auc_score(onehot[:, i], y_prob[:, i])
    return {k_: float(v) for k_, v in out.items()}


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray | None = None,
                       label_names=None, thresholds: np.ndarray | None = None) -> dict:
    """Per-label AUROC/AUPRC (masked entries ignored), macro averages, and F1 at given thresholds
    (default 0.5; pass validation-selected thresholds to avoid tuning on test)."""
    n_lab = y_true.shape[1]
    mask = np.ones_like(y_true, dtype=bool) if mask is None else mask.astype(bool)
    names = label_names or [str(i) for i in range(n_lab)]
    thr = np.full(n_lab, 0.5) if thresholds is None else np.asarray(thresholds)
    out, aurocs, auprcs, f1s = {}, [], [], []
    for j, n in enumerate(names):
        m = mask[:, j]
        yt, yp = y_true[m, j], y_prob[m, j]
        if len(np.unique(yt)) < 2:
            continue
        a, p = roc_auc_score(yt, yp), average_precision_score(yt, yp)
        f = f1_score(yt, (yp >= thr[j]).astype(int), zero_division=0)
        out[f"auroc/{n}"], out[f"auprc/{n}"], out[f"f1/{n}"] = a, p, f
        out[f"prevalence/{n}"] = float(yt.mean())
        aurocs.append(a), auprcs.append(p), f1s.append(f)
    out["macro_auroc"] = float(np.mean(aurocs))
    out["macro_auprc"] = float(np.mean(auprcs))
    out["macro_f1"] = float(np.mean(f1s))
    out["n_labels_scored"] = len(aurocs)
    return {k: float(v) for k, v in out.items()}


def select_thresholds(y_true, y_prob, mask=None, grid=np.linspace(0.05, 0.95, 91)) -> np.ndarray:
    """Per-label max-F1 thresholds. Call on the VALIDATION split only."""
    mask = np.ones_like(y_true, dtype=bool) if mask is None else mask.astype(bool)
    thr = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        m = mask[:, j]
        if len(np.unique(y_true[m, j])) < 2:
            continue
        scores = [f1_score(y_true[m, j], (y_prob[m, j] >= t).astype(int), zero_division=0) for t in grid]
        thr[j] = grid[int(np.argmax(scores))]
    return thr


def compute_metrics(pred: dict, multilabel: bool, names=None, thresholds=None) -> dict:
    """Dispatch on a saved prediction dict (see ``engine.predict``)."""
    if multilabel:
        return multilabel_metrics(pred["y_true"], pred["y_prob"], pred.get("mask"), names, thresholds)
    return single_label_metrics(pred["y_true"], pred["y_prob"], names)
