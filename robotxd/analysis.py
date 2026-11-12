"""Latent-space diagnostics used in the factorial ablation (Reviewer 4, comments 7 and priority experiment 2)."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def collect_latents(model, loader, device, max_batches=None) -> np.ndarray:
    model.eval()
    zs = []
    for b, (x, *_r) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        _, z = model(x.to(device), return_latent=True)
        zs.append(z.float().cpu())
    return torch.cat(zs).numpy()


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy of normalised singular values) of the centred embedding (Roy & Vetterli, 2007)."""
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def mean_abs_offdiag_corr(z: np.ndarray) -> float:
    c = np.corrcoef(z, rowvar=False)
    c = np.nan_to_num(c)
    d = c.shape[0]
    return float((np.abs(c).sum() - np.abs(np.diag(c)).sum()) / (d * (d - 1)))


def latent_report(z: np.ndarray) -> dict:
    return {
        "effective_rank": effective_rank(z),
        "mean_abs_offdiag_corr": mean_abs_offdiag_corr(z),
        "mean_embedding_norm": float(np.linalg.norm(z, axis=1).mean()),
        "dim": int(z.shape[1]),
    }
