"""Regularisation and domain-alignment losses.

Notation: ``z`` is a (B, d) batch of latent embeddings; ``zs`` / ``zt`` are source / target batches.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- decorrelation
def batch_gram_penalty(z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """The original "orthogonality" term ||L^T L - I||_F^2 (Eq. 41), as in the notebook.

    With B < d the d x d matrix L^T L has rank <= B, so
        ||L^T L - I_d||_F^2 = ||L L^T - I_B||_F^2 + (d - B).
    The term therefore has a floor of (d - B) and its gradient acts on *sample* similarity within
    the batch, not on feature correlation. Kept only as a comparison arm.
    """
    if normalize:
        z = F.normalize(z, dim=1)
    gram = z.T @ z
    eye = torch.eye(gram.size(0), device=z.device, dtype=z.dtype)
    return torch.linalg.matrix_norm(gram - eye) ** 2


def covariance_decorrelation(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Rank-aware feature decorrelation: mean squared off-diagonal correlation of standardised features.

    L_dec = sum_{i != j} C_ij^2 / (d (d - 1)),   C = Zs^T Zs / (B - 1),   Zs = (Z - mean) / std.
    Scale-invariant, so it cannot be reduced by shrinking the embedding.
    """
    b, d = z.shape
    if b < 2:
        return z.new_zeros(())
    zs = (z - z.mean(0)) / (z.std(0, unbiased=True) + eps)
    c = zs.T @ zs / (b - 1)
    off = c - torch.diag(torch.diagonal(c))
    return (off ** 2).sum() / (d * (d - 1))


def latent_l2(z: torch.Tensor) -> torch.Tensor:
    """gamma ||L||^2 term of Eq. 45 (mean over the batch)."""
    return (z ** 2).sum(1).mean()


# ----------------------------------------------------------------------------- alignment
def centroid_alignment(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """First-order alignment ||mu_s - mu_t||^2 (Eq. 42) between a source and a *target* batch."""
    return ((zs.mean(0) - zt.mean(0)) ** 2).sum()


def coral(zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    """Deep CORAL (Sun & Saenko, 2016): squared Frobenius distance between covariances / (4 d^2)."""
    d = zs.size(1)

    def cov(x):
        x = x - x.mean(0, keepdim=True)
        return x.T @ x / (x.size(0) - 1)

    return torch.linalg.matrix_norm(cov(zs) - cov(zt)) ** 2 / (4 * d * d)


def mmd_rbf(zs: torch.Tensor, zt: torch.Tensor, scales=(0.25, 0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    """Biased multi-kernel Gaussian MMD^2 with median-heuristic bandwidth (Gretton et al., 2012)."""
    x = torch.cat([zs, zt], 0)
    d2 = torch.cdist(x, x).pow(2)
    med = d2.detach()[d2.detach() > 0].median()
    k = sum(torch.exp(-d2 / (s * med + 1e-8)) for s in scales)
    n = zs.size(0)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


ALIGNMENT = {"centroid": centroid_alignment, "coral": coral, "mmd": mmd_rbf}
DECORRELATION = {"covariance": covariance_decorrelation, "gram": batch_gram_penalty}


def notebook_alignment_placeholder(z: torch.Tensor) -> torch.Tensor:
    """What the original notebook computed as "alignment": the centroid distance between the two
    halves of the *same source batch*. No target data is involved. Provided only so the notebook
    results can be reproduced; do not describe it as domain alignment."""
    if z.size(0) < 4:
        return z.new_zeros(())
    a, b = torch.chunk(z, 2, 0)
    return ((a.mean(0) - b.mean(0)) ** 2).sum()


# ----------------------------------------------------------------------------- classification
def classification_loss(logits, target, mask, multilabel: bool, class_weights=None, label_smoothing=0.0):
    """Softmax cross-entropy (single-label) or masked sigmoid BCE (multi-label)."""
    if multilabel:
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        return (bce * mask).sum() / mask.sum().clamp_min(1.0)
    return F.cross_entropy(logits, target, weight=class_weights, label_smoothing=label_smoothing)
