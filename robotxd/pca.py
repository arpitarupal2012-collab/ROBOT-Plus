"""Fixed PCA basis fitted on training-split features only, plus drift diagnostics (Reviewer 4, comment 6)."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def collect_fused_features(model, loader, device, max_batches: int | None = None) -> torch.Tensor:
    """Fused features F_c for every image in ``loader`` (eval mode, no augmentation expected)."""
    model.eval()
    out = []
    for b, (x, *_rest) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        out.append(model.fused_features(x.to(device)).float().cpu())
    return torch.cat(out)


def pca_basis(features: torch.Tensor, k: int):
    """Mean, top-k eigenvectors (columns) and eigenvalues of the empirical covariance."""
    f = features.double()
    mean = f.mean(0)
    cov = (f - mean).T @ (f - mean) / (f.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
    return mean.float(), evecs[:, :k].float().contiguous(), evals.float()


def fit_fixed_pca(model, train_eval_loader, device, k: int | None = None, max_batches=None) -> dict:
    """Fit the basis on *training* features (loader must iterate the training split with eval
    transforms) and install it in the model. Returns diagnostics to log."""
    k = k or model.k
    feats = collect_fused_features(model, train_eval_loader, device, max_batches)
    mean, basis, evals = pca_basis(feats, k)
    model.set_projection(mean, basis)
    retained = float(evals[:k].sum() / evals.sum())
    print(f"PCA fitted on {len(feats)} training images: k={k}, retained variance={retained:.4f}")
    return {"n_fit": len(feats), "k": k, "retained_variance_at_fit": retained,
            "eigenvalues": evals.numpy().tolist()}


def retained_variance(features: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of the total variance of ``features`` captured by a *given* (possibly stale) basis."""
    f = features.double() - features.double().mean(0)
    total = (f ** 2).sum()
    proj = f @ basis.double()
    return float((proj ** 2).sum() / total)


def principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor) -> np.ndarray:
    """Principal angles (degrees) between two subspaces with orthonormal column bases."""
    s = torch.linalg.svdvals(basis_a.double().T @ basis_b.double()).clamp(-1, 1)
    return np.degrees(np.arccos(s.numpy()))


@torch.no_grad()
def basis_drift_report(model, train_eval_loader, device, max_batches=None) -> dict:
    """End-of-training check: how much variance does the fixed basis still capture, compared with a
    basis refitted on the final representation, and how far apart are the two subspaces?"""
    feats = collect_fused_features(model, train_eval_loader, device, max_batches)
    k = model.proj_basis.shape[1]
    _, new_basis, evals = pca_basis(feats, k)
    ang = principal_angles(model.proj_basis.cpu(), new_basis)
    return {
        "retained_variance_fixed_basis": retained_variance(feats, model.proj_mean.cpu(), model.proj_basis.cpu()),
        "retained_variance_refitted_basis": float(evals[:k].sum() / evals.sum()),
        "principal_angle_mean_deg": float(ang.mean()),
        "principal_angle_median_deg": float(np.median(ang)),
        "principal_angle_max_deg": float(ang.max()),
    }
