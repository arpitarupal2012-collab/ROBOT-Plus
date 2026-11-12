"""ROBOT-XD++: dimensionality-constrained dual-stream chest X-ray classification.

Modules
-------
data        datasets, label handling and leakage-safe splits
models      ROBOT-XD++ (revised), the original notebook model, and baselines
losses      decorrelation and domain-alignment losses
pca         fixed (training-only) PCA basis and drift diagnostics
analysis    latent-space diagnostics (effective rank, feature correlation)
engine      training loop and prediction
metrics     single-label and multi-label metrics
stats       t-based CIs, bootstrap CIs, paired tests, Holm, factorial model
explain     dual-stream Grad-CAM and localisation metrics
complexity  parameters, FLOPs and latency
"""

__version__ = "1.0.0"
