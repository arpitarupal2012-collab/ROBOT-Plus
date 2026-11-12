"""Fast unit tests (CPU, no downloads). Run: pytest -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robotxd import losses, pca, stats  # noqa: E402
from robotxd.analysis import effective_rank, mean_abs_offdiag_corr  # noqa: E402
from robotxd.data import split_dataframe  # noqa: E402
from robotxd.metrics import multilabel_metrics, single_label_metrics  # noqa: E402
from robotxd.models import ROBOTXDPP, build_model  # noqa: E402


def test_gram_penalty_floor_equals_d_minus_b():
    """||L^T L - I_d||^2 = ||L L^T - I_B||^2 + (d - B): the original term cannot go below d - B."""
    b, d = 32, 256
    q, _ = torch.linalg.qr(torch.randn(d, b))
    z = q.T  # rows orthonormal -> L L^T = I_B, the best case
    assert torch.isclose(losses.batch_gram_penalty(z, normalize=False), torch.tensor(float(d - b)), atol=1e-3)


def test_covariance_decorrelation_is_scale_invariant_and_zero_for_whitened():
    z = torch.randn(512, 16)
    assert torch.isclose(losses.covariance_decorrelation(z), losses.covariance_decorrelation(10 * z), atol=1e-6)
    zw = torch.linalg.qr(torch.randn(512, 16) - 0)[0] * 22.6
    zw = zw - zw.mean(0)
    assert losses.covariance_decorrelation(zw) < 1e-3


def test_alignment_losses_zero_for_identical_batches():
    z = torch.randn(64, 8)
    for fn in losses.ALIGNMENT.values():
        assert float(fn(z, z.clone())) < 1e-5


def test_t_ci_uses_t_quantile():
    c = stats.t_ci([1, 2, 3, 4, 5])
    assert abs(c["t_crit"] - 2.776) < 1e-3


def test_holm():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_wilcoxon_floor_with_five_pairs():
    r = stats.paired_compare([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])
    assert r["p_wilcoxon_exact"] == pytest.approx(0.0625)


def test_factorial_recovers_effects():
    rng = np.random.default_rng(0)
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for s in range(5):
                rows.append({"a": a, "b": b, "y": 1 + 0.5 * a + 0.2 * a * b + rng.normal(0, 0.01)})
    res = stats.factorial_effects(pd.DataFrame(rows), ["a", "b"], "y").set_index("term")
    assert res.loc["a", "effect(2*beta)"] == pytest.approx(0.6, abs=0.02)  # 0.5 + half the interaction
    assert res.loc["a:b", "effect(2*beta)"] == pytest.approx(0.1, abs=0.02)


def test_patient_disjoint_split():
    df = pd.DataFrame({"path": [f"img{i}" for i in range(600)], "patient": [f"p{i // 3}" for i in range(600)]})
    tr, va, te = split_dataframe(df, seed=1, group_col="patient")
    assert not (set(tr.patient) & set(te.patient)) and not (set(tr.patient) & set(va.patient))
    assert len(tr) + len(va) + len(te) == 600


def test_pca_basis_and_drift():
    f = torch.randn(400, 20) @ torch.randn(20, 20)
    mean, basis, ev = pca.pca_basis(f, 5)
    assert torch.allclose(basis.T @ basis, torch.eye(5), atol=1e-4)
    rv = pca.retained_variance(f, mean, basis)
    assert rv == pytest.approx(float(ev[:5].sum() / ev.sum()), rel=1e-4)
    assert np.allclose(pca.principal_angles(basis, basis), 0, atol=0.1)  # degrees; float32 basis


def test_model_forward_all_compressions():
    x = torch.randn(4, 3, 64, 64)
    for comp in ("pca", "random", "linear", "none"):
        m = ROBOTXDPP(num_outputs=3, compression=comp, k=16, pretrained=False).eval()
        if comp == "pca":
            with pytest.raises(RuntimeError):
                m(x)
            q, _ = torch.linalg.qr(torch.randn(m.fused_dim, 16))
            m.set_projection(torch.zeros(m.fused_dim), q)
        logits, z = m(x, return_latent=True)
        assert logits.shape == (4, 3) and z.shape == (4, 256)


def test_build_model_ignores_irrelevant_keys():
    cfg = {"model": {"arch": "backbone", "name": "resnet18", "use_eff": True, "k": 256, "pretrained": False}}
    assert build_model(cfg, 5)(torch.randn(2, 3, 64, 64)).shape == (2, 5)


def test_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[y] * 0.8 + 0.2 / 3
    assert single_label_metrics(y, p)["macro_f1"] == 1.0
    yt = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    mask = np.array([[1, 1], [1, 1], [1, 0], [1, 1]])
    m = multilabel_metrics(yt, yt * 0.9 + 0.05, mask)
    assert m["macro_auroc"] == 1.0


def test_latent_diagnostics():
    z = np.random.default_rng(0).normal(size=(500, 10))
    assert 8 < effective_rank(z) <= 10
    assert mean_abs_offdiag_corr(z) < 0.1
