"""Model definitions.

* :class:`ROBOTXDPP` - revised ROBOT-XD++ as described in the manuscript (Sections 3.1-3.7),
  with a *real* fixed PCA basis (training-split mean + top-k eigenvectors, stored as buffers)
  and switchable compression for controlled ablations.
* :class:`ROBOTXDPPNotebook` - exact re-implementation of the model in the original Colab
  notebook (cells 4-6), kept for transparency. Note that its "PCA" is a learned linear layer.
* :class:`EffB3Bottleneck` - EfficientNet-B3 + MLP bottleneck (notebook cell 26), kept as a baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EFF_DIM, DENSE_DIM = 1280, 1024


def _efficientnet_b0(pretrained: bool):
    w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    return models.efficientnet_b0(weights=w).features


def _densenet121(pretrained: bool):
    w = models.DenseNet121_Weights.DEFAULT if pretrained else None
    return models.densenet121(weights=w).features


class ROBOTXDPP(nn.Module):
    """Dual-stream ROBOT-XD++.

    Parameters
    ----------
    num_outputs : number of classes (softmax head) or labels (sigmoid head). The head returns logits;
                  the loss decides between softmax/CE and sigmoid/BCE.
    use_eff, use_dense : enable the EfficientNet-B0 / DenseNet-121 streams.
    compression : ``"pca"``    fixed PCA basis, fitted on training features before training (Eq. 36)
                  ``"random"`` fixed random orthonormal basis of the same size (control)
                  ``"linear"`` learned linear projection of the same size (control)
                  ``"none"``   no compression (fused features go straight to the bottleneck)
    k : compressed dimension.
    """

    def __init__(self, num_outputs: int, use_eff: bool = True, use_dense: bool = True,
                 compression: str = "pca", k: int = 256, proj_dim: int = 512,
                 latent_dim: int = 256, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        assert use_eff or use_dense, "at least one stream is required"
        assert compression in {"pca", "random", "linear", "none"}
        self.use_eff, self.use_dense, self.compression, self.k = use_eff, use_dense, compression, k
        self.pool = nn.AdaptiveAvgPool2d(1)

        n_streams = 0
        if use_eff:
            self.eff_features = _efficientnet_b0(pretrained)
            self.eff_proj = nn.Linear(EFF_DIM, proj_dim)  # Eq. 34
            n_streams += 1
        if use_dense:
            self.dense_features = _densenet121(pretrained)
            self.dense_proj = nn.Linear(DENSE_DIM, proj_dim)
            n_streams += 1

        self.fused_dim = proj_dim * n_streams
        self.fusion = nn.Linear(self.fused_dim, self.fused_dim)  # Eq. 35 (tanh applied in forward)

        if compression in {"pca", "random"}:
            self.register_buffer("proj_mean", torch.zeros(self.fused_dim))
            self.register_buffer("proj_basis", torch.zeros(self.fused_dim, k))
            self.register_buffer("proj_fitted", torch.tensor(False))
            if compression == "random":
                q, _ = torch.linalg.qr(torch.randn(self.fused_dim, k))
                self.set_projection(torch.zeros(self.fused_dim), q)
            comp_out = k
        elif compression == "linear":
            self.proj_linear = nn.Linear(self.fused_dim, k, bias=False)
            comp_out = k
        else:
            comp_out = self.fused_dim

        self.bottleneck = nn.Sequential(  # Eqs. 38-40
            nn.Linear(comp_out, latent_dim), nn.BatchNorm1d(latent_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(latent_dim, latent_dim),
        )
        self.head = nn.Linear(latent_dim, num_outputs)  # Eq. 43 (logits)

    # -- projection --------------------------------------------------------
    @torch.no_grad()
    def set_projection(self, mean: torch.Tensor, basis: torch.Tensor) -> None:
        """Install a fixed basis. ``basis`` has shape (fused_dim, k) with orthonormal columns."""
        assert self.compression in {"pca", "random"}
        self.proj_mean.copy_(mean.to(self.proj_mean))
        self.proj_basis.copy_(basis.to(self.proj_basis))
        self.proj_fitted.fill_(True)

    # -- forward -----------------------------------------------------------
    def stream_features(self, x):
        feats = []
        if self.use_eff:
            e = torch.flatten(self.pool(self.eff_features(x)), 1)
            feats.append(self.eff_proj(e))
        if self.use_dense:
            d = torch.flatten(self.pool(F.relu(self.dense_features(x))), 1)
            feats.append(self.dense_proj(d))
        return torch.cat(feats, 1)

    def fused_features(self, x):
        """F_c in Eq. 35 - the representation the PCA basis is fitted on."""
        return torch.tanh(self.fusion(self.stream_features(x)))

    def compress(self, fc):
        if self.compression in {"pca", "random"}:
            if not bool(self.proj_fitted):
                raise RuntimeError("PCA basis not fitted - call robotxd.pca.fit_fixed_pca(model, loader) first")
            return (fc - self.proj_mean) @ self.proj_basis  # centred projection (corrected Eq. 36)
        if self.compression == "linear":
            return self.proj_linear(fc)
        return fc

    def forward(self, x, return_latent: bool = False):
        latent = self.bottleneck(self.compress(self.fused_features(x)))
        logits = self.head(latent)
        return (logits, latent) if return_latent else logits

    def cam_target_layers(self):
        layers = []
        if self.use_eff:
            layers.append(self.eff_features[-1])
        if self.use_dense:
            layers.append(self.dense_features[-1])
        return layers


class ROBOTXDPPNotebook(nn.Module):
    """Model exactly as in the original Colab notebook (cells 4-6), for transparency.

    Differences from the manuscript: fusion is Linear->BN->GELU->Dropout to 512 (not tanh 1024);
    "PCA" is a *learned* ``nn.Linear(512, 256, bias=False)``; the classifier follows a single
    BN/GELU bottleneck layer.
    """

    def __init__(self, num_outputs: int, use_eff=True, use_dense=True, use_pca=True,
                 use_orth=True, use_align=True, pca_dim=256, latent_dim=256, pretrained=True):
        super().__init__()
        self.use_eff, self.use_dense, self.use_pca = use_eff, use_dense, use_pca
        self.use_orth, self.use_align = use_orth, use_align
        self.pool = nn.AdaptiveAvgPool2d(1)
        if use_eff:
            self.eff_features = _efficientnet_b0(pretrained)
            self.eff_proj = nn.Linear(EFF_DIM, 512)
        if use_dense:
            self.dense_features = _densenet121(pretrained)
            self.dense_proj = nn.Linear(DENSE_DIM, 512)
        fusion_dim = 1024 if (use_eff and use_dense) else 512
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3))
        self.pca_like = nn.Linear(512, pca_dim, bias=False)
        self.bottleneck = nn.Sequential(nn.Linear(pca_dim if use_pca else 512, latent_dim),
                                        nn.BatchNorm1d(latent_dim), nn.GELU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(latent_dim, num_outputs)

    def forward(self, x, return_latent=False):
        feats = []
        if self.use_eff:
            feats.append(self.eff_proj(torch.flatten(self.pool(self.eff_features(x)), 1)))
        if self.use_dense:
            feats.append(self.dense_proj(torch.flatten(self.pool(F.relu(self.dense_features(x))), 1)))
        f = self.fusion(torch.cat(feats, 1) if len(feats) > 1 else feats[0])
        z = self.pca_like(f) if self.use_pca else f
        latent = self.bottleneck(z)
        out = self.classifier(latent)
        return (out, latent) if return_latent else out

    def cam_target_layers(self):
        return [m[-1] for m in (getattr(self, "eff_features", None), getattr(self, "dense_features", None)) if m is not None]


class EffB3Bottleneck(nn.Module):
    """EfficientNet-B3 + two-layer MLP bottleneck + MLP classifier (notebook cell 26)."""

    def __init__(self, num_outputs: int, latent_dim: int = 128, pretrained: bool = True):
        super().__init__()
        w = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.features = models.efficientnet_b3(weights=w).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            nn.Linear(1536, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, latent_dim), nn.BatchNorm1d(latent_dim), nn.GELU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(64, num_outputs))

    def forward(self, x, return_latent=False):
        latent = self.bottleneck(torch.flatten(self.pool(self.features(x)), 1))
        out = self.classifier(latent)
        return (out, latent) if return_latent else out

    def cam_target_layers(self):
        return [self.features[-1]]


class SingleBackbone(nn.Module):
    """Plain ImageNet backbone + linear head, for baselines (densenet121, efficientnet_b0, resnet50, ...)."""

    def __init__(self, name: str, num_outputs: int, pretrained: bool = True):
        super().__init__()
        ctor = getattr(models, name)
        net = ctor(weights="DEFAULT" if pretrained else None)
        if hasattr(net, "classifier"):
            last = net.classifier if isinstance(net.classifier, nn.Linear) else net.classifier[-1]
            in_f = last.in_features
            net.classifier = nn.Linear(in_f, num_outputs)
        elif hasattr(net, "fc"):
            net.fc = nn.Linear(net.fc.in_features, num_outputs)
        elif hasattr(net, "heads"):
            net.heads = nn.Linear(net.heads.head.in_features, num_outputs)
        else:
            raise ValueError(f"cannot replace head of {name}")
        self.net, self.name = net, name

    def forward(self, x, return_latent=False):
        out = self.net(x)
        return (out, out) if return_latent else out

    def cam_target_layers(self):
        n = self.net
        if hasattr(n, "features"):
            return [n.features[-1]]
        if hasattr(n, "layer4"):
            return [n.layer4[-1]]
        raise ValueError("no CAM layer for this backbone")


ARCHS = {"robotxdpp": ROBOTXDPP, "robotxdpp_notebook": ROBOTXDPPNotebook,
         "effb3_bottleneck": EffB3Bottleneck, "backbone": SingleBackbone}


def build_model(cfg: dict, num_outputs: int) -> nn.Module:
    """Build from ``cfg["model"]``. Keys a given architecture does not accept are ignored, so the
    shared defaults in ``_common.yaml`` can be overridden by ``model.arch`` alone."""
    import inspect

    m = dict(cfg.get("model", {}))
    arch = m.pop("arch", "robotxdpp")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}")
    cls = ARCHS[arch]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "num_outputs"}
    kwargs = {k: v for k, v in m.items() if k in accepted}
    return cls(num_outputs=num_outputs, **kwargs)
