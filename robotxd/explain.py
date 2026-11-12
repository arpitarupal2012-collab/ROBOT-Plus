"""Dual-stream Grad-CAM (Eq. 49-51) and localisation metrics (Pointing Game, IoU).

Saliency maps are post-hoc, non-causal visual aids. Peak activation inside an annotated region does
not show that a prediction is free of shortcut features.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _cam_class(method: str):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM

    return {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "scorecam": ScoreCAM}[method]


def dual_stream_cam(model, x: torch.Tensor, class_idx: int, method: str = "gradcam") -> np.ndarray:
    """Average of per-stream CAMs after bilinear up-sampling to the input size (Eq. 51).

    ``x`` is a (1, 3, H, W) tensor on the model's device. Returns an (H, W) map in [0, 1].
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    maps = []
    for layer in model.cam_target_layers():
        with _cam_class(method)(model=model, target_layers=[layer]) as cam:
            m = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
        m = torch.from_numpy(m)[None, None]
        maps.append(F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy())
    agg = np.mean(maps, 0)
    return (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)


def pointing_game(cam: np.ndarray, boxes) -> int:
    """1 if the arg-max pixel falls in any box (x1, y1, x2, y2) in CAM pixel coordinates."""
    yy, xx = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return int(any(x1 <= xx <= x2 and y1 <= yy <= y2 for x1, y1, x2, y2 in boxes))


def cam_box_iou(cam: np.ndarray, boxes, threshold: float = 0.5) -> float:
    """IoU between the CAM binarised at ``threshold`` x peak and the union of annotated boxes."""
    pred = cam >= threshold * cam.max()
    gt = np.zeros_like(pred)
    for x1, y1, x2, y2 in boxes:
        gt[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = True
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 0.0
