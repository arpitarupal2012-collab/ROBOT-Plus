"""Parameter count, FLOPs (via thop, MACs x 2) and inference latency."""
from __future__ import annotations

import copy
import time

import torch


def count_params(model, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def latency_ms(model, device, image_size: int = 224, batch: int = 1, warmup: int = 50, iters: int = 500) -> float:
    model = model.to(device).eval()
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def complexity_report(model, device, image_size: int = 224, iters: int = 500) -> dict:
    from thop import profile

    m = copy.deepcopy(model).to(device).eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    macs, _ = profile(m, inputs=(x,), verbose=False)
    return {"params_M": count_params(model) / 1e6, "GMACs": macs / 1e9, "GFLOPs(2xMACs)": 2 * macs / 1e9,
            "latency_ms": latency_ms(model, device, image_size, iters=iters), "device": str(device)}
