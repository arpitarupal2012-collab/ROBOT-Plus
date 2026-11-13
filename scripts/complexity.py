#!/usr/bin/env python
"""Parameters, FLOPs and latency for a config (no data needed).

Example: python scripts/complexity.py --config configs/covid_robotxdpp.yaml --num-outputs 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from robotxd.complexity import complexity_report  # noqa: E402
from robotxd.models import build_model  # noqa: E402
from robotxd.utils import get_device, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-outputs", type=int, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained"] = False
    model = build_model(cfg, args.num_outputs)
    if getattr(model, "compression", None) == "pca":  # dummy basis so the forward pass runs
        q, _ = torch.linalg.qr(torch.randn(model.fused_dim, model.k))
        model.set_projection(torch.zeros(model.fused_dim), q)
    print(complexity_report(model, get_device(args.device), cfg.get("data", {}).get("image_size", 224), args.iters))


if __name__ == "__main__":
    main()
