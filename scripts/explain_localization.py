#!/usr/bin/env python
"""Dual-stream Grad-CAM / Grad-CAM++ / Score-CAM localisation against radiologist boxes.

Annotation CSV (one row per box, coordinates in ORIGINAL image pixels):
    path,label,x1,y1,x2,y2,reader
Images without pathology (e.g. Normal) must not have boxes: they are reported separately and are
excluded from the mean Pointing Game / IoU (Reviewer 4, comment 9).

Example:
    python scripts/explain_localization.py --run results/covid_robotxdpp/seed42 \
        --config configs/covid_robotxdpp.yaml --annotations annotations/covid_boxes.csv --method gradcam
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from robotxd.data import build_transforms  # noqa: E402
from robotxd.explain import cam_box_iou, dual_stream_cam, pointing_game  # noqa: E402
from robotxd.models import build_model  # noqa: E402
from robotxd.utils import get_device, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++", "scorecam"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-maps", action="store_true")
    args = ap.parse_args()

    import json
    cfg = load_config(args.config)
    meta = json.load(open(os.path.join(args.run, "run.json")))
    names = meta["names"]
    device = get_device()
    model = build_model(cfg, len(names)).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pth"), map_location=device))
    model.eval()
    size = cfg["data"].get("image_size", 224)
    tf = build_transforms(size, train=False, policy="notebook")  # plain resize keeps box geometry simple

    ann = pd.read_csv(args.annotations)
    rows = []
    for path, g in ann.groupby("path"):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        x = tf(img)[None].to(device)
        cls = names.index(g["label"].iloc[0])
        cam = dual_stream_cam(model, x, cls, args.method)
        sx, sy = size / w, size / h
        boxes = [(r.x1 * sx, r.y1 * sy, r.x2 * sx, r.y2 * sy) for r in g.itertuples() if not np.isnan(r.x1)]
        row = {"path": path, "label": g["label"].iloc[0], "n_boxes": len(boxes)}
        if boxes:
            row["pointing_game"] = pointing_game(cam, boxes)
            row["iou"] = cam_box_iou(cam, boxes, args.threshold)
        rows.append(row)
        if args.save_maps:
            os.makedirs(os.path.join(args.run, f"cams_{args.method}"), exist_ok=True)
            np.save(os.path.join(args.run, f"cams_{args.method}", os.path.basename(path) + ".npy"), cam)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.run, f"localization_{args.method}.csv"), index=False)
    scored = df.dropna(subset=["pointing_game"]) if "pointing_game" in df else df.iloc[0:0]
    per_class = scored.groupby("label")[["pointing_game", "iou"]].agg(["mean", "count"])
    print(per_class.round(3))
    if len(scored):
        print(f"\nMean over abnormal images: PG={scored.pointing_game.mean():.3f}  "
              f"IoU={scored.iou.mean():.3f}  (n={len(scored)})")
    print(f"images without boxes (reported separately): {int((df.n_boxes == 0).sum())}")


if __name__ == "__main__":
    main()
