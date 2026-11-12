#!/usr/bin/env python
"""Prepare the RSNA Pneumonia Detection Challenge data.

1. Converts ``stage_2_train_images/*.dcm`` to 8-bit PNG in ``<root>/png/`` (original 1024x1024 size).
2. Writes ``annotations/rsna_boxes.csv`` in the format used by ``scripts/explain_localization.py``:
       path,label,x1,y1,x2,y2,reader
   One row per expert box for "Lung Opacity" images (coordinates in original pixels).
   Images of the other two classes get one row without coordinates; they are reported separately
   and never enter the Pointing Game / IoU means (Reviewer 4, comment 9).

These boxes come from the RSNA/STR expert annotation (Shih et al., Radiology: AI 2019), so the
localisation study no longer depends on unreported in-house annotations.

Example:
    python scripts/prepare_rsna.py --root /data/rsna-pneumonia-detection-challenge --workers 8
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def _convert(args):
    src, dst = args
    if os.path.exists(dst):
        return dst
    import pydicom
    from PIL import Image

    ds = pydicom.dcmread(src)
    a = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255
    Image.fromarray(a.astype(np.uint8)).save(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-annotations", default="annotations/rsna_boxes.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    dcm_dir = os.path.join(args.root, "stage_2_train_images")
    png_dir = os.path.join(args.root, "png")
    os.makedirs(png_dir, exist_ok=True)
    if not args.skip_convert:
        jobs = [(os.path.join(dcm_dir, f), os.path.join(png_dir, f[:-4] + ".png"))
                for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
        print(f"converting {len(jobs)} DICOM files ...")
        with ProcessPoolExecutor(args.workers) as ex:
            for i, _ in enumerate(ex.map(_convert, jobs, chunksize=64)):
                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1}/{len(jobs)}")

    boxes = pd.read_csv(os.path.join(args.root, "stage_2_train_labels.csv"))
    info = pd.read_csv(os.path.join(args.root, "stage_2_detailed_class_info.csv")).drop_duplicates("patientId")
    df = boxes.merge(info, on="patientId")
    df["path"] = df["patientId"].map(lambda p: os.path.join(png_dir, f"{p}.png"))
    df["x1"], df["y1"] = df["x"], df["y"]
    df["x2"], df["y2"] = df["x"] + df["width"], df["y"] + df["height"]
    df["reader"] = "rsna_consensus"
    out = df.rename(columns={"class": "label"})[["path", "label", "x1", "y1", "x2", "y2", "reader"]]
    os.makedirs(os.path.dirname(args.out_annotations) or ".", exist_ok=True)
    out.to_csv(args.out_annotations, index=False)
    n_img = out.dropna(subset=["x1"]).path.nunique()
    print(f"wrote {len(out)} rows ({n_img} images with boxes) to {args.out_annotations}")
    print(out.drop_duplicates("path").label.value_counts().to_string())


if __name__ == "__main__":
    main()
