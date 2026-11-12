#!/usr/bin/env python
"""Find target images that also occur in the source dataset (perceptual hashing).

Why: the COVID-19 Radiography Database takes its Normal and Lung Opacity images from the RSNA
Pneumonia Detection Challenge, which is a subset of NIH ChestX-ray14. A model trained on NIH can
therefore be tested on images (or patients) it saw in training. Run this before any NIH -> COVID
transfer experiment and pass the output to ``data.exclusions`` of the target config.

The script writes every match with its Hamming distance and the matched NIH image / patient, so you
can (a) drop the matched target images and (b) drop *all* target images whose NIH patient appears in
the NIH training split (use --nih-train-split to get that list).

Example:
    python scripts/find_cross_dataset_duplicates.py \
        --source-root /data/NIH --target-root /data/COVID-19_Radiography_Dataset \
        --nih-train-split splits/nih/train.csv --out exclusions/covid_vs_nih.csv
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _phash(path):
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as im:
            return path, int(str(imagehash.phash(im.convert("L"), hash_size=8)), 16)
    except Exception as e:  # unreadable file
        print(f"skip {path}: {e}")
        return path, None


def hash_all(paths, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_phash, paths, chunksize=256))
    res = [(p, h) for p, h in res if h is not None]
    return [p for p, _ in res], np.array([h for _, h in res], dtype=np.uint64)


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True, help="NIH ChestX-ray14 root (Data_Entry_2017.csv + images*/)")
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-distance", type=int, default=4, help="Hamming distance (of 64 bits) counted as a match")
    ap.add_argument("--nih-train-split", default=None, help="train.csv of the NIH split (adds 'in_nih_train')")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    src = sorted(glob.glob(os.path.join(args.source_root, "**", "*.png"), recursive=True))
    tgt = sorted(p for p in glob.glob(os.path.join(args.target_root, "**", "*.*"), recursive=True)
                 if p.lower().endswith((".png", ".jpg", ".jpeg")) and "/masks/" not in p.lower())
    print(f"hashing {len(src)} source and {len(tgt)} target images ...")
    src_p, src_h = hash_all(src, args.workers)
    tgt_p, tgt_h = hash_all(tgt, args.workers)

    rows = []
    for i in range(0, len(tgt_h), args.chunk):
        block = tgt_h[i:i + args.chunk]
        best_d = np.full(len(block), 65)
        best_j = np.zeros(len(block), dtype=np.int64)
        for j in range(0, len(src_h), 20000):
            d = popcount64(block[:, None] ^ src_h[None, j:j + 20000]).astype(np.int16)
            arg = d.argmin(1)
            val = d[np.arange(len(block)), arg]
            better = val < best_d
            best_d[better], best_j[better] = val[better], arg[better] + j
        for k in np.flatnonzero(best_d <= args.max_distance):
            rows.append({"path": tgt_p[i + k], "nih_path": src_p[best_j[k]], "hamming": int(best_d[k])})
        print(f"  {min(i + args.chunk, len(tgt_h))}/{len(tgt_h)} target images checked, matches so far: {len(rows)}")

    df = pd.DataFrame(rows, columns=["path", "nih_path", "hamming"])
    df["nih_image"] = df["nih_path"].map(os.path.basename)
    meta_csv = os.path.join(args.source_root, "Data_Entry_2017.csv")
    if os.path.exists(meta_csv):
        meta = pd.read_csv(meta_csv, usecols=["Image Index", "Patient ID"])
        df = df.merge(meta.rename(columns={"Image Index": "nih_image", "Patient ID": "nih_patient"}),
                      on="nih_image", how="left")
    if args.nih_train_split:
        train_pat = set(pd.read_csv(args.nih_train_split)["patient"].astype(str))
        df["in_nih_train"] = df["nih_patient"].astype(str).isin(train_pat)
    df["target_class"] = df["path"].map(lambda p: os.path.relpath(p, args.target_root).split(os.sep)[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} target images matched an NIH image (distance <= {args.max_distance}); written to {args.out}")
    if len(df):
        print(df.groupby("target_class").size().rename("matches").to_string())
        print("Inspect a sample of matches visually before relying on the threshold.")


if __name__ == "__main__":
    main()
