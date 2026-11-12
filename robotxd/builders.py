"""Build split tables and DataLoaders from a config section.

A data section looks like::

    data:
      dataset: covid          # covid | nih | chexpert
      root: /path/to/dataset
      split_dir: splits/covid # split CSVs are written here once and re-used by every run
      split_seed: 42
      fractions: [0.70, 0.15, 0.15]
      labels: all             # all | shared  (shared = NIH/CheXpert harmonised findings, NIH names)
      uncertain: ignore       # CheXpert -1 policy: ignore | zeros | ones
      exclusions: null        # CSV with a `path` column of images to drop (e.g. cross-dataset duplicates)
      image_size: 224
      augment: paper          # paper | notebook
      batch_size: 32
      num_workers: 4
      sampler: balanced       # balanced | none (single-label only)
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from . import data as D


SINGLE_LABEL = {"covid", "rsna", "covidx"}


def _table(dcfg: dict):
    ds = dcfg["dataset"]
    if ds == "covid":
        return D.covid_radiography_table(dcfg["root"]), None, False
    if ds == "rsna":
        return D.rsna_table(dcfg["root"], dcfg.get("image_dir", "png")), None, False
    if ds == "covidx":
        df = D.covidx_table(dcfg["root"], dcfg.get("label_map"), dcfg.get("split_files"))
        drop = dcfg.get("exclude_sources") or []
        if drop:  # e.g. ["rsna"] when NIH ChestX-ray14 is the source domain
            n = len(df)
            df = df[~df["source"].str.lower().isin([s.lower() for s in drop])].reset_index(drop=True)
            print(f"COVIDx: excluded {n - len(df)} images from sources {drop}")
        return df, None, False
    if ds == "nih":
        df, names = D.nih_table(dcfg["root"]), list(D.NIH_LABELS)
    elif ds == "chexpert":
        df, names = D.chexpert_table(dcfg["root"], dcfg.get("csv", "train.csv")), list(D.CHEXPERT_LABELS)
    elif ds == "mimic":
        df, names = D.mimic_cxr_table(dcfg["root"]), list(D.CHEXPERT_LABELS)
    else:
        raise ValueError(f"unknown dataset {ds!r}")
    labels = dcfg.get("labels", "all")
    if labels == "shared":  # NIH <-> CheXpert/MIMIC harmonised findings, NIH names
        if ds in ("chexpert", "mimic"):
            df = df.rename(columns={v: k for k, v in D.SHARED_NIH_CHEXPERT.items()})
        names = list(D.SHARED_NIH_CHEXPERT)
    elif labels == "chexpert14_no_nofinding" and ds in ("chexpert", "mimic"):
        names = [n for n in D.CHEXPERT_LABELS if n != "No Finding"]
    return df, names, True


def build_splits(dcfg: dict):
    """Returns (train_df, val_df, test_df, label_names, class_names, multilabel)."""
    df, label_names, multilabel = _table(dcfg)
    df = D.apply_exclusions(df, dcfg.get("exclusions"))
    split_dir = dcfg.get("split_dir")
    if split_dir and os.path.exists(os.path.join(split_dir, "test.csv")):
        parts = D.load_split(split_dir)
        print(f"Loaded fixed split from {split_dir}")
    else:
        if dcfg.get("split", "random") == "official":
            if "official_split" not in df:
                raise ValueError(f"{dcfg['dataset']} has no official split; use split: random")
            parts = D.official_split(df, dcfg.get("val_fraction", 0.15), dcfg.get("split_seed", 42))
        else:
            parts = D.split_dataframe(df, tuple(dcfg.get("fractions", (0.70, 0.15, 0.15))),
                                      seed=dcfg.get("split_seed", 42), group_col="patient",
                                      stratify_col=None if multilabel else "label")
        if split_dir:
            D.save_split(parts, split_dir)
    class_names = None if multilabel else sorted(df["label"].unique())
    for n, p in zip(("train", "val", "test"), parts):
        npat = p["patient"].nunique() if "patient" in p and p["patient"].notna().all() else "n/a"
        print(f"{n:5s}: {len(p):7d} images, patients={npat}")
    return (*parts, label_names, class_names, multilabel)


def build_loaders(dcfg: dict, parts, label_names, class_names, multilabel, shuffle_train=True):
    size, policy = dcfg.get("image_size", 224), dcfg.get("augment", "paper")
    bs, nw = dcfg.get("batch_size", 32), dcfg.get("num_workers", 4)
    kw = dict(label_names=label_names, class_names=class_names, multilabel=multilabel,
              uncertain=dcfg.get("uncertain", "ignore"))
    tr, va, te = parts
    ds_train = D.CXRDataset(tr, D.build_transforms(size, True, policy), **kw)
    ds_train_eval = D.CXRDataset(tr, D.build_transforms(size, False, policy), **kw)  # for PCA fitting
    ds_val = D.CXRDataset(va, D.build_transforms(size, False, policy), **kw)
    ds_test = D.CXRDataset(te, D.build_transforms(size, False, policy), **kw)

    sampler = None
    if not multilabel and dcfg.get("sampler", "balanced") == "balanced":
        sampler = D.balanced_sampler(ds_train.targets, len(class_names))
    common = dict(batch_size=bs, num_workers=nw, pin_memory=torch.cuda.is_available())
    return {
        "train": DataLoader(ds_train, sampler=sampler, shuffle=shuffle_train and sampler is None,
                            drop_last=True, **common),
        "train_eval": DataLoader(ds_train_eval, shuffle=False, **common),
        "val": DataLoader(ds_val, shuffle=False, **common),
        "test": DataLoader(ds_test, shuffle=False, **common),
    }
