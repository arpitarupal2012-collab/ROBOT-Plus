"""Datasets, label handling and leakage-safe splits.

Every dataset item is a tuple ``(image, target, mask, index)``:

* single-label: ``target`` is a class index (int64), ``mask`` is 1.0
* multi-label:  ``target`` is a float vector in {0, 1}, ``mask`` is 1 where the label is
  known and 0 where it is uncertain/missing and should be ignored by the loss and metrics.
"""
from __future__ import annotations

import glob
import os
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# Findings defined in both label sets (NIH name -> CheXpert name). Used for NIH <-> CheXpert transfer.
SHARED_NIH_CHEXPERT = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}

COVID_CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


# ----------------------------------------------------------------------------- transforms
class _RandomCLAHE:
    def __init__(self, p: float = 0.3, clip: float = 2.0, tile: int = 8):
        self.p, self.clip, self.tile = p, clip, tile

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        import cv2  # optional dependency (installed with grad-cam)

        arr = np.asarray(img.convert("L"))
        arr = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile, self.tile)).apply(arr)
        return Image.fromarray(arr).convert("RGB")


def build_transforms(image_size: int = 224, train: bool = True, policy: str = "paper"):
    """`policy="paper"` follows Section 3.9 of the manuscript; `"notebook"` reproduces the Colab code."""
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if policy == "notebook":
        resize = [transforms.Resize((image_size, image_size))]
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)] if train else []
        return transforms.Compose(resize + aug + norm)
    resize = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    aug = []
    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            _RandomCLAHE(p=0.3),
        ]
    return transforms.Compose(resize + aug + norm)


# ----------------------------------------------------------------------------- dataset
class CXRDataset(Dataset):
    """Generic chest X-ray dataset backed by a DataFrame.

    Parameters
    ----------
    df : DataFrame with a ``path`` column and either ``label`` (single-label, class name)
         or one column per entry of ``label_names`` (multi-label: 1, 0, -1 = uncertain, NaN = missing).
    uncertain : multi-label policy for -1 values: ``"ignore"`` (masked), ``"zeros"`` or ``"ones"``.
    """

    def __init__(self, df: pd.DataFrame, transform, label_names: Sequence[str] | None = None,
                 class_names: Sequence[str] | None = None, multilabel: bool = False,
                 uncertain: str = "ignore"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.multilabel = multilabel
        self.label_names = list(label_names) if label_names is not None else None
        if multilabel:
            raw = self.df[self.label_names].to_numpy(dtype=np.float32)
            mask = ~np.isnan(raw)
            unc = raw == -1
            if uncertain == "ignore":
                mask &= ~unc
            elif uncertain == "zeros":
                raw[unc] = 0
            elif uncertain == "ones":
                raw[unc] = 1
            else:
                raise ValueError(f"unknown uncertain policy {uncertain!r}")
            raw = np.nan_to_num(raw, nan=0.0)
            raw[raw < 0] = 0
            self.targets = raw
            self.masks = mask.astype(np.float32)
        else:
            self.class_names = list(class_names)
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
            self.targets = self.df["label"].map(self.class_to_idx).to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.at[i, "path"]).convert("RGB")
        img = self.transform(img)
        if self.multilabel:
            return img, torch.from_numpy(self.targets[i]), torch.from_numpy(self.masks[i]), i
        return img, torch.tensor(self.targets[i]), torch.tensor(1.0), i


def balanced_sampler(targets: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Inverse-frequency sampler for single-label data (Eq. 48). Do not combine with a weighted loss."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    w = (len(targets) / (num_classes * counts))[targets]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)


# ----------------------------------------------------------------------------- dataset tables
def covid_radiography_table(root: str) -> pd.DataFrame:
    """COVID-19 Radiography Database: one sub-folder per class, ``masks`` sub-folders skipped."""
    rows = []
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for r, dirs, files in os.walk(cdir):
            dirs[:] = [d for d in dirs if d.lower() != "masks"]
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    rows.append((os.path.join(r, f), cls))
    df = pd.DataFrame(rows, columns=["path", "label"])
    df["patient"] = np.nan  # patient identifiers are not released for this database
    return df


def nih_table(root: str, csv_name: str = "Data_Entry_2017.csv") -> pd.DataFrame:
    """NIH ChestX-ray14: multi-hot labels from ``Finding Labels``; ``No Finding`` = all zeros."""
    meta = pd.read_csv(os.path.join(root, csv_name))
    paths = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True)}
    meta = meta[meta["Image Index"].isin(paths)].copy()
    meta["path"] = meta["Image Index"].map(paths)
    meta["patient"] = meta["Patient ID"].astype(str)
    found = meta["Finding Labels"].str.split("|")
    for lab in NIH_LABELS:
        meta[lab] = found.apply(lambda xs, lab=lab: float(lab in xs))
    return meta[["path", "patient", "Image Index"] + NIH_LABELS].reset_index(drop=True)


def chexpert_table(root: str, csv_name: str = "train.csv", frontal_only: bool = True) -> pd.DataFrame:
    """CheXpert: labels 1 / 0 / -1 (uncertain) / NaN; patient ID parsed from the path."""
    meta = pd.read_csv(os.path.join(root, csv_name))
    if frontal_only and "Frontal/Lateral" in meta:
        meta = meta[meta["Frontal/Lateral"] == "Frontal"]
    parent = os.path.dirname(os.path.normpath(root))
    meta["path"] = meta["Path"].apply(lambda p: os.path.join(parent, p))
    meta["patient"] = meta["Path"].str.extract(r"(patient\d+)")[0]
    return meta[["path", "patient"] + CHEXPERT_LABELS].reset_index(drop=True)


# ----------------------------------------------------------------------------- splits
def split_dataframe(df: pd.DataFrame, fractions=(0.70, 0.15, 0.15), seed: int = 42,
                    group_col: str | None = "patient", stratify_col: str | None = None):
    """Train/val/test split.

    If ``group_col`` is present and non-null, the split is **patient-disjoint** (no patient in two
    partitions). Otherwise it falls back to an image-level split, stratified on ``stratify_col``,
    and prints a warning so the limitation is never silent.
    """
    tr, va, te = fractions
    assert abs(tr + va + te - 1) < 1e-6
    if group_col and group_col in df and df[group_col].notna().all():
        g = df[group_col].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, train_size=tr, random_state=seed)
        tr_idx, rest_idx = next(gss.split(df, groups=g))
        rest = df.iloc[rest_idx]
        gss2 = GroupShuffleSplit(n_splits=1, train_size=va / (va + te), random_state=seed)
        va_rel, te_rel = next(gss2.split(rest, groups=rest[group_col].to_numpy()))
        out = df.iloc[tr_idx], rest.iloc[va_rel], rest.iloc[te_rel]
        _assert_disjoint(out, group_col)
        return tuple(x.reset_index(drop=True) for x in out)

    print("WARNING: no patient identifiers - using an IMAGE-LEVEL split. "
          "Report this as a limitation; results may be optimistic.")
    strat = df[stratify_col] if stratify_col else None
    train, rest = train_test_split(df, train_size=tr, stratify=strat, random_state=seed)
    strat_rest = rest[stratify_col] if stratify_col else None
    val, test = train_test_split(rest, train_size=va / (va + te), stratify=strat_rest, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _assert_disjoint(parts, col):
    sets = [set(p[col]) for p in parts]
    for i in range(3):
        for j in range(i + 1, 3):
            inter = sets[i] & sets[j]
            assert not inter, f"patient leakage between partitions {i} and {j}: {list(inter)[:5]}"


def save_split(parts, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, p in zip(("train", "val", "test"), parts):
        p.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


def load_split(out_dir: str):
    return tuple(pd.read_csv(os.path.join(out_dir, f"{n}.csv")) for n in ("train", "val", "test"))


def apply_exclusions(df: pd.DataFrame, exclusion_csv: str | None, key: str = "path") -> pd.DataFrame:
    """Drop rows listed in an exclusion file (e.g. cross-dataset duplicates found by
    ``scripts/find_cross_dataset_duplicates.py``)."""
    if not exclusion_csv:
        return df
    ex = pd.read_csv(exclusion_csv)
    before = len(df)
    df = df[~df[key].isin(set(ex[key]))].reset_index(drop=True)
    print(f"Excluded {before - len(df)} images listed in {exclusion_csv}")
    return df
