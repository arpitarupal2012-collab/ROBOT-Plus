#!/usr/bin/env python
"""Smoke test for the MIMIC-CXR, COVIDx and RSNA paths, the leakage check and the figures
(synthetic data, CPU, ~4-6 min). Checks that everything runs; the numbers are meaningless.

    python tests/smoke_new_datasets.py
"""
import gzip
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from robotxd.data import CHEXPERT_LABELS, NIH_LABELS  # noqa: E402

PY = sys.executable
FAST = {"train": {"epochs": 1, "amp": False}, "model": {"pretrained": False, "k": 16}}
DATA_FAST = {"image_size": 64, "batch_size": 8, "num_workers": 0}


def png(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def run(*args):
    print("+", " ".join(a if len(a) < 90 else a[:87] + "..." for a in args))
    subprocess.run([PY, *args], check=True, cwd=ROOT)


def write_cfg(path, base, sets):
    c = {"base": os.path.join(ROOT, "configs", base), **sets}
    yaml.safe_dump(c, open(path, "w"))
    return path


def dicom(path, arr):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = arr.astype(np.uint8).tobytes()
    ds.save_as(path, enforce_file_format=True)


def main():
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp(prefix="robotxd_smoke2_")
    try:
        # ---- NIH (some images are re-used as RSNA images to test the leakage check)
        nih = os.path.join(tmp, "nih")
        nih_imgs, rows = {}, []
        for i in range(100):
            name = f"{i:08d}_000.png"
            a = (rng.random((96, 96)) * 255).astype("uint8")
            a[: 20 + i % 50, :] //= 2  # make images distinguishable for perceptual hashing
            nih_imgs[name] = a
            png(os.path.join(nih, "images_001", "images", name), a)
            labs = [l for l in NIH_LABELS if rng.random() < 0.15] or ["No Finding"]
            rows.append({"Image Index": name, "Finding Labels": "|".join(labs), "Patient ID": i // 2})
        pd.DataFrame(rows).to_csv(os.path.join(nih, "Data_Entry_2017.csv"), index=False)

        # ---- CheXpert
        chx = os.path.join(tmp, "CheXpert-v1.0-small")
        rows = []
        for i in range(100):
            rel = f"CheXpert-v1.0-small/train/patient{i // 2:05d}/study1/view{i % 2}_frontal.jpg"
            png(os.path.join(tmp, rel), (rng.random((72, 72)) * 255).astype("uint8"))
            r = {"Path": rel, "Frontal/Lateral": "Frontal"}
            r.update({l: rng.choice([1.0, 0.0, -1.0, np.nan], p=[0.25, 0.55, 0.1, 0.1]) for l in CHEXPERT_LABELS})
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(chx, "train.csv"), index=False)

        # ---- MIMIC-CXR-JPG
        mim = os.path.join(tmp, "mimic")
        lab, meta, split = [], [], []
        for i in range(90):
            sub, study, dic = 10000000 + i // 2, 50000000 + i, f"d{i:05d}"
            png_path = os.path.join(mim, "files", f"p{str(sub)[:2]}", f"p{sub}", f"s{study}", f"{dic}.jpg")
            png(png_path, (rng.random((72, 72)) * 255).astype("uint8"))
            r = {"subject_id": sub, "study_id": study}
            r.update({l: rng.choice([1.0, 0.0, -1.0, np.nan], p=[0.25, 0.55, 0.1, 0.1]) for l in CHEXPERT_LABELS})
            lab.append(r)
            meta.append({"dicom_id": dic, "subject_id": sub, "study_id": study, "ViewPosition": "PA"})
            split.append({"dicom_id": dic, "study_id": study, "subject_id": sub,
                          "split": "test" if i >= 70 else ("validate" if i >= 60 else "train")})
        for name, rows_ in (("chexpert", lab), ("metadata", meta), ("split", split)):
            with gzip.open(os.path.join(mim, f"mimic-cxr-2.0.0-{name}.csv.gz"), "wt") as f:
                pd.DataFrame(rows_).to_csv(f, index=False)

        # ---- COVIDx (official split files)
        cx = os.path.join(tmp, "COVIDx")
        for split_, n in (("train", 60), ("val", 16), ("test", 16)):
            lines = []
            for i in range(n):
                fn = f"{split_}_{i}.png"
                png(os.path.join(cx, split_, fn), (rng.random((72, 72)) * 255).astype("uint8"))
                lines.append(f"{split_}{i // 2} {fn} {'positive' if i % 2 else 'negative'} {'rsna' if i % 5 == 0 else 'bimcv'}")
            open(os.path.join(cx, f"{split_}.txt"), "w").write("\n".join(lines))

        # ---- RSNA: 40 DICOMs, the first 10 are copies of NIH images (planted overlap)
        rs = os.path.join(tmp, "rsna")
        os.makedirs(os.path.join(rs, "stage_2_train_images"))
        lab_rows, info_rows = [], []
        nih_names = list(nih_imgs)
        for i in range(40):
            pid = f"rsna{i:04d}"
            a = nih_imgs[nih_names[i]] if i < 10 else (rng.random((96, 96)) * 255).astype("uint8")
            dicom(os.path.join(rs, "stage_2_train_images", pid + ".dcm"), a)
            cls = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"][i % 3]
            info_rows.append({"patientId": pid, "class": cls})
            if cls == "Lung Opacity":
                lab_rows.append({"patientId": pid, "x": 20, "y": 20, "width": 30, "height": 40, "Target": 1})
            else:
                lab_rows.append({"patientId": pid, "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan, "Target": 0})
        pd.DataFrame(lab_rows).to_csv(os.path.join(rs, "stage_2_train_labels.csv"), index=False)
        pd.DataFrame(info_rows).to_csv(os.path.join(rs, "stage_2_detailed_class_info.csv"), index=False)

        out, sp = os.path.join(tmp, "results"), os.path.join(tmp, "splits")
        cfg = os.path.join(tmp, "cfg")
        os.makedirs(cfg)

        # 1. RSNA preparation and leakage check against the NIH training split
        run("scripts/prepare_rsna.py", "--root", rs, "--out-annotations", os.path.join(tmp, "ann", "rsna_boxes.csv"), "--workers", "2")
        nih_cfg = write_cfg(os.path.join(cfg, "nih.yaml"), "nih_robotxdpp.yaml",
                            {**FAST, "data": {**DATA_FAST, "root": nih, "split_dir": os.path.join(sp, "nih")}})
        run("scripts/train.py", "--config", nih_cfg, "--seed", "1", "--out", out, "--name", "nih")
        exc = os.path.join(tmp, "exclusions", "rsna_vs_nih.csv")
        run("scripts/find_cross_dataset_duplicates.py", "--source-root", nih, "--target-root", os.path.join(rs, "png"),
            "--nih-train-split", os.path.join(sp, "nih", "train.csv"), "--out", exc, "--workers", "2")
        found = pd.read_csv(exc)
        planted = {os.path.join(rs, "png", f"rsna{i:04d}.png") for i in range(10)}
        assert planted <= set(found.path), f"leakage check missed planted duplicates: {planted - set(found.path)}"
        print(f"leakage check found all {len(planted)} planted NIH duplicates ({len(found)} matches in total)")

        # 2. localisation of the NIH model on RSNA expert boxes
        run("scripts/explain_localization.py", "--run", os.path.join(out, "nih", "seed1"), "--config", nih_cfg,
            "--annotations", os.path.join(tmp, "ann", "rsna_boxes.csv"), "--cam-class", "Pneumonia")

        # 3. COVIDx with the official patient-level split
        cx_cfg = write_cfg(os.path.join(cfg, "covidx.yaml"), "covidx_robotxdpp.yaml",
                           {**FAST, "data": {**DATA_FAST, "root": cx, "split_dir": os.path.join(sp, "covidx"),
                                             "exclude_sources": ["rsna"]}})
        for s in ("1", "2"):
            run("scripts/train.py", "--config", cx_cfg, "--seed", s, "--out", os.path.join(out, "covidx_runs"))

        # 4. CheXpert -> MIMIC-CXR transfer, source-only and UDA
        tr_sets = {**FAST, "data": {**DATA_FAST, "root": chx, "split_dir": os.path.join(sp, "chx")},
                   "target": {**DATA_FAST, "root": mim, "split_dir": os.path.join(sp, "mimic")}}
        tr_cfg = write_cfg(os.path.join(cfg, "chx2mimic.yaml"), "transfer_chexpert_to_mimic.yaml", tr_sets)
        for s in ("1", "2"):
            run("scripts/train.py", "--config", tr_cfg, "--seed", s, "--out", os.path.join(out, "transfer"),
                "--name", "chx2mimic_source_only")
            run("scripts/train.py", "--config", tr_cfg, "--seed", s, "--out", os.path.join(out, "transfer"),
                "--name", "chx2mimic_uda", "--set", "loss.alignment=centroid")

        # 5. tables and figures
        run("scripts/summarize.py", "--root", os.path.join(out, "covidx_runs"), "--bootstrap", "50")
        run("scripts/summarize.py", "--root", os.path.join(out, "transfer"), "--split", "target_test",
            "--reference", "chx2mimic_source_only", "--bootstrap", "50")
        run("scripts/make_figures.py", "all", "--root", out, "--out", os.path.join(tmp, "figures"))
        figs = [f for _, _, fs in os.walk(os.path.join(tmp, "figures")) for f in fs if f.endswith(".png")]
        print(f"{len(figs)} figures:", sorted(figs))
        assert len(figs) >= 8, "expected at least 8 figures"
        keep = os.environ.get("KEEP_FIGURES")
        if keep:
            shutil.copytree(os.path.join(tmp, "figures"), keep, dirs_exist_ok=True)
        print("\nSMOKE TEST (new datasets) PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
