#!/usr/bin/env python
"""End-to-end smoke test on tiny synthetic data (CPU, no downloads, ~2-4 min).

Builds fake COVID-style (single-label) and NIH/CheXpert-style (multi-label) datasets, then runs:
train -> summarize for an in-domain grid, and source-only + UDA transfer -> factorial analysis.
It checks that the pipeline runs and files are produced; the numbers are meaningless.

    python tests/smoke_end_to_end.py
"""
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
PY = sys.executable
sys.path.insert(0, ROOT)

from robotxd.data import CHEXPERT_LABELS, NIH_LABELS  # noqa: E402


def img(path, rng):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((rng.random((72, 72)) * 255).astype("uint8")).save(path)


def run(*args):
    print("+", " ".join(args))
    subprocess.run([PY, *args], check=True, cwd=ROOT)


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="robotxd_smoke_")
    try:
        # fake COVID-19 Radiography
        covid = os.path.join(tmp, "covid")
        for c in ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]:
            for i in range(24):
                img(os.path.join(covid, c, "images", f"{c}-{i}.png"), rng)
        # fake NIH
        nih = os.path.join(tmp, "nih")
        rows = []
        for i in range(120):
            name = f"{i:08d}_000.png"
            img(os.path.join(nih, "images_001", "images", name), rng)
            labs = [l for l in NIH_LABELS if rng.random() < 0.15] or ["No Finding"]
            rows.append({"Image Index": name, "Finding Labels": "|".join(labs), "Patient ID": i // 2})
        pd.DataFrame(rows).to_csv(os.path.join(nih, "Data_Entry_2017.csv"), index=False)
        # fake CheXpert
        chx = os.path.join(tmp, "CheXpert-v1.0")
        rows = []
        for i in range(120):
            rel = f"CheXpert-v1.0/train/patient{i // 2:05d}/study1/view{i % 2}_frontal.jpg"
            img(os.path.join(tmp, rel), rng)
            r = {"Path": rel, "Frontal/Lateral": "Frontal"}
            r.update({l: rng.choice([1.0, 0.0, -1.0, np.nan], p=[0.2, 0.6, 0.1, 0.1]) for l in CHEXPERT_LABELS})
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(chx, "train.csv"), index=False)

        cfgdir = os.path.join(tmp, "configs")
        shutil.copytree(os.path.join(ROOT, "configs"), cfgdir)
        fast = {"train": {"epochs": 1, "amp": False}, "model": {"pretrained": False, "k": 16},
                "data": {"image_size": 64, "batch_size": 8, "num_workers": 0}}

        def patch(name, extra):
            p = os.path.join(cfgdir, name)
            c = yaml.safe_load(open(p))
            for sec, v in {**fast, **extra}.items():
                c.setdefault(sec, {}).update(v) if isinstance(v, dict) else c.__setitem__(sec, v)
            yaml.safe_dump(c, open(p, "w"))

        patch("covid_robotxdpp.yaml", {"data": {**fast["data"], "root": covid, "split_dir": os.path.join(tmp, "s/covid")}})
        patch("nih_robotxdpp.yaml", {"data": {**fast["data"], "root": nih, "split_dir": os.path.join(tmp, "s/nih")}})
        patch("transfer_nih_to_chexpert.yaml", {
            "data": {**fast["data"], "root": nih, "split_dir": os.path.join(tmp, "s/nih_sh")},
            "target": {"root": chx, "split_dir": os.path.join(tmp, "s/chx_sh"), "image_size": 64,
                       "batch_size": 8, "num_workers": 0}})
        for g in ("grid_ablation_covid.yaml", "grid_factorial_transfer.yaml"):
            p = os.path.join(cfgdir, g)
            c = yaml.safe_load(open(p))
            c["seeds"] = [1, 2]
            if g.startswith("grid_ablation"):
                c["grid"] = {k: c["grid"][k] for k in ("full", "no_decorrelation", "learned_projection")}
            else:
                c["grid"] = {k: v for k, v in c["grid"].items() if not k.startswith("ctrl")}
            yaml.safe_dump(c, open(p, "w"))

        out = os.path.join(tmp, "results")
        run("scripts/run_grid.py", "--grid", os.path.join(cfgdir, "grid_ablation_covid.yaml"), "--out", out)
        run("scripts/summarize.py", "--root", os.path.join(out, "grid_ablation_covid"), "--reference", "full",
            "--bootstrap", "50")
        run("scripts/train.py", "--config", os.path.join(cfgdir, "nih_robotxdpp.yaml"), "--seed", "1", "--out", out)
        run("scripts/run_grid.py", "--grid", os.path.join(cfgdir, "grid_factorial_transfer.yaml"), "--out", out)
        run("scripts/summarize.py", "--root", os.path.join(out, "grid_factorial_transfer"), "--split", "target_test",
            "--bootstrap", "50")
        run("scripts/factorial_analysis.py", "--grid", os.path.join(cfgdir, "grid_factorial_transfer.yaml"),
            "--per-run", os.path.join(out, "grid_factorial_transfer", "tables_target_test", "per_run.csv"),
            "--outcome", "macro_auroc", "latent_test.effective_rank")
        run("scripts/complexity.py", "--config", os.path.join(cfgdir, "covid_robotxdpp.yaml"),
            "--num-outputs", "4", "--iters", "3", "--device", "cpu")
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
