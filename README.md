# ROBOT-XD++

**Dimensionality-constrained, dual-stream chest X-ray classification with controlled cross-dataset evaluation.**

Code for the manuscript *“ROBOT-XD++: A Dimensionality-Constrained, Transferable, and Explainable Framework for Cross-Dataset Chest X-Ray Classification”* (A. Saha Chowdhury and K. Abhishek, NIT Patna), under revision.

> **Status (v1.0.0).** The pipeline below is implemented and passes unit tests and an end-to-end smoke test on synthetic data. It has **not yet been run on the real datasets**. Results will be added to this README, with the saved per-seed predictions, once the experiments are complete. The original exploratory Colab notebook is archived unchanged in [`notebooks/archive/`](notebooks/archive/). Its differences from this implementation are listed in [`docs/NOTEBOOK_NOTES.md`](docs/NOTEBOOK_NOTES.md).

---

## Method

```
image ─┬─ EfficientNet-B0 ─ GAP ─ Linear(1280→512) ─┐
       └─ DenseNet-121   ─ GAP ─ Linear(1024→512) ─┴─ concat ─ tanh(Linear 1024→1024) = F_c
F_c ─ fixed PCA  (F_c − μ_train) U_k ,  k = 256      ← fitted once on TRAINING features, then frozen
    ─ bottleneck  Linear→BN→GELU→Dropout→Linear  = L ∈ R^256
    ─ head  (softmax for single-label, independent sigmoids for multi-label)

loss = L_cls + α·L_dec(L) + β·L_align(L_source, L_target) + γ·‖L‖²
```

| Component | Implementation | Switch |
|---|---|---|
| Compression | `pca` (fixed basis + training mean), `random` (fixed random orthonormal), `linear` (learned), `none` | `model.compression` |
| Decorrelation | `covariance`: rank-aware, scale-invariant off-diagonal correlation penalty. `gram`: the original ‖LᵀL − I‖², kept as a comparison arm (its floor is d − B; see `robotxd/losses.py`) | `loss.decorrelation` |
| Alignment | `centroid`, `coral`, `mmd`, computed between a **source** batch and an **unlabelled target** batch. Any alignment makes the run unsupervised domain adaptation (UDA). | `loss.alignment` |
| Classification | softmax + CE (COVID-19 Radiography); sigmoid + masked BCE (NIH ChestX-ray14, CheXpert) | automatic |
| Explanation | dual-stream Grad-CAM / Grad-CAM++ / Score-CAM averaged over streams; Pointing Game and IoU | `scripts/explain_localization.py` |

## Leakage controls

- **Patient-disjoint splits** whenever patient IDs exist (NIH, CheXpert); a disjointness assertion runs on every split. COVID-19 Radiography releases no patient IDs, so its split is image-level and a warning is printed.
- **Splits are fixed** and saved once (`data.split_dir`), so every variant and seed uses identical partitions. Seeds vary training only.
- **PCA** is fitted on training-split features only (evaluation transforms), and drift is measured at the end of training (`diagnostics.json`).
- **Target-data access** is recorded in every `run.json`. Target labels are never read, and model selection and thresholds use the source validation split.
- **Cross-dataset duplicates.** The COVID-19 Radiography Database takes its Normal and Lung Opacity images from the RSNA Pneumonia Challenge, which is a subset of NIH ChestX-ray14. `scripts/find_cross_dataset_duplicates.py` finds the overlap so it can be excluded before any NIH → COVID-19 experiment.

## Installation

```bash
git clone https://github.com/arpitarupal2012-collab/ROBOT-Plus.git
cd ROBOT-Plus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install the CUDA build of torch first if needed: https://pytorch.org
pytest -q                              # unit tests, ~10 s on CPU
python tests/smoke_end_to_end.py       # full pipeline on synthetic data, ~5 min on CPU
python tests/smoke_new_datasets.py     # MIMIC-CXR, COVIDx, RSNA, leakage check, figures, ~5 min
```

## Data

Six public datasets are supported. Details, folder layout and **overlaps between datasets** are in [`docs/DATASETS.md`](docs/DATASETS.md).

| Key | Dataset | Get it with | Used for |
|---|---|---|---|
| `nih` | NIH ChestX-ray14 | `bash scripts/download_data.sh nih` (Kaggle `nih-chest-xrays/data`) | in-domain multi-label; transfer source/target |
| `chexpert` | CheXpert v1.0 | `... chexpert` (Kaggle `ashery/chexpert`) | in-domain multi-label; transfer source/target |
| `mimic` | MIMIC-CXR-JPG v2.1.0 | `PHYSIONET_USER=… ... mimic` (PhysioNet, credentialed) | institutionally independent transfer target |
| `rsna` | RSNA Pneumonia Detection Challenge | `... rsna`, then `scripts/prepare_rsna.py` | expert bounding boxes for localisation |
| `covidx` | COVIDx CXR-3/4 (COVID-Net) | `... covidx` (Kaggle `andyczhao/covidx-cxr2`) | COVID-19 task with a **patient-level** official split |
| `covid` | COVID-19 Radiography Database | `... covid19rad` | original in-domain experiments (image-level split) |

Leakage-free transfer pairs: **CheXpert ↔ MIMIC-CXR** (identical labels) and **NIH ↔ CheXpert / MIMIC-CXR** (7 shared findings). RSNA, COVIDx and COVID-19 Radiography all contain NIH-derived images: run `scripts/find_cross_dataset_duplicates.py` before evaluating an NIH-trained model on them.

## One-click run on Google Colab

[`notebooks/run_all_colab.ipynb`](notebooks/run_all_colab.ipynb) downloads the data, runs every experiment requested in review (resumable), writes all tables, and draws all figures to Google Drive. [`docs/REVIEW_TRACEABILITY.md`](docs/REVIEW_TRACEABILITY.md) maps each review comment to the run and output file that answers it.

## Running the experiments

Every run writes `results/<name>/seed<S>/` containing `history.csv`, `best.pth`, `pred_val.npz`, `pred_test.npz` (and `pred_target_test.npz` for transfer), `run.json` and `diagnostics.json`. **All tables are produced from the saved predictions by `scripts/summarize.py`.**

```bash
# 1. In-domain COVID-19 Radiography + component-removal ablation (5 seeds)
python scripts/run_grid.py  --grid configs/grid_ablation_covid.yaml
python scripts/summarize.py --root results/grid_ablation_covid --reference full --bootstrap 2000

# 2. NIH and CheXpert (sigmoid/BCE, patient-disjoint) + single-backbone baselines
python scripts/run_grid.py  --grid configs/grid_baselines_nih.yaml
python scripts/train.py     --config configs/chexpert_robotxdpp.yaml --seed 42

# 3. NIH <-> CheXpert transfer on 7 harmonised findings (source-only)
python scripts/train.py --config configs/transfer_nih_to_chexpert.yaml --seed 42
python scripts/train.py --config configs/transfer_chexpert_to_nih.yaml --seed 42
#    UDA variant: same config with an alignment term
python scripts/train.py --config configs/transfer_nih_to_chexpert.yaml --seed 42 \
       --name transfer_nih_to_chexpert_uda --set loss.alignment=centroid
python scripts/summarize.py --root results --split target_test --bootstrap 2000

# 4. Matched alignment comparison (none / centroid / CORAL / MMD)
python scripts/run_grid.py  --grid configs/grid_alignment_matched.yaml
python scripts/summarize.py --root results/grid_alignment_matched --split target_test --reference source_only

# 5. 2^3 factorial ablation (compression x decorrelation x alignment) + controls
python scripts/run_grid.py  --grid configs/grid_factorial_transfer.yaml
python scripts/summarize.py --root results/grid_factorial_transfer --split target_test
python scripts/factorial_analysis.py --grid configs/grid_factorial_transfer.yaml \
       --per-run results/grid_factorial_transfer/tables_target_test/per_run.csv \
       --outcome macro_auroc latent_test.effective_rank latent_test.mean_abs_offdiag_corr \
                 pca_drift.retained_variance_fixed_basis

# 6. Cross-dataset duplicate check before any NIH -> COVID-19 experiment
python scripts/find_cross_dataset_duplicates.py --source-root /data/NIH \
       --target-root /data/COVID-19_Radiography_Dataset \
       --nih-train-split splits/nih_seed42/train.csv --out exclusions/covid_vs_nih.csv

# 7. Localisation against radiologist boxes, and complexity
python scripts/explain_localization.py --run results/covid_robotxdpp/seed42 \
       --config configs/covid_robotxdpp.yaml --annotations annotations/boxes.csv --method gradcam
python scripts/complexity.py --config configs/covid_robotxdpp.yaml --num-outputs 4
```

```bash
# 8. Figures for the paper, drawn only from saved outputs (PNG 600 dpi + PDF)
python scripts/make_figures.py all --root results --out figures
#    or individually: roc | confusion | curves | bars | transfer | factorial | drift
python scripts/make_figures.py transfer --per-run results/priority1/tables_target_test/per_run.csv --out figures

# 9. Additional transfer pairs without dataset overlap
python scripts/train.py --config configs/transfer_chexpert_to_mimic.yaml --seed 42
python scripts/train.py --config configs/transfer_nih_to_mimic.yaml --seed 42
```

`run_grid.py` skips finished runs and `train.py` resumes interrupted ones from `last.pth`, which makes Colab-style sessions safe.

## Statistics

- Seed-level 95% CIs use the t-distribution (t₀.₉₇₅,₄ = 2.776 for five seeds).
- Bootstrap 95% CIs resample **patients** (or images when no patient ID exists), with 2,000 resamples.
- Paired comparisons report the paired t-test, the exact Wilcoxon test and Cohen's d_z, with Holm correction. With five pairs the smallest attainable two-sided Wilcoxon p-value is 0.0625.
- The factorial analysis fits an OLS model with all main effects and interactions (factors coded ±1).

## Model size

Measured with `scripts/complexity.py` at 224×224 (thop counts multiply–accumulates):

| Config | Params (M) | GMACs |
|---|---|---|
| ROBOT-XD++ (`covid_robotxdpp`) | 13.3 | 3.31 |
| Original notebook model (`covid_notebook_reproduction`) | 12.9 | 3.31 |
| EfficientNet-B3 + bottleneck (`covid_effb3_bottleneck`) | 11.6 | 1.02 |

Report MACs or FLOPs (= 2 × MACs) explicitly; they differ by a factor of two.

## Repository layout

```
robotxd/            library: data, models, losses, pca, engine, metrics, stats, explain, analysis, complexity
scripts/            train, run_grid, summarize, factorial_analysis, make_figures, download_data.sh,
                    prepare_rsna, find_cross_dataset_duplicates, explain_localization, complexity
configs/            one YAML per experiment; grid_*.yaml define ablation / factorial / baseline grids
tests/              unit tests (pytest) and an end-to-end synthetic smoke test
paper/figures/      the 33 figures of the manuscript (file names as in the LaTeX source)
notebooks/          run_all_colab.ipynb (full pipeline); archive/ = original exploratory notebook
docs/               DATASETS.md, REVIEW_TRACEABILITY.md, NOTEBOOK_NOTES.md
```

## Manuscript figures

All figures of the manuscript are in [`paper/figures/`](paper/figures/), with an index mapping each file to its LaTeX label. Result figures will be replaced by versions regenerated with `scripts/make_figures.py` once the revision experiments are complete.

## Citation

See [`CITATION.cff`](CITATION.cff). A Zenodo DOI will be added on the first release.

## License

MIT, see [`LICENSE`](LICENSE). The datasets are subject to their own licences and terms of use.
