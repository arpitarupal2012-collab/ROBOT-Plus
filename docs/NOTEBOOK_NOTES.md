# Notes on the original Colab notebook

`notebooks/archive/ROBOT3_ALL_original_colab.ipynb` is the exploratory notebook the project grew out of. It is kept for transparency, with outputs stripped and code unchanged. It contains several successive versions of the models (27 cells). This file records how its implementation differs from the `robotxd` package, so that anyone comparing the two knows what changed and why.

## Which cells contain what

| Cells | Content |
|---|---|
| 0, 18 | Early "ROBOT family" models (EfficientNet-B3/B4 variants; cell 18 recomputes a PCA on each training batch inside the loss) |
| 2, 3 | Feature-cache "fast ablation" versions |
| 4, 5, 6 | Dual-stream EfficientNet-B0 + DenseNet-121 ablation study (5 runs per variant), resume-safe |
| 7, 9–12 | Helper blocks for crash-safe separate ablation runs |
| 13–17 | Plotting of saved training histories |
| 19–23 | Intermediate multi-scale / SE-block versions |
| 24–26 | EfficientNet-B3 + MLP bottleneck trained on COVID-19 Radiography (cell 26: 40 epochs, single run) |

## Dual-stream model: notebook (cells 4–6) vs `robotxd.models.ROBOTXDPP`

| Aspect | Notebook | `robotxd` |
|---|---|---|
| Fusion | Linear(1024→512) → BN → GELU → Dropout | tanh(Linear(1024→1024)), as in Eq. 35 |
| "PCA" | `nn.Linear(512, 256, bias=False)`, **learned** end to end | **fixed** PCA basis and training mean fitted on training features before training (`compression: pca`); learned (`linear`) and random (`random`) projections available as controls |
| Orthogonality | ‖ZᵀZ − I‖² on row-normalised latents | same term available as `decorrelation: gram`; default is the rank-aware `covariance` penalty |
| Alignment | centroid distance between the two halves of the **same source batch** (no target data) | centroid / CORAL / MMD between a source batch and an **unlabelled target** batch; reproduce the notebook with `alignment: notebook_placeholder` |
| Class imbalance | balanced sampler **and** class-weighted loss | one or the other (`data.sampler`, `train.class_weighted_loss`) |
| Augmentation | Resize 224, flip, rotation ±10° | `augment: paper` (resize + centre crop, flip, affine ±7°, CLAHE p=0.3) or `augment: notebook` |
| LR scheduler | ReduceLROnPlateau, patience 3 | patience 5 (configurable) |
| Seeds | 42, 43, 44, 45, 46 (`SEED + run`) | any list; configs use 42, 7, 13, 21, 99 |
| Reported statistics | best-epoch **validation** metrics per run; CI = 1.96·SD/√n; Wilcoxon | best-validation checkpoint evaluated on the **test** split; t-based CI; paired t + exact Wilcoxon + Holm; patient bootstrap |
| Multi-label data | not implemented | NIH / CheXpert with sigmoid + masked BCE |

`configs/covid_notebook_reproduction.yaml` re-creates the notebook's dual-stream setting with the `robotxd` training loop, and `configs/covid_effb3_bottleneck.yaml` re-creates cell 26.

## Not present in the notebook

Code for NIH ChestX-ray14, CheXpert, cross-dataset transfer, the NIH → COVID-19 label mapping, MMD/CORAL, Grad-CAM++/Score-CAM and localisation metrics (Pointing Game, IoU) does not appear in the notebook. These are implemented for the first time in this package.
