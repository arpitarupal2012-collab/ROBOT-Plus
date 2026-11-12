# Reviewer comments → experiments → outputs

This table maps each second-round review comment that needs a new or corrected result to the run that produces it, and to the table or figure file it generates. A comment counts as addressed in the manuscript only once its run has finished, its table and figure exist, and the corresponding `\regen{}` / `\authornote{}` markers in the revised `.tex` have been replaced.

Commands assume configs with real `root:` paths. `run_grid` = `python scripts/run_grid.py --grid`, `sum` = `python scripts/summarize.py`, `fig` = `python scripts/make_figures.py`.

| Comment | What is needed | Run | Output → manuscript |
|---|---|---|---|
| **Editor E3** (code with DOI) | public code + DOI | push this repo, make a GitHub release, Zenodo archives it | Code Availability (DOI) |
| **R2.1**, **R4.4** patient leakage | patient-disjoint splits | `chexpert_robotxdpp.yaml`, `mimic_robotxdpp.yaml`, `covidx_robotxdpp.yaml` (official patient split); NIH already patient-disjoint | `tables_test/summary.csv` → Tables 6, 8, 10 |
| **R2.1**, **R4.2** NIH ⊂ RSNA ⊂ COVID-19 Radiography / COVIDx | overlap removed | `scripts/find_cross_dataset_duplicates.py` → `data.exclusions` / `exclude_sources: [rsna]` | `exclusions/*.csv` (counts) → Section 4.1 |
| **R2.2**, **R2.3**, **R4.8** decimals, table/text agreement | one source for every number | `sum --root …` for every experiment | `per_run.csv`, `summary.csv`, `paired.csv` → all tables |
| **R2.5** CI with 1.96 | t-based CI + bootstrap | `sum … --bootstrap 2000` | `summary.csv` (`*_ci_low/high`), `per_run.csv` (`*_boot_*`) |
| **R2.6** unfair MMD vs centroid | matched alignment comparison | `run_grid configs/grid_alignment_matched.yaml`; `sum --root results/grid_alignment_matched --split target_test --reference source_only` | `paired.csv`, `fig bars` → new table |
| **R4.1** softmax for multi-label | sigmoid + BCE | all NIH / CheXpert / MIMIC configs (multi-label is automatic) | Tables 8, 10, transfer tables; `fig roc` |
| **R4.2** proxy COVID mapping | shared-label transfer instead | **Priority 1** below | new transfer table + `fig transfer` |
| **R4.3** classifier dimension | 14 sigmoid outputs; no "No Finding" class | built into configs (`labels: all` / `shared` / `chexpert14_no_nofinding`) | Section 3.7 text |
| **R4.5** target-data access | per-run record | every transfer run writes `run.json → target_access` | Table "Target-data access" |
| **R4.6** PCA drift, Eq. 36 centring | centred PCA; end-of-training retained variance; principal angles | any run with `compression: pca` (`diagnostics.json → pca_drift`) | `fig drift` → Section 3.3 |
| **R4.7** rank-limited orthogonality | covariance penalty; effective rank; feature correlation | `loss.decorrelation: covariance` (default) vs `gram` (`ctrl_gram_penalty` in the factorial grid) | `diagnostics.json → latent_test.*`; `fig factorial` |
| **R4.9** localisation target, reader data | expert boxes with documented provenance | `scripts/prepare_rsna.py` → `annotations/rsna_boxes.csv`; `scripts/explain_localization.py --cam-class Pneumonia` (NIH model; RSNA images whose NIH patient was in NIH training excluded first) | `localization_gradcam.csv` → Table 5 (abnormal images only) |
| **R4.10** "independently" / pending analyses | factorial evidence | **Priority 2** below | `factorial_*.csv`, `fig factorial` → Discussion and Conclusion |
| **Priority 1** shared-label transfer, source-only vs UDA, per-class AUROC/AUPRC, patient bootstrap | NIH↔CheXpert, NIH→MIMIC, CheXpert↔MIMIC | `train.py --config configs/transfer_*.yaml --seed S` for S in 42 7 13 21 99, with and without `--set loss.alignment=centroid --name <…>_uda`; or `run_grid configs/grid_transfer_all.yaml` per base | `sum --split target_test --bootstrap 2000`; `fig transfer` |
| **Priority 2** 2³ factorial + random projection + covariance controls | 8 cells × 5 seeds + 2 controls | `run_grid configs/grid_factorial_transfer.yaml` | `factorial_analysis.py` → `factorial_macro_auroc.csv`, `factorial_latent_test.effective_rank.csv`, …; `fig factorial`, `fig drift` |
| **R1.7**, headline COVID-19 numbers | rerun of the model in Section 3 | `run_grid configs/grid_ablation_covid.yaml` (COVID-19 Radiography) and/or `covidx_robotxdpp.yaml` (patient-level) | `summary.csv` → Abstract, Tables 6, 7, 12 |

## Compute estimate (single RTX 3090-class GPU, 40 epochs)

| Block | Runs | Approx. GPU-hours |
|---|---|---|
| CheXpert in-domain, 5 seeds | 5 | ~75 |
| NIH in-domain + 3 baselines, 5 seeds | 20 | ~150 |
| Transfer, 5 directions × source-only/UDA × 5 seeds | 50 | ~400–600 (dominated by CheXpert/MIMIC sources) |
| Factorial 2³ + 2 controls, 5 seeds | 50 | ~400 |
| Matched alignment, 4 arms × 5 seeds | 20 | ~160 |

Ways to cut this if needed: reduce MIMIC-CXR to its frontal official split (already the default), use CheXpert-small (the default download), and run the factorial on the cheapest clean pair (NIH → CheXpert). If anything must be dropped, report the omission in the manuscript rather than describing the analysis as pending.
