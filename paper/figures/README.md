# Manuscript figures

These are the 33 figures of the submitted manuscript *"ROBOT-XD++: A Dimensionality-Constrained, Transferable, and Explainable Framework for Cross-Dataset Chest X-Ray Classification"*, extracted at full resolution from the manuscript PDF. File names match the `\includegraphics{...}` names in the LaTeX source, so they can be dropped into the Overleaf project unchanged.

The result figures show the results **as reported in the current manuscript version**. Following the second-round review, the result figures are being regenerated from saved per-seed predictions with `scripts/make_figures.py` (see [`docs/REVIEW_TRACEABILITY.md`](../../docs/REVIEW_TRACEABILITY.md)). The regenerated versions will replace these files, and the change will be recorded in the commit history.

## Method

| File | LaTeX label |
|---|---|
| [`dual diag robot (1).png`](dual%20diag%20robot%20(1).png) | `fig:robot_architecture` |
| [`pca_variance_curve.png`](pca_variance_curve.png) | `fig:pca_variance` |

## Datasets

| File | LaTeX label |
|---|---|
| [`covid data sample.png`](covid%20data%20sample.png) | `fig:covid_samples` |
| [`nih image.png`](nih%20image.png) | `fig:nih_samples` |
| [`NIH_ORIGINAL DATA SAMPLE PLOT.png`](NIH_ORIGINAL%20DATA%20SAMPLE%20PLOT.png) | `fig:nih_distribution` |

## COVID-19 Radiography results

| File | LaTeX label |
|---|---|
| [`covid_loss_accuracy.png`](covid_loss_accuracy.png) | `fig:covid_loss_accuracy` |
| [`F1SCORE_ROBOT_XD_PLUS2_COVID19.png`](F1SCORE_ROBOT_XD_PLUS2_COVID19.png) | `fig:covid_f1_curve` |
| [`covid_covergence1_UP1.png`](covid_covergence1_UP1.png) | `fig:covid_ablation_f1` |
| [`ABALATION_TABLE11.png`](ABALATION_TABLE11.png) | `fig:covid_ablation_table11` |
| [`confusion_ROBOT_XD_PLUS2_COVID19.png`](confusion_ROBOT_XD_PLUS2_COVID19.png) | `fig:covid_confusion_matrix` |
| [`roc_curves.png`](roc_curves.png) | `fig:covid_roc` |
| [`covid_performance_comparison.png`](covid_performance_comparison.png) | `fig:covid_performance_comparison` |
| [`No_Alignment_Loss_loss_curve.png`](No_Alignment_Loss_loss_curve.png) | `fig:no_alignment_loss_curve` |
| [`No_Alignment_Loss_metrics_curve.png`](No_Alignment_Loss_metrics_curve.png) | `fig:no_alignment_metrics_curve` |

## CheXpert results

| File | LaTeX label |
|---|---|
| [`chexpert_ablation_all_metrics.png`](chexpert_ablation_all_metrics.png) | `fig:chexpert_ablation_metrics` |
| [`chexpert_ablation_f1_macro.png`](chexpert_ablation_f1_macro.png) | `fig:chexpert_f1_bar` |
| [`chexpert_ablation_auc_macro.png`](chexpert_ablation_auc_macro.png) | `fig:chexpert_auc_bar` |
| [`chexpert_all_models_f1_curve.png`](chexpert_all_models_f1_curve.png) | `fig:chexpert_f1_curve` |
| [`chexpert_all_models_auc_curve.png`](chexpert_all_models_auc_curve.png) | `fig:chexpert_auc_curve` |
| [`chexpert_Backbone_Bottleneck_Only_loss_curve.png`](chexpert_Backbone_Bottleneck_Only_loss_curve.png) | `fig:chexpert_backbone_loss` |
| [`chexpert_Backbone_Bottleneck_Only_metrics_curve.png`](chexpert_Backbone_Bottleneck_Only_metrics_curve.png) | `fig:chexpert_backbone_metrics` |
| [`chexpert_Full_ROBOT_XDPP_loss_curve.png`](chexpert_Full_ROBOT_XDPP_loss_curve.png) | `fig:chexpert_full_loss` |
| [`chexpert_Full_ROBOT_XDPP_metrics_curve.png`](chexpert_Full_ROBOT_XDPP_metrics_curve.png) | `fig:chexpert_full_metrics` |
| [`chexpert_No_PCA_loss_curve.png`](chexpert_No_PCA_loss_curve.png) | `fig:chexpert_no_pca_loss` |
| [`chexpert_No_PCA_metrics_curve.png`](chexpert_No_PCA_metrics_curve.png) | `fig:chexpert_no_pca_metrics` |
| [`chexpert_No_Orthogonal_Loss_loss_curve.png`](chexpert_No_Orthogonal_Loss_loss_curve.png) | `fig:chexpert_no_orth_loss` |
| [`chexpert_No_Orthogonal_Loss_metrics_curve.png`](chexpert_No_Orthogonal_Loss_metrics_curve.png) | `fig:chexpert_no_orth_metrics` |
| [`chexpert_No_Alignment_Loss_loss_curve.png`](chexpert_No_Alignment_Loss_loss_curve.png) | `fig:chexpert_no_align_loss` |
| [`chexpert_No_Alignment_Loss_metrics_curve.png`](chexpert_No_Alignment_Loss_metrics_curve.png) | `fig:chexpert_no_align_metrics` |

## NIH ChestX-ray14 results

| File | LaTeX label |
|---|---|
| [`output.png`](output.png) | `fig:nih_auc_curve` |
| [`confusion matrix_all_NIH DATA.png`](confusion%20matrix_all_NIH%20DATA.png) | `fig:nih_confusion_matrix` |

## Explainability

| File | LaTeX label |
|---|---|
| [`covid_gradcam.png`](covid_gradcam.png) | `fig:covid_gradcam` |
| [`supplementary_cam_comparison.png`](supplementary_cam_comparison.png) | `fig:supp_cam_comparison` |
