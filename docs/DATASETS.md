# Datasets

No data is distributed with this repository. `scripts/download_data.sh` fetches each dataset from the source listed below; then set the `root:` path in the matching config.

| Key | Dataset | Official source | Download used here | Labels | Patient IDs | Config |
|---|---|---|---|---|---|---|
| `nih` | NIH ChestX-ray14: 112,120 frontal images, 30,805 patients | https://nihcc.app.box.com/v/ChestXray-NIHCC | Kaggle `nih-chest-xrays/data` | 14 findings, multi-label (NLP-derived) | yes | `nih_robotxdpp.yaml` |
| `chexpert` | CheXpert v1.0: 224,316 images, 65,240 patients | https://stanfordmlgroup.github.io/competitions/chexpert/ | Kaggle `ashery/chexpert` (the downsampled "small" version) | 14 observations; 1 / 0 / −1 uncertain / blank | yes (in the path) | `chexpert_robotxdpp.yaml` |
| `mimic` | MIMIC-CXR-JPG v2.1.0: about 377k images, 65k patients | https://physionet.org/content/mimic-cxr-jpg/ (credentialed) | PhysioNet only | the same 14 CheXpert-labeler observations | yes; official patient-level split | `mimic_robotxdpp.yaml` |
| `rsna` | RSNA Pneumonia Detection Challenge: 26,684 images | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge | Kaggle competition | 3 classes, plus expert opacity bounding boxes | per-image IDs only | `rsna_robotxdpp.yaml` |
| `covidx` | COVIDx CXR-3/CXR-4 (COVID-Net project) | https://github.com/lindawangg/COVID-Net | Kaggle `andyczhao/covidx-cxr2` | COVID-19 positive / negative (CXR-3B/4) or 3-class (CXR-3A) | yes; official patient-level split | `covidx_robotxdpp.yaml` |
| `covid` | COVID-19 Radiography Database (Rahman et al.) | https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database | Kaggle | 4 classes | **no** | `covid_robotxdpp.yaml` |

**COVIDx and the COVID-19 Radiography Database are different datasets.** The Kaggle link `tawsifurrahman/covid19-radiography-database` is the COVID-19 Radiography Database. COVIDx is a separate compilation from the COVID-Net project that *uses* that database as one of its sources.

**MIMIC-CXR.** Use only the PhysioNet release. Unofficial copies, including Kaggle mirrors, break the data use agreement, and a journal may ask you to name the official source. The JPG release (`mimic-cxr-jpg`) is what the loader expects; the 4.7 TB DICOM release is not needed.

## Overlap between datasets: read before any cross-dataset experiment

```
NIH ChestX-ray14  ⊃  RSNA Pneumonia Challenge (30,000 NIH images re-annotated)
                        ├─ COVID-19 Radiography Database: Normal (8,851) and Lung Opacity (6,012)
                        └─ COVIDx: RSNA-sourced negative / pneumonia images
CheXpert (Stanford)          no known overlap with the others
MIMIC-CXR (Beth Israel)      no known overlap with the others
```

| Train on | Evaluate on | Clean? | What to do |
|---|---|---|---|
| CheXpert | MIMIC-CXR (and the reverse) | **yes** | none; identical 14-label space (`transfer_chexpert_to_mimic.yaml`) |
| NIH | CheXpert or MIMIC-CXR (and the reverse) | **yes** | none; 7 harmonised findings (`labels: shared`) |
| NIH | RSNA | no: RSNA ⊂ NIH | exclude RSNA images whose NIH patient is in the NIH training split (`find_cross_dataset_duplicates.py`) |
| NIH | COVIDx | no | `exclude_sources: [rsna]`, then run the duplicate check |
| NIH | COVID-19 Radiography | no | run the duplicate check; results without it are not valid transfer evidence |

## Harmonised findings (`labels: shared`)

| NIH ChestX-ray14 | CheXpert / MIMIC-CXR |
|---|---|
| Atelectasis | Atelectasis |
| Cardiomegaly | Cardiomegaly |
| Consolidation | Consolidation |
| Edema | Edema |
| Effusion | Pleural Effusion |
| Pneumonia | Pneumonia |
| Pneumothorax | Pneumothorax |

## Expected folder layout

```
data/NIH_ChestXray14/Data_Entry_2017.csv, images_001/images/*.png, ...
data/CheXpert/CheXpert-v1.0-small/train.csv, train/patientXXXXX/...   -> root: data/CheXpert/CheXpert-v1.0-small
data/physionet.org/files/mimic-cxr-jpg/2.1.0/files/p10/..., mimic-cxr-2.0.0-{chexpert,metadata,split}.csv.gz
data/rsna-pneumonia-detection-challenge/stage_2_train_images/*.dcm, stage_2_train_labels.csv,
                                        stage_2_detailed_class_info.csv, png/ (created by prepare_rsna.py)
data/COVIDx/train/ val/ test/, train.txt val.txt test.txt
data/COVID-19_Radiography_Dataset/COVID/ Lung_Opacity/ Normal/ Viral Pneumonia/
```

## Dataset citations

- Wang X. et al. ChestX-ray8. CVPR 2017.
- Irvin J. et al. CheXpert. AAAI 2019.
- Johnson A.E.W. et al. MIMIC-CXR-JPG. PhysioNet (2019); MIMIC-CXR, Scientific Data 6, 317 (2019).
- Shih G. et al. Augmenting the NIH chest radiograph dataset with expert annotations of possible pneumonia. Radiology: AI 1(1), e180041 (2019).
- Wang L., Lin Z.Q., Wong A. COVID-Net. Scientific Reports 10, 19549 (2020); COVIDx CXR-4: arXiv:2311.17677 (2023).
- Chowdhury M.E.H. et al., IEEE Access (2020); Rahman T. et al., Computers in Biology and Medicine 132, 104319 (2021).
