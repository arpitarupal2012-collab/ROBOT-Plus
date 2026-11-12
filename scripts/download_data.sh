#!/usr/bin/env bash
# Download the datasets used by ROBOT-XD++ into $DATA (default: ./data).
#
# Requirements
#   Kaggle  : pip install kaggle ; put kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > Create token)
#             For RSNA you must first click "Join competition" / accept the rules on the competition page.
#   MIMIC   : a credentialed PhysioNet account that has signed the MIMIC-CXR-JPG data use agreement.
#             Do NOT use unofficial Kaggle copies of MIMIC-CXR: redistribution is not allowed by the DUA,
#             and journals may ask you to state the official source.
#
# Usage
#   bash scripts/download_data.sh nih chexpert rsna covid19rad covidx       # any subset
#   PHYSIONET_USER=yourname bash scripts/download_data.sh mimic
set -euo pipefail
DATA=${DATA:-./data}
mkdir -p "$DATA"

need_kaggle() { command -v kaggle >/dev/null || { echo "install the Kaggle CLI: pip install kaggle"; exit 1; }; }

for ds in "$@"; do
  case "$ds" in
    nih)        # NIH ChestX-ray14, 112,120 frontal images, 14 findings (~45 GB)
      need_kaggle
      kaggle datasets download -d nih-chest-xrays/data -p "$DATA/NIH_ChestXray14" --unzip ;;
    chexpert)   # CheXpert-v1.0-small (downsampled, ~11 GB). For full resolution use the Stanford AIMI site.
      need_kaggle
      kaggle datasets download -d ashery/chexpert -p "$DATA/CheXpert" --unzip ;;
    rsna)       # RSNA Pneumonia Detection Challenge (subset of NIH ChestX-ray14, with expert boxes)
      need_kaggle
      kaggle competitions download -c rsna-pneumonia-detection-challenge -p "$DATA/rsna-pneumonia-detection-challenge"
      (cd "$DATA/rsna-pneumonia-detection-challenge" && unzip -q -o rsna-pneumonia-detection-challenge.zip)
      echo "next: python scripts/prepare_rsna.py --root $DATA/rsna-pneumonia-detection-challenge" ;;
    covid19rad) # COVID-19 Radiography Database (Rahman et al.) - NOT COVIDx
      need_kaggle
      kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p "$DATA" --unzip ;;
    covidx)     # COVIDx CXR (COVID-Net project). Check https://github.com/lindawangg/COVID-Net for the
                # current release; the Kaggle slug below hosts COVIDx CXR-3/CXR-4.
      need_kaggle
      kaggle datasets download -d andyczhao/covidx-cxr2 -p "$DATA/COVIDx" --unzip ;;
    mimic)      # MIMIC-CXR-JPG v2.1.0 (~570 GB). The DICOM release (physionet.org/content/mimic-cxr/) is not needed.
      : "${PHYSIONET_USER:?set PHYSIONET_USER to your PhysioNet username}"
      wget -r -N -c -np --user "$PHYSIONET_USER" --ask-password \
           -P "$DATA" https://physionet.org/files/mimic-cxr-jpg/2.1.0/ ;;
    *) echo "unknown dataset '$ds' (choose: nih chexpert rsna covid19rad covidx mimic)"; exit 1 ;;
  esac
done
echo "done. Set the 'root:' paths in configs/*.yaml to the folders under $DATA."
