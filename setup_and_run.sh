#!/usr/bin/env bash
# Prepare dependencies and DIV2K. Training/evaluation are opt-in.
#
# Examples:
#   bash setup_and_run.sh
#   RUN_TRAIN=1 LATENT_CH=2 bash setup_and_run.sh
#   RUN_EVAL=1 LATENT_CH=2 bash setup_and_run.sh

set -euo pipefail

DOWNLOAD_DIV2K="${DOWNLOAD_DIV2K:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
LATENT_CH="${LATENT_CH:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export SIRA_IMPORTANCE_MODE="${SIRA_IMPORTANCE_MODE:-dino}"
export SIRA_DINO_HUB_DIR="${SIRA_DINO_HUB_DIR:-$PWD/.torch_hub}"

echo "============================================"
echo " SIRA setup"
echo "============================================"
echo "Importance mode: ${SIRA_IMPORTANCE_MODE}"
echo "DINOv2 cache:    ${SIRA_DINO_HUB_DIR}"
echo "Latent channels: ${LATENT_CH}"

echo ""
echo "[1/3] Installing Python dependencies..."
python -m pip install -r requirements.txt

download_and_extract() {
    local url="$1"
    local archive="$2"
    local destination="$3"

    if [ -d "${destination}" ]; then
        echo "  already exists: ${destination}"
        return
    fi

    mkdir -p "$(dirname "${archive}")"
    echo "  downloading: ${url}"
    wget --show-progress "${url}" -O "${archive}"
    if [ ! -s "${archive}" ]; then
        echo "ERROR: downloaded archive is empty: ${archive}" >&2
        exit 1
    fi
    unzip -q "${archive}" -d "$(dirname "${destination}")"
    rm "${archive}"
}

echo ""
echo "[2/3] Preparing datasets..."
if [ "${DOWNLOAD_DIV2K}" = "1" ]; then
    download_and_extract \
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" \
        "./data/DIV2K/DIV2K_train_HR.zip" \
        "./data/DIV2K/DIV2K_train_HR"
    download_and_extract \
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" \
        "./data/DIV2K/DIV2K_valid_HR.zip" \
        "./data/DIV2K/DIV2K_valid_HR"
else
    echo "  DIV2K download skipped (DOWNLOAD_DIV2K=${DOWNLOAD_DIV2K})"
fi

mkdir -p ./data/kodak
KODAK_COUNT="$(find ./data/kodak -maxdepth 1 -type f -size +0c | wc -l | tr -d ' ')"
if [ "${KODAK_COUNT}" -lt 24 ]; then
    echo "  Kodak-24 is incomplete (${KODAK_COUNT}/24 non-empty files)."
    echo "  Download Kodak manually and place images in ./data/kodak/."
else
    echo "  Kodak-24: ${KODAK_COUNT} non-empty files"
fi

echo ""
echo "[3/3] Optional execution..."
if [ "${RUN_TRAIN}" = "1" ]; then
    python train.py \
        --latent_ch "${LATENT_CH}" \
        --batch_size "${BATCH_SIZE}" \
        --methods cnn semantic sira_b1_init sira_b2_init \
        --importance_mode "${SIRA_IMPORTANCE_MODE}"
else
    echo "  training skipped; use RUN_TRAIN=1 to enable"
fi

if [ "${RUN_EVAL}" = "1" ]; then
    python eval.py \
        --latent_ch "${LATENT_CH}" \
        --methods cnn semantic sira_b1_init sira_b2_init \
        --importance_mode "${SIRA_IMPORTANCE_MODE}" \
        --result_dir "results/run_c${LATENT_CH}"
else
    echo "  evaluation skipped; use RUN_EVAL=1 to enable"
fi

echo ""
echo "Setup complete."
