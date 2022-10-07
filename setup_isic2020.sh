#!/bin/bash
# STEP 1: Export ENV VARs
# shellcheck source=/dev/null
source .config

# STEP 2: Download data via Kaggle API CLI
kaggle datasets download -p "$ISIC2020_BASE_FOLDER" --unzip mnowak061/isic2020-384x384-jpeg

# # STEP 3: Get Labels and metadata files from the official page
wget -O "$ISIC2020_BASE_FOLDER"/"$ISIC2020_TRAIN_GT" https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv
wget -O "$ISIC2020_BASE_FOLDER"/"$ISIC2020_TRAIN_DUPL" https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv
