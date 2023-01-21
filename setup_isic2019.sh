#!/bin/bash
# STEP 1: Export ENV VARs
# shellcheck source=/dev/null
source .config

# STEP 2: Download data via Kaggle API CLI
kaggle datasets download -p "$ISIC2019_BASE_FOLDER" --unzip cdeotte/jpeg-isic2019-384x384
