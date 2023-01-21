# skin-lesion-classification
Repository for skin lesion classification task


## Setup:
**Clone the repository**
```
git clone https://github.com/pavelg0t/skin-lesion-classification.git
cd skin-lesion-classification/
```

**Setup virtual environment with `venv`**
```
python3 -m venv venv
source venv/bin/activate
```
**Install necessary packages (Python version used: `Python 3.8.10`)**
```
pip install -r requirements.txt
```

## Setup pre-commit hooks
```
pre-commit install
```
Initially run the `pre-commit` hooks on all of the files in the repo:
```
pre-commit run --all-files
```
## Setup WandB for experiment tracking
Install the pip package (if not already installed with the requirements.txt file)
```
pip install wandb --upgrade
wandb login
```
Go to https://wandb.ai/authorize, login, copy the key and paste it in the terminal. The API key will be automatically saved to `/home/<user>/.netrc`.

## Download the data (via Kaggle)
**Set up [Kaggle API CLI](https://github.com/Kaggle/kaggle-api)**
Install the pip package (if not already installed with the requirements.txt file)
```
pip install kaggle
```
**Create an API Token**

Go to (`https://www.kaggle.com/<username>/account`) and select 'Create New API Token'.
Then move this JSON file to `~/.kaggle/kaggle.json`:
```
mv <path_to_file>/kaggle.json ~/.kaggle/kaggle.json
```
To ensure that other computer users do not have read access to your credentials:
```
chmod 600 ~/.kaggle/kaggle.json
```

**Download the datasets**

**HAM10000**
```
kaggle datasets download -p data/HAM10000 --unzip kmader/skin-cancer-mnist-ham10000
```

**ISIC2020**
```
bash setup_isic2020.sh
```

## Train
**Binary classification with timm model (xcit nano)**
```
python3 train.py First_Project --m_name xcit_nano_12_p16_224 --dataset ISIC2020 --multiclass false --transform timm_original --num_workers 4 --batch_size 8 --n_epoch 2
```


## References

- Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Caffery, L., Chousakos, E., Codella, N., Combalia, M., Dusza, S., Guitera, P., Gutman, D., Halpern, A., Helba, B., Kittler, H., Kose, K., Langer, S., Lioprys, K., Malvehy, J., Musthaq, S., Nanda, J., Reiter, O., Shih, G., Stratigos, A., Tschandl, P., Weber, J. & Soyer, P. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. Sci Data 8, 34 (2021). https://doi.org/10.1038/s41597-021-00815-z
- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).
