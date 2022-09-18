# skin-lesion-classification
Repository for skin lesion classification task


## Setup:
### Clone the repository 
```
git clone https://github.com/pavelg0t/skin-lesion-classification.git
cd skin-lesion-classification/
```

### Setup virtual environment with `venv`
```
python3 -m venv venv
source venv/bin/activate
```
### Install necessary packages
```
pip install -r requirements.txt
```

## Download the data (via Kaggle)
### Set up [Kaggle API CLI](https://github.com/Kaggle/kaggle-api):
Install the pip package (if not already installed with the requirements.txt file)
```
pip install kaggle
```
Create an API Token:
Go to (`https://www.kaggle.com/<username>/account`) and select 'Create New API Token'.
Then move this JSON file to `~/.kaggle/kaggle.json`: 
```
mv <path_to_file>/kaggle.json ~/.kaggle/kaggle.json
```
To ensure that other computer users do not have read access to your credentials: 
```
chmod 600 ~/.kaggle/kaggle.json
```

### Download the datasets:
#### HAM10000
```
kaggle datasets download -p data/HAM10000 --unzip kmader/skin-cancer-mnist-ham10000
```

#### ISIC2020
```
kaggle datasets download -p data/ISIC2020 --unzip mnowak061/isic2020-384x384-jpeg
wget -O data/ISIC2020/ISIC_2020_Training_GroundTruth_v2.csv https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv
wget -O data/ISIC2020/ISIC_2020_Training_Duplicates.csv https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv
wget -O data/ISIC2020/ISIC_2020_Test_Metadata.csv https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv
```


## References

- Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Caffery, L., Chousakos, E., Codella, N., Combalia, M., Dusza, S., Guitera, P., Gutman, D., Halpern, A., Helba, B., Kittler, H., Kose, K., Langer, S., Lioprys, K., Malvehy, J., Musthaq, S., Nanda, J., Reiter, O., Shih, G., Stratigos, A., Tschandl, P., Weber, J. & Soyer, P. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. Sci Data 8, 34 (2021). https://doi.org/10.1038/s41597-021-00815-z
- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).
