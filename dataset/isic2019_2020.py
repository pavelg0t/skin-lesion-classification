import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .factory import DatasetFactory


@DatasetFactory.register('ISIC2019_2020')
class ISIC2019_2020(Dataset):

    _train_image_files = []
    _train_image_categories = []
    _val_image_files = []
    _val_image_categories = []
    _n_class = None

    def __init__(
        self,
        base_dir_2019=None,
        base_dir_2020=None,
        train_imgs_2019=None,
        train_imgs_2020=None,
        train_gt_2019=None,
        train_gt_2020=None,
        train=None,
        split_ratio=0.90,
        seed=42,
        preproc=Callable,
        **kwargs,
    ):
        """
        Load the image information from the drive

        Parameters
        ----------
        """
        if not base_dir_2019:
            base_dir_2019 = os.getenv('ISIC2019_BASE_FOLDER')
        if not train_imgs_2019:
            train_imgs_2019 = os.getenv('ISIC2019_TRAIN_IMGS_FOLDER')
        if not train_gt_2019:
            train_gt_2019 = os.getenv('ISIC2019_TRAIN_GT')

        if not base_dir_2020:
            base_dir_2020 = os.getenv('ISIC2020_BASE_FOLDER')
        if not train_imgs_2020:
            train_imgs_2020 = os.getenv('ISIC2020_TRAIN_IMGS_FOLDER')
        if not train_gt_2020:
            train_gt_2020 = os.getenv('ISIC2020_TRAIN_GT')

        self.train = train
        self.preproc = preproc

        if not ISIC2019_2020.is_data_initially_split():
            ISIC2019_2020.rand_split(
                base_dir_2019=base_dir_2019,
                base_dir_2020=base_dir_2020,
                train_imgs_2019=train_imgs_2019,
                train_imgs_2020=train_imgs_2020,
                train_gt_2019=train_gt_2019,
                train_gt_2020=train_gt_2020,
                split_ratio=split_ratio,
                seed=seed,
            )

    def __len__(self):
        'Denotes the total number of samples'
        if self.train:
            return len(ISIC2019_2020._train_image_files)
        else:
            return len(ISIC2019_2020._val_image_files)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.train:
            image = self.preproc(ISIC2019_2020._train_image_files[index])
            cat = ISIC2019_2020._train_image_categories[index]
            return {'image': image, 'category': cat}
        else:
            image = self.preproc(ISIC2019_2020._val_image_files[index])
            cat = ISIC2019_2020._val_image_categories[index]
            return {'image': image, 'category': cat}

    @classmethod
    def rand_split(
        cls, base_dir_2019, base_dir_2020,
        train_imgs_2019, train_imgs_2020,
        train_gt_2019, train_gt_2020,
        split_ratio, seed,
    ):

        cats_2019, files_2019 = cls.parse_files_as_binary_class(
            base_dir=base_dir_2019,
            train_imgs=train_imgs_2019,
            train_gt=train_gt_2019,
        )
        cats_2020, files_2020 = cls.parse_files_as_binary_class(
            base_dir=base_dir_2020,
            train_imgs=train_imgs_2020,
            train_gt=train_gt_2020,
        )

        # Concatenate and shuffle ISIC2019 and ISIC2020 data together
        cats_2019_20 = np.concatenate((cats_2019, cats_2020), axis=None)
        files_2019_20 = np.concatenate((files_2019, files_2020), axis=None)

        cats_2019_20, files_2019_20 = cls.unison_shuffled_copies(
            cats_2019_20,
            files_2019_20,
        )

        train_files, val_files, train_cats, val_cats = train_test_split(
            files_2019_20, cats_2019_20,
            train_size=split_ratio,
            random_state=seed,
            stratify=cats_2019_20,
        )
        cls._train_image_files = train_files
        cls._train_image_categories = train_cats
        cls._val_image_files = val_files
        cls._val_image_categories = val_cats
        cls._n_class = np.unique(cats_2019_20).size

    @classmethod
    def is_data_initially_split(cls):
        'Check if data already split into train/val'
        if (
            len(cls._train_image_files) and len(cls._train_image_categories)
            and len(cls._val_image_files) and len(cls._val_image_categories)
        ):
            return True
        else:
            return False

    @staticmethod
    def parse_files_as_binary_class(base_dir, train_imgs, train_gt):
        'Parse ISIC data files - labeled as 0s or 1s under "target" column'

        cats, files = [], []
        data = sorted(
            os.listdir(os.path.join(base_dir, train_imgs)), key=lambda a: int(
                os.path.basename(a).split('_')[1].split('.')[0],
            ),
        )
        # Parse ISIC Ground Truth CSV file
        df = pd.read_csv(os.path.join(base_dir, train_gt))
        df = df[['image_name', 'target']]
        df.set_index('image_name', inplace=True)

        for file in data:
            file_path = os.path.join(base_dir, train_imgs, file)
            cat = df.loc[os.path.basename(file).split('.')[0]]['target']
            cats.append(cat)
            files.append(file_path)

        return np.array(cats), np.array(files)

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    @property
    def image_files(self):
        """
        List of image files. The order of the list is important
        for other methods.

        Returns
        -------
        file_list : list(str)
            List of file names
        """
        if self.train:
            return ISIC2019_2020._train_image_files
        else:
            return ISIC2019_2020._val_image_files

    @property
    def image_categories(self):
        """
        List of image categories. The order of the list is important
        for other methods.

        Returns
        -------
        image_categories : list(str)
            List of file categories
        """
        if self.train:
            return ISIC2019_2020._train_image_categories
        else:
            return ISIC2019_2020._val_image_categories

    @property
    def n_class(self):
        """
        Return the number of distinct classes (in this case - 2)

        Returns
        -------
        n_classes : int
            Number of classes
        """
        return ISIC2019_2020._n_class
