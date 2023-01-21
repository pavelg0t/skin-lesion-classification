import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .factory import DatasetFactory


@DatasetFactory.register('ISIC2019')
class ISIC2019(Dataset):

    _train_image_files = []
    _train_image_categories = []
    _val_image_files = []
    _val_image_categories = []
    _n_class = None

    def __init__(
        self,
        base_dir=None,
        train_imgs=None,
        train_gt=None,
        multiclass=None,
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
        if not base_dir:
            base_dir = os.getenv('ISIC2019_BASE_FOLDER')
        if not train_imgs:
            train_imgs = os.getenv('ISIC2019_TRAIN_IMGS_FOLDER')
        if not train_gt:
            train_gt = os.getenv('ISIC2019_TRAIN_GT')

        self.train = train
        self.preproc = preproc

        if not ISIC2019.is_data_initially_split():
            ISIC2019.rand_split(
                base_dir=base_dir,
                train_imgs=train_imgs,
                train_gt=train_gt,
                multiclass=multiclass,
                split_ratio=split_ratio,
                seed=seed,
            )

    def __len__(self):
        'Denotes the total number of samples'
        if self.train:
            return len(ISIC2019._train_image_files)
        else:
            return len(ISIC2019._val_image_files)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.train:
            image = self.preproc(ISIC2019._train_image_files[index])
            cat = ISIC2019._train_image_categories[index]
            return {'image': image, 'category': cat}
        else:
            image = self.preproc(ISIC2019._val_image_files[index])
            cat = ISIC2019._val_image_categories[index]
            return {'image': image, 'category': cat}

    @classmethod
    def rand_split(
        cls, base_dir, train_imgs, train_gt, multiclass,
        split_ratio, seed,
    ):

        if multiclass:
            cats, files = cls.parse_files_as_multi_class(
                base_dir=base_dir,
                train_imgs=train_imgs,
                train_gt=train_gt,
            )

            # TO DO: add label encoding map?
        else:
            cats, files = cls.parse_files_as_binary_class(
                base_dir=base_dir,
                train_imgs=train_imgs,
                train_gt=train_gt,
            )

        train_files, val_files, train_cats, val_cats = train_test_split(
            files, cats,
            train_size=split_ratio,
            random_state=seed,
            stratify=cats,
        )

        cls._train_image_files = train_files
        cls._train_image_categories = train_cats
        cls._val_image_files = val_files
        cls._val_image_categories = val_cats
        cls._n_class = np.unique(cats).size

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
        """Parse ISIC data files as binary class (labeled as 0s or 1s)
        - parsing the 'image_name' and 'target' columns"""

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
    def parse_files_as_multi_class(base_dir, train_imgs, train_gt):
        """Parse ISIC data files as multi class - parsing the 'image_name'
        and 'diagnosis' columns"""

        cats, files = [], []
        data = sorted(
            os.listdir(os.path.join(base_dir, train_imgs)), key=lambda a: int(
                os.path.basename(a).split('_')[1].split('.')[0],
            ),
        )
        # Parse ISIC Ground Truth CSV file
        df = pd.read_csv(os.path.join(base_dir, train_gt))
        df = df[['image_name', 'diagnosis']]
        df.set_index('image_name', inplace=True)

        for file in data:
            file_path = os.path.join(base_dir, train_imgs, file)
            cat = df.loc[os.path.basename(file).split('.')[0]]['diagnosis']
            cats.append(cat)
            files.append(file_path)

        return np.array(cats), np.array(files)

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
            return ISIC2019._train_image_files
        else:
            return ISIC2019._val_image_files

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
            return ISIC2019._train_image_categories
        else:
            return ISIC2019._val_image_categories

    @property
    def n_class(self):
        """
        Return the number of distinct classes (in this case - 2)

        Returns
        -------
        n_classes : int
            Number of classes
        """
        return ISIC2019._n_class
