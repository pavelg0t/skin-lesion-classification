import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .factory import DatasetFactory


@DatasetFactory.register('ISIC2020')
class ISIC2020(Dataset):

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
        base_dir : string
            ISIC2020 base directory path
        train_imgs : string
            ISIC2020 train images folder (relative to the base_dir)
        train_gt : string
            ISIC2020 train data ground truth
        train : bool
            Return train dataset (validation if False)
        split_ratio : float
            Train data ration used for data spliting
        seed : int
            Seed value for random generators
        preproc : Callable
            per-image preprocessing method
        """

        if not base_dir:
            base_dir = os.getenv('ISIC2020_BASE_FOLDER')
        if not train_imgs:
            train_imgs = os.getenv('ISIC2020_TRAIN_IMGS_FOLDER')
        if not train_gt:
            train_gt = os.getenv('ISIC2020_TRAIN_GT')

        self.train = train
        self.preproc = preproc

        if not ISIC2020.is_data_initially_split():
            ISIC2020.rand_split(
                base_dir=base_dir,
                train_imgs=train_imgs,
                train_gt=train_gt,
                split_ratio=split_ratio,
                seed=seed,
            )

    def __len__(self):
        'Denotes the total number of samples'
        if self.train:
            return len(ISIC2020._train_image_files)
        else:
            return len(ISIC2020._val_image_files)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.train:
            image = self.preproc(ISIC2020._train_image_files[index])
            cat = ISIC2020._train_image_categories[index]
            return {'image': image, 'category': cat}
        else:
            image = self.preproc(ISIC2020._val_image_files[index])
            cat = ISIC2020._val_image_categories[index]
            return {'image': image, 'category': cat}

    @classmethod
    def rand_split(cls, base_dir, train_imgs, train_gt, split_ratio, seed):

        cats, files = [], []
        data = sorted(
            os.listdir(os.path.join(base_dir, train_imgs)), key=lambda a: int(
                os.path.basename(a).split('_')[1].split('.')[0],
            ),
        )
        # Parse ISIC2020 Ground Truth CSV file
        df = pd.read_csv(os.path.join(base_dir, train_gt))
        df = df[['image_name', 'target']]
        df.set_index('image_name', inplace=True)

        for file in data:
            file_path = os.path.join(base_dir, train_imgs, file)
            cat = df.loc[os.path.basename(file).split('.')[0]]['target']
            cats.append(cat)
            files.append(file_path)

        cats, files = np.array(cats), np.array(files)

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
            return ISIC2020._train_image_files
        else:
            return ISIC2020._val_image_files

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
            return ISIC2020._train_image_categories
        else:
            return ISIC2020._val_image_categories

    @property
    def n_class(self):
        """
        Return the number of distinct classes (in this case - 2)

        Returns
        -------
        n_classes : int
            Number of classes
        """
        return ISIC2020._n_class
