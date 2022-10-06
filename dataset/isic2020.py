import os
from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .factory import DatasetFactory


@DatasetFactory.register('ISIC2020')
class ISIC2020(Dataset):

    def __init__(
        self,
        base_dir=None,
        train_imgs=None,
        train_gt=None,
        preproc=Callable,
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
        preproc : Callable
            per-image preprocessing method
        """
        if not base_dir:
            base_dir = os.getenv('ISIC2020_BASE_FOLDER')
        if not train_imgs:
            train_imgs = os.getenv('ISIC2020_TRAIN_IMGS_FOLDER')
        if not train_gt:
            train_gt = os.getenv('ISIC2020_TRAIN_GT')

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

        self._image_files = files
        self._image_categories = cats
        self._n_class = len(set(self._image_categories))
        self.preproc = preproc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._image_files)

    def __getitem__(self, index):
        'Generates one sample of data'
        image = self.preproc(self._image_files[index])
        cat = self._image_categories[index]
        return {'image': image, 'category': cat}

    @property
    def image_files(self):
        """
        List of image files. The order of the list is important for other methods.

        Returns
        -------
        file_list : list(str)
            List of file names
        """
        return self._image_files

    @property
    def image_categories(self):
        """
        List of image categories. The order of the list is important for other methods.

        Returns
        -------
        image_categories : list(str)
            List of file categories
        """
        return self._image_categories

    @property
    def n_class(self):
        """
        Return the number of distinct classes (in this case - 2)

        Returns
        -------
        n_classes : int
            Number of classes
        """
        return self._n_class
