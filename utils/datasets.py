import json
import numpy as np
import os
import torch

class TrainDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, file_paths, labels, preproc):
        'Initialization'
        self.file_paths = file_paths
        self.labels = labels
        self.preproc = preproc
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        image = self.preproc(self.file_paths[index])
        label = self.labels[index]
        return {'image': image, 'label': label}


class ISIC2020:

    def __init__(self, base_dir, labels_path):
        """
        Load the image information from the drive
        
        Parameters
        ----------
        base_dir : string 
            ISIC2020 base directory path
        labels_path : str
            Path for the ISIC2020 labels JSON file
        """
        self._base_dir = base_dir
        
        cats, files = [], []

        data = sorted(os.listdir(base_dir), key=lambda a: int(os.path.basename(a).split("_")[1].split('.')[0]))
        with open(labels_path) as f:
            labels_map = json.load(f)
        
        for file in data:
            file_path = os.path.join(base_dir, file)
            cat = labels_map.get(os.path.basename(file).split(".")[0], None)
            cats.append(cat)
            files.append(file_path)

        cats, files = np.array(cats), np.array(files)
        
        self._image_files = files
        self._image_categories = cats


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