"""
A module to implement the MASATI (v2) dataset as a usable PyTorch dataset.
"""


from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from aerial_anomaly_detection.preprocess.norm.functions.mean_scale_norm import (Norm,
                                                                                MeanScaleNorm)
from aerial_anomaly_detection.utils.dataset_utils import str_to_value_list


@dataclass
class MASATIv2(torch.utils.data.Dataset):
    """
    A class to provide access to the MASATI (v2) dataset.
    """


    root_folder : Path
    index_df : pd.DataFrame
    partition : str
    img_norm : Norm | None


    def __post_init__(self) -> None:
        """
        Post-init method of the MASATI (v2) dataset.
        """
        self.index_df = self.index_df[self.index_df.partition == self.partition]


    @classmethod
    def load(cls, root_folder : str | Path, partition : str) -> 'MASATIv2':
        """
        Method to load the MASATI (v2) dataset from a given path.

        Args:
            root_folder (str | Path): the root folder of the dataset.
            partition (str): the partition to be loaded.
        """
        root_folder = Path(root_folder)

        # Step 1: Loading MASATI (v2) data
        index_df = pd.read_csv(root_folder / 'index.csv')
        (info_file := ConfigParser()).read(root_folder / 'info.ini')
        img_norm = MeanScaleNorm(mean = str_to_value_list(info_file['stats']['mean']),
                                 scale = str_to_value_list(info_file['stats']['scale']))

        # Step 2: Creating the MASATI (v2) object
        return cls(root_folder = root_folder,
                   index_df = index_df,
                   partition = partition,
                   img_norm = img_norm)


    def __len__(self) -> int:
        """
        Method to calculate the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return self.index_df.shape[0]


    def __getitem__(self, idx_item : int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Method to get an item of the dataset by numerical position.

        Args:
            idx_item (int): the numerical position of the item to be retrieved.

        Returns:
            A tuple containing the name of the retrieved sample, a tensor with its data, and a tensor with its mask.
        """

        # Step 1: Loading sample name
        sample = self.index_df.iloc[idx_item]

        # Step 2: Loading sample data
        img = np.fromfile(self.root_folder / sample.norm_path, dtype = np.float64).reshape((-1,
                                                                                            int(sample.tile_x_size),
                                                                                            int(sample.tile_y_size)))

        # Step 3: Loading sample mask
        mask = cv2.imread(self.root_folder / sample.mask_path, cv2.IMREAD_GRAYSCALE)

        return (Path(sample.path).name, img, mask)
