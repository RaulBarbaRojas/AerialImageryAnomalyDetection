"""
Module to implement a PyTorch dataset of MASATI (v2)
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class MASATIv2(torch.utils.data.Dataset):
    """
    A class to provide access to the MASATI (v2) dataset.
    """


    root_folder : Path
    index_df : pd.DataFrame
    partition : str


    def __post_init__(self) -> None:
        """
        Post-init method of the MASATI (v2) dataset.
        """
        self.index_df = self.index_df[self.index_df['partition'] == self.partition]


    @classmethod
    def load(cls, root_folder : str | Path, partition : str) -> 'MASATIv2':
        """
        Method to load the MASATI (v2) dataset from a given path.

        Args:
            root_folder (str | Path): the root folder of the dataset.
            partition (str): the partition to be loaded.
        """
        root_folder = Path(root_folder)

        # Step 1: Loading LandCover.ai data
        index_df = pd.read_csv(root_folder / 'tile_index.csv')

        # Step 2: Creating the LandCover.ai object
        return cls(root_folder = root_folder,
                   index_df = index_df,
                   partition = partition)


    def __len__(self) -> int:
        """
        Method to calculate the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return self.index_df.shape[0]


    def __getitem__(self, idx_item : int) -> Tuple[str, torch.Tensor]:
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
        img = np.fromfile(self.root_folder / sample.tile_path, dtype = np.float32).reshape((-1,
                                                                                            int(sample.tile_height),
                                                                                            int(sample.tile_width)))
        img = torch.from_numpy(img)

        # Step 3: Loading sample mask
        mask = np.fromfile(self.root_folder / sample.tile_mask_path, dtype = np.uint8).reshape((int(sample.tile_height),
                                                                                                int(sample.tile_width)))

        return (sample.tile_path, img, mask)
