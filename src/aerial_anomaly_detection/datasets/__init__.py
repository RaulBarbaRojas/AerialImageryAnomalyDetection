"""
Module to implement general utilities for all datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch


class DataLoader(torch.utils.data.DataLoader):
    """
    Custom DataLoader implementation to obtain name and other relevant aspects of each tile.
    """


    def collate_fn(data):
        """
        Custom collate function for the custom DataLoader class.
        """
        sample_names, *data = zip(*data)
        return (sample_names,
                *(torch.utils.data.default_collate(data_piece) for data_piece in data))


    def __init__(self, *args : Tuple[Any,...], **kwargs : Dict[str, Any]) -> None:
        """
        Constructor method of the custom DataLoader class.

        Args:
            *args (Tuple[Any,...]): positional arguments of a normal PyTorch DataLoader.
            **kwargs (Dict[str, Any]): keyword arguments of a normal PyTorch DataLoader.
        """
        super().__init__(*args, **kwargs, collate_fn = DataLoader.collate_fn)


@dataclass
class Dataset(torch.utils.data.Dataset):
    """
    A class to provide access to a processed dataset of the Aerial Satellite Imagery Anomaly Detection project.
    """


    root_folder : Path
    index_df : pd.DataFrame
    partition : str


    def __post_init__(self) -> None:
        """
        Finish setting up the data access.
        """
        self.index_df = self.index_df[self.index_df['partition'] == self.partition]


    @classmethod
    def load(cls, root_folder : str | Path, partition : str) -> 'Dataset':
        """
        Method to load a partition of the dataset from its root path.

        Args:
            root_folder (str | Path): the root folder of the dataset.
            partition (str): the partition to be loaded.

        Returns:
            A new dataset with an index file containing information only from the specified partition.
        """
        root_folder = Path(root_folder)

        # Step 1: Loading tile metadata
        index_df = pd.read_csv(root_folder / 'tile_index.csv')

        # Step 2: Creating the new Dataset object
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

        # Step 3: Loading sample class
        tile_type = sample.tile_type

        return (sample.tile_path, img, tile_type)
