"""
Module to implement general utilities for all datasets.
"""


from typing import Any, Dict, Tuple

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
