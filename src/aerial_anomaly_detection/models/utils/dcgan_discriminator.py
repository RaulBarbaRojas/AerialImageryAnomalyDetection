"""
Module to implement the discriminator of a DCGAN.

Cite: Contreras-Cruz, M. A., Correa-Tome, F. E., Lopez-Padilla, R., & Ramirez-Paredes, J. P. (2023).\
Generative Adversarial Networks for anomaly detection in aerial images. Computers and Electrical Engineering, 106, 108470.

PyTorch tutorial was also used to understand DCGAN concepts: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""


from typing import Any, Dict, Literal

import torch
from munch import Munch


class Discriminator(torch.nn.Module):
    """
    Class to implement the discriminator of a basic DCGAN (BiGAN not supported).
    """


    def __init__(self, img_width : int, img_height : int, **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method for the Discriminator class.

        Args:
            img_width (int): the width of the image.
            img_height (int): the height of the image.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """
        super().__init__()

        self.model_settings = Munch(model_settings)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3,
                            out_channels = 64,
                            kernel_size = 5,
                            stride = 2,
                            padding = 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels = 64,
                            out_channels = 128,
                            kernel_size = 5,
                            stride = 2,
                            padding = 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 128 * (img_width // 4) * (img_height // 4),
                            out_features = 1)
        )
        self.head = torch.nn.Sigmoid()


    def forward(self, x : torch.Tensor, mode : Literal['train', 'inference'] = 'train',
                retrieve_features : bool = False) -> torch.Tensor:
        """
        Method to implement the forward pass of the discriminator.

        Args:
            x (Tensor): the input tensor.
            mode (Literal['train', 'inference']): whether the model is being used for training (raw logits) or inference (sigmoid).
            retrieve_features (bool): whether discriminator features will be stored.

        Returns:
            The resulting tensor after performing the forward pass of the discriminator.
        """
        features = self.features[:-1](x)
        x = self.features[-1](features)

        if mode == 'inference':
          x = self.head(x)

        output = x if retrieve_features is False else (x, features)

        return output
