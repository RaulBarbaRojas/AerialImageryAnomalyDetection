"""
Module to implement the discriminator of a DCGAN.

Cite: Contreras-Cruz, M. A., Correa-Tome, F. E., Lopez-Padilla, R., & Ramirez-Paredes, J. P. (2023).\
Generative Adversarial Networks for anomaly detection in aerial images. Computers and Electrical Engineering, 106, 108470.

PyTorch tutorial was also used to understand DCGAN concepts: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""


from typing import Dict, Any

import torch
from munch import Munch


class Discriminator(torch.nn.Module):
    """
    Class to implement the discriminator of a basic DCGAN (BiGAN not supported).
    """


    def __init__(self, img_width : int, img_height : int, use_sigmoid_head : bool = False,
                 **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method for the Discriminator class.

        Args:
            img_width (int): the width of the image.
            img_height (int): the height of the image.
            use_sigmoid_head (bool): a flag to use sigmoid head or not.
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
        self.head = torch.nn.Sigmoid() if use_sigmoid_head else None


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Method to implement the forward pass of the discriminator.

        Args:
            x (Tensor): the input tensor.

        Returns:
            The resulting tensor after performing the forward pass of the discriminator.
        """
        x = self.features(x)

        if self.head is not None:
          x = self.head(x)

        return x
