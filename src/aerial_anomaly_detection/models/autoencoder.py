"""
Module to implement a convolutional autoencoder for anomaly detection on aerial imagery.
"""


import torch

from .utils import (Encoder,
                    Decoder)


class AutoEncoder(torch.nn.Module):
    """
    A class to implement the convolutional autoencoder anomaly detection method.
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int) -> None:
        """
        Constructor method of the AutoEncoder class.

        Args:
            latent_dimension (int): the size of the latent dimension of the autoencoder.
            img_width (int): the width of the input (and output) images.
            img_height (int): the height of the input (and output) images.
        """
        super().__init__()

        self.latent_dimension = latent_dimension
        self.img_width = img_width
        self.img_height = img_height

        self.encoder = Encoder(self.latent_dimension, self.img_width, self.img_height)
        self.decoder = Decoder(self.latent_dimension, self.img_width, self.img_height)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Method to implement the forward pass of the convolutional autoencoder.

        Args:
            x (Tensor): the input tensor of the convolutional autoencoder forward pass.

        Returns:
            The output tensor after the forward pass of the convolutional autoencoder.
        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x
