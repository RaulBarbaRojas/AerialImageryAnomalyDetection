"""
Module to implement a convolutional autoencoder for anomaly detection on aerial imagery.
"""


from typing import Any, Dict

import torch
from munch import Munch

from aerial_anomaly_detection.models.utils import Encoder, Decoder


class AutoEncoder(torch.nn.Module):
    """
    A class to implement the convolutional autoencoder anomaly detection method.
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int, **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method of the AutoEncoder class.

        Args:
            latent_dimension (int): the size of the latent dimension of the autoencoder.
            img_width (int): the width of the input (and output) images.
            img_height (int): the height of the input (and output) images.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """
        super().__init__()

        self.latent_dimension = latent_dimension
        self.img_width = img_width
        self.img_height = img_height
        self.model_settings = Munch(model_settings)

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
