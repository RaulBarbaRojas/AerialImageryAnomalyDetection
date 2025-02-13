"""
Module to implement the izi architecture for anomaly detection on aerial imagery.

Cite: Contreras-Cruz, M. A., Correa-Tome, F. E., Lopez-Padilla, R., & Ramirez-Paredes, J. P. (2023).\
Generative Adversarial Networks for anomaly detection in aerial images. Computers and Electrical Engineering, 106, 108470.
"""


from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import torch
from munch import Munch

from aerial_anomaly_detection.models.utils import Encoder, Decoder, Discriminator


class FAnoGAN(torch.nn.Module):
    """
    A class to implement the f-AnoGAN architecture.
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int,
                 pretrained_decoder_weights_path : str | Path ,
                 pretrained_discriminator_weights_path : str | Path ,
                 **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method of the Izi class.

        Args:
            latent_dimension (int): the size of the latent dimension of the izi architecture.
            img_width (int): the width of the input (and output) images.
            img_height (int): the height of the input (and output) images.
            pretrained_decoder_weights_path (str | Path): the pre-trained weights of the decoder.
            pretrained_discriminator_weights_path (str | Path): the pre-trained weights of the discriminator.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """

        # Step 1: Setting up the PyTorch module
        super().__init__()

        # Step 2: Setting up the f-AnoGAN model
        self.encoder = Encoder(latent_dimension, img_width, img_height)
        self.decoder = Decoder(latent_dimension, img_width, img_height)
        self.discriminator = Discriminator(img_width, img_height)
        self.model_settings = Munch(model_settings)

        # Step 3: Loading pre-trained decoder weights and disabling decoder learning (f-AnoGAN architecture specifics)
        self.decoder.load_state_dict(torch.load(pretrained_decoder_weights_path, weights_only = True))

        for parameter in self.decoder.parameters():
            parameter.requires_grad = False

        # Step 3: Loading pre-trained decoder weights and disabling decoder learning (f-AnoGAN architecture specifics)
        self.discriminator.load_state_dict(torch.load(pretrained_discriminator_weights_path, weights_only = True))

        for parameter in self.discriminator.parameters():
            parameter.requires_grad = False


    def forward(self, x : torch.Tensor,
                mode : Literal['train', 'test'] = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to carry out the forward pass of the f-AnoGAN model.

        Args:
            x (torch.Tensor): the input tensor to be used for the forward pass of the model.
            mode (Literal['train', 'test']): whether the model will be used for training or inference.

        Returns:
            A tuple (G(E(x)), f(x), f(G(E(x)))) containing the reconstructed image, the features\
            of the discriminator for the input tensor, and the features of the discriminator for the\
            reconstructed image.
        """
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)
        _, x_features = self.discriminator(x, mode = mode, retrieve_features = True)
        _, x_reconstructed_features = self.discriminator(x_reconstructed, mode = mode, retrieve_features = True)

        return x_reconstructed, x_features, x_reconstructed_features
