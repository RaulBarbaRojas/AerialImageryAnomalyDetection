"""
Module to implement the ziz architecture for anomaly detection on aerial imagery.

Cite: Contreras-Cruz, M. A., Correa-Tome, F. E., Lopez-Padilla, R., & Ramirez-Paredes, J. P. (2023).\
Generative Adversarial Networks for anomaly detection in aerial images. Computers and Electrical Engineering, 106, 108470.
"""


from pathlib import Path
from typing import Any, Dict, Literal

import torch

from aerial_anomaly_detection.models.autoencoder import AutoEncoder


class Ziz(AutoEncoder):
    """
    A class to implement the ziz architecture.
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int,
                 pretrained_decoder_weights_path : str | Path , **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method of the Ziz class.

        Args:
            latent_dimension (int): the size of the latent dimension of the ziz architecture.
            img_width (int): the width of the input (and output) images.
            img_height (int): the height of the input (and output) images.
            pretrained_decoder_weights_path (str | Path): the pre-trained weights of the decoder.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """

        # Step 1: Setting up the Ziz architecture
        super().__init__(latent_dimension = latent_dimension,
                         img_width = img_width,
                         img_height = img_height,
                         **model_settings)

        # Step 2: Loading pre-trained decoder weights and disabling decoder learning (izi architecture specifics)
        self.decoder.load_state_dict(torch.load(pretrained_decoder_weights_path, weights_only = True))

        for parameter in self.decoder.parameters():
            parameter.requires_grad = False


    def forward(self, x : torch.Tensor, mode : Literal['train', 'inference'], **kwargs : Dict[str, Any]) -> torch.Tensor:
        """
        Method to implement the forward pass of the Ziz architecture.

        Args:
            x (torch.Tensor): the input tensor of the ziz architecture.
            mode (Literal): whether Ziz will be used training (`train`) or inference (`inference`).
            **kwargs (Dict[str, Any]): extra unused parameters required due to the generic implementation of train/test scripts.

        Returns:
            The output tensor after the forward pass of the ziz architecture.
        """
        assert mode == 'train' or mode == 'inference'

        if mode == 'train':
            x = self.decoder(x)
            x = self.encoder(x)
        else:
            x = self.encoder(x)
            x = self.decoder(x)

        return x
