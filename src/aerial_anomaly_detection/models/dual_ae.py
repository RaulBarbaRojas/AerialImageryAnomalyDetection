"""
Module to implement a Dual AE for anomaly detection on satellite imagery.
"""


from pathlib import Path
from typing import Any, Dict, Literal

import torch
from munch import Munch


class HighLevelAnoDAE(torch.nn.Module):
    """
    A class to define an AE with large receptive field (High Level Anomaly Detection AE).
    """


    def __init__(self, latent_dimension: int, img_width : int, img_height : int,
                 **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method of the HighLevelAnoDAE class.

        Args:
            latent_dimension (int): the size of the latent dimension of the HighLevelAnoDAE architecture.
            img_width (int): the width of the input (and output) images.
            img_height (int): the height of the input (and output) images.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """

        # Step 1: Setting up the network as a PyTorch module
        super().__init__()

        # Step 2: Defining model settings (could be required in train/eval) and its features
        self.model_settings = Munch(model_settings)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3,
                            out_channels = 64,
                            kernel_size = 5,
                            stride = 2,
                            padding = 2),
            torch.nn.BatchNorm2d(num_features = 64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels = 64,
                            out_channels = 128,
                            kernel_size = 5,
                            stride = 2,
                            padding = 2),
            torch.nn.BatchNorm2d(num_features = 128),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels = 128,
                            out_channels = 256,
                            kernel_size = 5,
                            stride = 1,
                            padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 256),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 256 * (img_width // 4) * (img_height // 4),
                            out_features = latent_dimension),
            torch.nn.Tanh()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features = latent_dimension,
                            out_features = 16 * img_width * img_height),
            torch.nn.BatchNorm1d(16 * img_width * img_height),
            torch.nn.LeakyReLU(),
            torch.nn.Unflatten(1, (256, img_height // 4, img_width // 4)),

            torch.nn.ConvTranspose2d(in_channels = 256,
                                     out_channels = 128,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 2),
            torch.nn.BatchNorm2d(num_features = 128),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels = 128,
                                     out_channels = 64,
                                     kernel_size = 5,
                                     stride = 2,
                                     padding = 2,
                                     output_padding = 1),
            torch.nn.BatchNorm2d(num_features = 64),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels = 64,
                                     out_channels = 3,
                                     kernel_size = 5,
                                     stride = 2,
                                     padding = 2,
                                     output_padding = 1),
        )

    def forward(self, x : torch.Tensor, mode : Literal['train', 'test'] = 'train') -> torch.Tensor:
        """
        Forward pass of the High Level AnoDAE.

        Args:
            x (torch.Tensor): the input tensor of images to be reconstructed with the High Level AnoDAE.
            mode (Literal['train', 'test']): whether the model will be used for training or inference.

        Returns:
            A PyTorch Tensor containing the reconstructed image/s, which can be used for Anomaly Detection.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LowLevelAnoDAE(torch.nn.Module):
    """
    A class to define an AE with small receptive field (Low Level Anomaly Detection AE).
    """


    def __init__(self, latent_dimension: int, img_width : int, img_height : int,
                 **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method of the LowLevelAnoDAE class.

        Args:
            latent_dimension (int): the size of the latent dimension of the LowLevelAnoDAE architecture.
            img_width (int): the width of the input (and output) images.
            img_height (int): the height of the input (and output) images.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """

        # Step 1: Setting up the network as a PyTorch module
        super().__init__()

        # Step 2: Defining model settings (could be required in train/eval) and its features
        self.model_settings = Munch(model_settings)

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3,
                            out_channels = 64,
                            kernel_size = 3,
                            stride = 2,
                            padding = 1),
            torch.nn.BatchNorm2d(num_features = 64),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels = 64,
                            out_channels = 128,
                            kernel_size = 3,
                            stride = 2,
                            padding = 1),
            torch.nn.BatchNorm2d(num_features = 128),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels = 128,
                            out_channels = 256,
                            kernel_size = 3,
                            stride = 1,
                            padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 256),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 256 * (img_width // 4) * (img_height // 4),
                            out_features = latent_dimension),
            torch.nn.Tanh()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features = latent_dimension,
                            out_features = 16 * img_width * img_height),
            torch.nn.BatchNorm1d(16 * img_width * img_height),
            torch.nn.LeakyReLU(),
            torch.nn.Unflatten(1, (256, img_height // 4, img_width // 4)),

            torch.nn.ConvTranspose2d(in_channels = 256,
                                     out_channels = 128,
                                     kernel_size = 3,
                                     stride = 2,
                                     padding = 1,
                                     output_padding = 1),
            torch.nn.BatchNorm2d(num_features = 128),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels = 128,
                                     out_channels = 64,
                                     kernel_size = 3,
                                     stride = 2,
                                     padding = 1,
                                     output_padding = 1),
            torch.nn.BatchNorm2d(num_features = 64),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels = 64,
                                     out_channels = 3,
                                     kernel_size = 3,
                                     stride = 1,
                                     padding = 1),
        )


    def forward(self, x : torch.Tensor, mode : Literal['train', 'test'] = 'train') -> torch.Tensor:
        """
        Forward pass of the Low Level AnoDAE.

        Args:
            x (torch.Tensor): the input tensor of images to be reconstructed with the Low Level AnoDAE.
            mode (Literal['train', 'test']): whether the model will be used for training or inference.

        Returns:
            A PyTorch Tensor containing the reconstructed image/s, which can be used for Anomaly Detection.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DualAnoDAE(torch.nn.Module):
    """
    A class to implement an ensemble of 2 AEs for Anomaly Detection.
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int,
                 low_level_anodae_weights : str | Path,
                 high_level_anodae_weights : str | Path,
                 **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method for the DualAnoDAE class.

        Args:
            latent_dimension (int): the size of the latent dimension of the DualAnoDAE architecture.
            img_width (int): the width of the input (and output) images.
            img_height (int): the height of the input (and output) images.
            low_level_anodae_weights (str | Path): 
            high_level_anodae_weights (str | Path): 
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model\
                in the general workflow.
        """
        super().__init__()

        # Step 1: Setting up the ensemble AE
        self.low_level_anodae = LowLevelAnoDAE(latent_dimension, img_width, img_height)
        self.high_level_anodae = HighLevelAnoDAE(latent_dimension, img_width, img_height)
        self.model_settings = Munch(model_settings)

        # Step 2: Loading the pre-trained weights
        self.low_level_anodae.load_state_dict(torch.load(low_level_anodae_weights, weights_only = True))
        self.high_level_anodae.load_state_dict(torch.load(high_level_anodae_weights, weights_only = True))


    def forward(self, x : torch.Tensor, mode : Literal['train', 'test'] = 'train') -> torch.Tensor:
        """
        Forward pass of the DualAnoDAE class.

        Args:
            x (torch.Tensor): the input tensor to perform anomaly detection on.
            mode (Literal['train', 'test']): whether the model will be used for training or inference.

        Returns:
            A tuple of two reconstructed images, where the first one comes from the low level AnoDAE, and the\
            second one comes from the high level AnoDAE.
        """
        low_level_reconstructed_x = self.low_level_anodae(x)
        high_level_reconstructed_x = self.high_level_anodae(x)

        return low_level_reconstructed_x, high_level_reconstructed_x
