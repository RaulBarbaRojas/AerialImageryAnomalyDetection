"""
A module to implement the BiGAN anomaly detection method.

Cite: Donahue, J., Krähenbühl, P., & Darrell, T. (2016). Adversarial feature learning. arXiv preprint arXiv:1605.09782.
"""


from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from munch import Munch

from aerial_anomaly_detection.models.utils import Encoder, Decoder as Generator


class BiGANDiscriminator(torch.nn.Module):
    """
    A class to implement the discriminator used in the BiGAN architecture.
    """


    def __init__(self, latent_dimension : int, img_width : int,
                 img_height : int, **model_settings : Dict[str, Any]) -> None:
        """
        Constructor method of the BiGANDiscriminator class.

        Args:
            latent_dimension (int): the latent dimension of the generated encodings.
            img_width (int): the width of the image.
            img_height (int): the height of the image.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """
        super().__init__()

        self.model_settings = Munch(model_settings)
        self.image_features = torch.nn.Sequential(
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
            torch.nn.Flatten()
        )
        self.encoding_features = torch.nn.Linear(in_features = latent_dimension, out_features = 512)
        self.global_features = torch.nn.Sequential(
            torch.nn.Linear(in_features = 128 * (img_width // 4) * (img_height // 4) + 512,
                            out_features = 1024),
            torch.nn.LeakyReLU()
        )
        self.classification_layer = torch.nn.Linear(in_features = 1024, out_features = 1)
        self.head = torch.nn.Sigmoid()


    def forward(self, img_tensor : torch.Tensor, encoding_tensor : torch.Tensor,
                mode : Literal['train', 'inference'] = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to implement the forward pass of the BiGAN.

        Args:
            img_tensor (torch.Tensor): the input tensor containing a batch of images.
            encoding_tensor (torch.Tensor): the input tensor containing the batch of encodings linked to the images.
            mode (Literal['train', 'inference']): whether the model is being used for training (raw logits) or inference (sigmoid).

        Returns:
            A tuple (y_pred, features) containing the prediction performed by the BiGAN Discriminator (raw logits\
            during training and probabilities during inference) and the features of the last layer of the discriminator.
        """

        # Step 1: Applying layers separately
        img_tensor = self.image_features(img_tensor)
        encoding_tensor = self.encoding_features(encoding_tensor)

        # Step 2: Merging intermediate features
        global_features = torch.cat([img_tensor, encoding_tensor], dim = 1)
        global_features = self.global_features(global_features)

        # Step 3: Calculating the final prediction (raw logit // probability based on mode)
        y_pred = self.classification_layer(global_features)
        if mode == 'inference':
            y_pred = self.head(y_pred)

        return y_pred, global_features


class BiGAN(torch.nn.Module):
    """
    A class to implement the BiGAN forward pass (mainly for inference).
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int,
                 pretrained_encoder_weights_path : Optional[str | Path],
                 pretrained_generator_weights_path : Optional[str | Path],
                 pretrained_discriminator_weights_path : Optional[str | Path],
                 **model_settings : Dict[str, Any])  -> None:
        """
        Constructor method for the BiGAN class.

        Args:
            latent_dimension (int): the latent dimension of the generated encodings.
            img_width (int): the width of the image.
            img_height (int): the height of the image.
            pretrained_encoder_weights_path (Optional[str | Path]): the file containing the pre-trained encoder weights.
            pretrained_generator_weights_path (Optional[str | Path]): the file containing the pre-trained generator weights.
            pretrained_discriminator_weights_path (Optional[str | Path]): the file containing the pre-trained discriminator weights.
            model_settings (Dict[str, Any]): a dictionary containing other settings relevant for using the model in the general workflow.
        """
        super().__init__()

        self.encoder = Encoder(latent_dimension, img_width, img_height)
        self.generator = Generator(latent_dimension, img_width, img_height)
        self.discriminator = BiGANDiscriminator(latent_dimension, img_width, img_height)
        self.model_settings = Munch(model_settings)

        if pretrained_encoder_weights_path:
            encoder_keys = self.encoder.load_state_dict(torch.load(pretrained_encoder_weights_path, weights_only = True))
            print('Encoder weights: ', encoder_keys)

        if pretrained_generator_weights_path:
            generator_keys = self.generator.load_state_dict(torch.load(pretrained_generator_weights_path, weights_only = True))
            print('Generator weights: ', generator_keys)

        if pretrained_discriminator_weights_path:
            discriminator_keys = self.discriminator.load_state_dict(torch.load(pretrained_discriminator_weights_path, weights_only = True))
            print('Discriminator weights: ', discriminator_keys)


    def forward(self, x : torch.Tensor,
                mode : Literal['train', 'inference'] = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to implement the forward pass of the BiGAN (inference-only, see training scripts for training forward pass).

        Args:
            x (torch.Tensor): the input tensor of the BiGAN when running inference.
            mode (Literal['train', 'inference']): whether the model is being used for training (raw logits) or inference (sigmoid).

        Returns:
            A tuple (G(E(x), f(x, E(x)), f(G(E(x)), E(x))) where the first element represents the reconstructed image,\
                the second element represents the last layer of the discriminator when receiving the real image and the\
                encoding predicted by the encoder, and the last element represents the last layer of the discriminator when\
                receiving the reconstructed input and the encoding predicted by the encoder.
        """
        encoded_image = self.encoder(x)
        reconstructed_image = self.generator(encoded_image)
        _, ground_truth_features = self.discriminator(x, encoded_image, mode)
        _, reconstructed_features = self.discriminator(reconstructed_image, encoded_image, mode)

        return reconstructed_image, ground_truth_features, reconstructed_features
