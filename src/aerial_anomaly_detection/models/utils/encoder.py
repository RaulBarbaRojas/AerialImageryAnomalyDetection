"""
Module to implement a convolutional encoder with batch normalization. The hyperparameters related to the\
architecture were chosed based on the original paper.

Cite: Contreras-Cruz, M. A., Correa-Tome, F. E., Lopez-Padilla, R., & Ramirez-Paredes, J. P. (2023).\
Generative Adversarial Networks for anomaly detection in aerial images. Computers and Electrical Engineering, 106, 108470.
"""


import torch


class Encoder(torch.nn.Module):
    """
    A class to implement a convolutional encoder with batch normalization.
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int) -> None:
        """
        Constructor method for the Encoder class.

        Args:
            latent_dimension (int): the latent dimension to be used as an output.
            img_width (int): the width of the image.
            img_height (int): the height of the image.
        """
        super().__init__()

        self.features = torch.nn.Sequential(
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
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Linear(in_features = int(256 * (img_width / 4) * (img_height / 4)),
                            out_features = latent_dimension)
        self.tanh = torch.nn.Tanh()


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Method to implement the forward pass of the encoder module.

        Args:
            x (Tensor): the input tensor.

        Returns:
            The output tensor in the latent space.
        """
        x = self.features(x)
        x = self.fc(x)
        x = self.tanh(x)

        return x
