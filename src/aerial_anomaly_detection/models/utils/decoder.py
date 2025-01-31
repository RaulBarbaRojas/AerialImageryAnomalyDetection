"""
Module to implement a convolutional decoder (generator) with batch normalization. The hyperparameters related to the\
architecture were chosed based on the original paper.

Cite: Contreras-Cruz, M. A., Correa-Tome, F. E., Lopez-Padilla, R., & Ramirez-Paredes, J. P. (2023).\
Generative Adversarial Networks for anomaly detection in aerial images. Computers and Electrical Engineering, 106, 108470.
"""


import torch


class Decoder(torch.nn.Module):
    """
    A class to implement a convolutional decoder (generator) with batch normalization.
    """


    def __init__(self, latent_dimension : int, img_width : int, img_height : int) -> None:
        """
        Constructor method for the Decoder class.

        Args:
            latent_dimension (int): the latent dimension to be used as an output.
            img_width (int): the width of the image.
            img_height (int): the height of the image.
        """
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.fc = torch.nn.Linear(in_features = latent_dimension,
                                  out_features = 16 * img_width * img_height)
        self.lr = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm1d(16 * img_width * img_height)
        self.features = torch.nn.Sequential(
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
        self.tanh = torch.nn.Tanh()


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Method to implement the forward pass of the decoder.

        Args:
            x (Tensor): the input tensor.

        Returns:
            The output tensor of the decoder (img of expected size).
        """
        x = self.lr(self.bn(self.fc(x)))
        x = x.view(-1, 256, self.img_height // 4, self.img_width // 4)
        x = self.features(x)
        x = self.tanh(x)

        return x

