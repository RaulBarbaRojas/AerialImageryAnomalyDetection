"""
Module to implement the Mean-Scale normalization (also known as Z-score normalization).
"""


from typing import Tuple

from numpy.typing import NDArray

from aerial_anomaly_detection.preprocess.norm.functions import Norm


class MeanScaleNorm(Norm):
    """
    A class to implement the mean-scale normalization.
    """


    def __init__(self, mean : Tuple[int, int, int], scale : Tuple[int, int, int]) -> None:
        """
        Constructor method.

        Args:
            mean (Tuple[int, int, int]): the mean to be used.
            scale (Tuple[int, int, int]): the scale to be used.
        """
        super().__init__()

        self.mean = mean
        self.scale = scale


    def __repr__(self) -> str:
        """
        Method to represent the MeanScaleNorm objects.

        Returns:
            A prettified string representation of the MeanScaleNorm objects.
        """
        return f'MeanScaleNorm(mean={self.mean}, scale={self.scale})'


    def __call__(self, image : NDArray) -> NDArray:
        """
        Method to apply mean-scale normalization over a given image.

        Args:
            image (NDArray): the image to be normalized. Planar image is expected.

        Returns:
            The normalized image with MeanScaleNormalization.
        """
        if image.shape[2] == 3:
            raise ValueError('[MeanScaleNorm] Expected a planar image, got an interleaved image (and potentially BGR?).')

        norm_image = image.copy()

        for color_channel in range(len(self.mean)):
            norm_image[color_channel,...] = (norm_image[color_channel] - self.mean[color_channel]) / self.scale[color_channel]

        return norm_image
