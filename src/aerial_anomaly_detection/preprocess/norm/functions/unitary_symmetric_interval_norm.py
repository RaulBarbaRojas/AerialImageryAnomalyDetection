"""
Module to implement a normalization so that images fall in the range [-1, 1]
"""


from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from aerial_anomaly_detection.preprocess.norm.functions import Norm


class UnitarySymmetricIntervalNorm(Norm):
    """
    A class to implement the [-1, 1] normalization.
    """


    def __init__(self, scale : int = 255) -> None:
        """
        Constructor method for the UnitarySymmetricIntervalNorm class.

        Args:
            scale (int): the scale to be used when normalizing the input.
        """
        self.scale = scale


    def __repr__(self) -> str:
        """
        Method to represent the MeanScaleNorm objects.

        Returns:
            A prettified string representation of the MeanScaleNorm objects.
        """
        return f'UnitarySymmetricIntervalNorm(scale = {self.scale})'


    def __call__(self, image : NDArray) -> NDArray:
        """
        Method to apply [-1, 1] normalization across the input image.

        Args:
            image (NDArray): the image to be normalized. Planar image is expected.

        Returns:
            The normalized image with UnitarySymmetricIntervalNorm.
        """
        if image.shape[2] == 3:
            raise ValueError('[UnitarySymmetricIntervalNorm] Expected a planar image, got an interleaved image (and potentially BGR?).')

        norm_image = image.copy().astype(np.float32)
        norm_image = norm_image / 255
        norm_image = 2 * norm_image - 1

        return norm_image

