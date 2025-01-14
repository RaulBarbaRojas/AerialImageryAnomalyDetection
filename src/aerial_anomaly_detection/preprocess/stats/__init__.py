"""
Module with stats calculation utilities for all datasets.
"""


import numpy as np
from numpy.typing import NDArray


class StatsCalculator:
    """
    Class to calculate certain statistics of a given dataset.
    """


    def __init__(self) -> None:
        """
        Method to create a stats calculator object.
        """
        self.pixel_sum_per_channel : NDArray = np.zeros(3)
        self.squared_pixel_sum_per_channel : NDArray = np.zeros(3)
        self.pixel_count : int = 0


    @property
    def mean(self) -> NDArray:
        """
        Property to return the mean value of each color channel based on the seen images.
        """
        return self.pixel_sum_per_channel / self.pixel_count


    @property
    def std(self) -> NDArray:
        """
        Property to return the standard deviation of each color channel based on the seen images.
        """
        variance = self.squared_pixel_sum_per_channel / self.pixel_count - self.mean ** 2
        return np.sqrt(variance)


    def update(self, image : NDArray) -> None:
        """
        Method to update the stats calculator with a new image.

        Args:
            image (NDArray): the image with which stats update will be run. Planar RGB image expected.
        """
        if image.shape[2] == 3:
            raise ValueError('[StatsCalculator] Planar/RGB image required. Got interleaved (possibly BGR?)')

        image = image.astype(np.float64)

        for color_channel in range(3):
            self.pixel_sum_per_channel[color_channel] += image[color_channel,...].sum()
            self.squared_pixel_sum_per_channel[color_channel] += (image[color_channel,...] ** 2).sum()

        self.pixel_count += image.shape[1] * image.shape[2]