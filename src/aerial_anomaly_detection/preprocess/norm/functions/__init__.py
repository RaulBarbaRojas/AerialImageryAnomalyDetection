"""
Module to implement general utilities when working with normalization techniques.
"""


from abc import ABC, abstractmethod

from numpy.typing import NDArray

class Norm(ABC):
    """
    Class to implement a generic normalization function.
    """


    @abstractmethod
    def __call__(self, image : NDArray) -> NDArray:
        """
        Method to implement the application of the normalization.

        Args:
            image (NDArray): the image to be normalized.

        Returns:
            The normalized image.
        """
