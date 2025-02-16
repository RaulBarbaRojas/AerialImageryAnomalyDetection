"""
Module to implement a generic tiler that could be overwritten to implement different tiling strategies.
"""


from abc import ABC, abstractmethod
from typing import Generator, Tuple

from numpy.typing import NDArray


class Tiler(ABC):
    """
    A class to implement a generic tiler.
    """


    def __init__(self, tile_x_size : int, tile_y_size : int, x_step : int, y_step : int) -> None:
        """
        Constructor method of the Tiler class.

        Args:
            tile_x_size (int): the size of the tile in the X axis.
            tile_y_size (int): the size of the tile in the Y axis.
            x_step (int): the step taken in the X coordinate (width).
            y_step (int): the step taken in the Y coordinate (height).
        """
        self.tile_x_size = tile_x_size
        self.tile_y_size = tile_y_size
        self.x_step = x_step
        self.y_step = y_step


    @abstractmethod
    def tile(self, image : NDArray) -> Generator[Tuple[NDArray, int, int], None, None]:
        """
        Method to tile an image.

        Args:
            image (NDArray): the NumPy array containing the image to be tiled.

        Returns:
            An iterator that yields tiles and their (x, y) coordinates.
        """
