"""
Module to implement the "overlap tiler" functionality.
"""

from typing import Generator, Tuple

from numpy.typing import NDArray

from aerial_anomaly_detection.preprocess.tile import Tiler


class OverlapTiler(Tiler):
    """
    A class to implement the overlap tiler.
    """


    def tile(self, image : NDArray) -> Generator[Tuple[NDArray, int, int], None, None]:
        """
        Method to implement the overlap tiling strategy.

        Args:
            image (NDArray): the image to be tiled in interleaved format (HxWxC).

        Returns:
            A generator that yields overlapped tiles and their (x, y) coordinates.
        """
        for y_coord in range(0, image.shape[0], self.y_step):
            for x_coord in range(0, image.shape[1], self.x_step):
                if x_coord + self.tile_x_size >= image.shape[1]:
                    break

                yield image[y_coord:y_coord + self.tile_y_size, x_coord:x_coord + self.tile_x_size], x_coord, y_coord

            if y_coord + self.tile_y_size >= image.shape[0]:
                break
