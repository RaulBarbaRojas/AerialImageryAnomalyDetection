"""
Module to implement the overlapped tiler strategy (overlapped tiling).
"""

from typing import Generator, Tuple

from numpy.typing import NDArray

from aerial_anomaly_detection.preprocess.tile.tilers import Tiler


class OverlappedTiler(Tiler):
    """
    A class to implement the overlapped tiler.
    """


    def tile(self, image : NDArray, mask : NDArray) -> Generator[Tuple[NDArray, NDArray, int, int], None, None]:
        """
        Method to implement the overlapped tiling strategy.

        Args:
            image (NDArray): the image to be tiled in the format (CxHxW).
            mask (NDArray): the mask to be tiled in the format (HxW).

        Returns:
            A generator that yields tiles and their (x, y) coordinates.
        """
        for y_coord in range(0, image.shape[1], self.y_step):
            for x_coord in range(0, image.shape[2], self.x_step):
                if x_coord + self.tile_x_size > image.shape[2]:
                    break

                yield (image[:, y_coord : y_coord + self.tile_y_size, x_coord : x_coord + self.tile_x_size],
                       mask[y_coord : y_coord + self.tile_y_size, x_coord : x_coord + self.tile_x_size],
                       x_coord,
                       y_coord)

            if y_coord + self.tile_y_size > image.shape[1]:
                break
