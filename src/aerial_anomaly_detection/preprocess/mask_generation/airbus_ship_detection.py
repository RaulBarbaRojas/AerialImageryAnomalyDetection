"""
Mask generation script for the Airbus Ship Detection dataset.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from afml.context import run_ctx
from numpy.typing import NDArray
from tqdm.auto import tqdm


def rle_ship_encoding_to_mask(rle_encoding : str, img_width : int, img_height : int) -> NDArray:
    """
    Function to transform a given RLE encoding to its corresponding mask.

    Args:
        rle_encoding (str): the RLE encoding of the ship/s to be considered in the mask.
        img_width (int): the width of the input image whose RLE encoding is given (used for mask generation).
        img_height (int): the height of the input image whose RLE encoding is given (used for mask generation).

    Returns:
        A Numpy array in uint8 format with the same width and height as the input image and with 255 values in ship locations (0 elsewhere).
    """
    mask = np.zeros(img_width * img_height, dtype = np.uint8)

    # Step 1: Check if RLE encoding only is available
    if len(rle_encoding) > 0:

        # Step 2: Transform the RLE string into an actual list of RLE encodings
        rle_encoding = [int(encoding_number) for encoding_number in rle_encoding.split(' ')]

        # Step 3: Generate the mask from the encodings
        for idx_pixel_row in range(0, len(rle_encoding), 2):
            start_index = rle_encoding[idx_pixel_row]
            pixel_length = rle_encoding[idx_pixel_row + 1]
            mask[start_index:start_index + pixel_length] = 255

    return mask.reshape((img_height, img_width)).T


if __name__ == '__main__':
    # Step 1: Loading ship RLE encodings CSV
    ship_encodings_df = pd.read_csv(run_ctx.dataset.folder / 'train_ship_segmentations_v2.csv')
    ship_encodings_df['EncodedPixels'] = ship_encodings_df['EncodedPixels'].fillna('')
    ship_encodings_df = ship_encodings_df.groupby(by = 'ImageId')['EncodedPixels'].apply(' '.join).reset_index()

    # Step 2: Generating masks from RLE encodings
    (mask_folder := Path(run_ctx.params.mask_output_folder)).mkdir(exist_ok = True, parents = True)
    img_width = run_ctx.dataset.params.get('img_width', 768)
    img_height = run_ctx.dataset.params.get('img_height', 768)
    mask_data : List[Tuple[str, str]] = []

    for row in tqdm(ship_encodings_df.itertuples(),
                    total = ship_encodings_df.shape[0],
                    file = sys.stdout,
                    dynamic_ncols = True):
        img_name = getattr(row, 'ImageId')
        mask_name = f'{img_name.split('.')[0]}.bin'

        # Step 2.1: Create mask file
        mask = rle_ship_encoding_to_mask(getattr(row, 'EncodedPixels'), img_width, img_height)
        mask.tofile(mask_folder / mask_name)

        # Step 2.2: Write metadata
        mask_data.append((img_name, (mask_folder / mask_name).relative_to(mask_folder.parent)))

    # Step 3: Generating the index file
    pd.DataFrame(mask_data, columns = ['img_name', 'mask_path']).to_csv(mask_folder.parent / 'index.csv', index = False)
