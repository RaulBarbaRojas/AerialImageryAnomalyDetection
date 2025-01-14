"""
A module to tile MASATI (v2) masks.
"""


import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import pandas as pd
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.preprocess.tile.overlap_tiler import OverlapTiler


if __name__ == '__main__':
    # Step 1: Setting up the tiling process
    tiling_metadata : List[Tuple[str, str, int, int, int, int, int, int]] = []
    processed_dataset_folder = Path(run_ctx.params.output_dataset_folder)
    index_df = pd.read_csv(processed_dataset_folder / 'index.csv')
    masks_df = pd.read_csv(processed_dataset_folder / 'masks.csv')

    tile_x_size = run_ctx.params.get('tile_x_size', 128)
    tile_y_size = run_ctx.params.get('tile_y_size', 128)
    x_step = run_ctx.params.get('x_step', 64)
    y_step = run_ctx.params.get('y_step', 64)
    tiler = OverlapTiler(tile_x_size = tile_x_size,
                         tile_y_size = tile_y_size,
                         x_step = x_step,
                         y_step = y_step)

    # Step 2: Running the script only if it was not previously run
    if 'mask_path' in index_df.columns:
        print('Skipping step, tile mask information already generated...')
        exit()

    # Step 3: Running tile partitioning
    mask_paths : List[str] = []
    for row in tqdm(masks_df.itertuples(),
                    desc = 'Tiling MASATI (v2) images',
                    total = masks_df.shape[0],
                    unit = 'image',
                    dynamic_ncols = True,
                    file = sys.stdout):
        partition = Path(row.mask_path).parent.name
        (masks_folder := processed_dataset_folder / partition / 'masks').mkdir(exist_ok = True, parents = True)
        mask_data = cv2.imread(processed_dataset_folder / row.mask_path)

        for tile, x_coord, y_coord in tiler.tile(mask_data):
            mask_path = masks_folder / f'{Path(row.mask_path).stem}_x{x_coord}_y{y_coord}.png'
            cv2.imwrite(mask_path, tile)
            mask_paths.append(mask_path.relative_to(processed_dataset_folder))

    # Step 4: Updating the index file
    index_df['mask_path'] = mask_paths
    index_df.to_csv(processed_dataset_folder / 'index.csv', index = False)
