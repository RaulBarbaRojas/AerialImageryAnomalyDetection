"""
A module to implement image tiling into smaller sub-images.
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
    dataset_folder = run_ctx.dataset.folder
    output_base_folder = Path(run_ctx.params.output_dataset_folder)
    partition_df = pd.read_csv(output_base_folder / 'partitions.csv')

    tile_x_size = run_ctx.params.get('tile_x_size', 128)
    tile_y_size = run_ctx.params.get('tile_y_size', 128)
    x_step = run_ctx.params.get('x_step', 64)
    y_step = run_ctx.params.get('y_step', 64)
    tiler = OverlapTiler(tile_x_size = tile_x_size,
                         tile_y_size = tile_y_size,
                         x_step = x_step,
                         y_step = y_step)

    # Step 2: Tiling images
    for row in tqdm(partition_df.itertuples(),
                    desc = 'Tiling MASATI (v2) images',
                    total = partition_df.shape[0],
                    unit = 'image',
                    dynamic_ncols = True,
                    file = sys.stdout):
        image_path = dataset_folder / row.path
        image_partition = row.partition
        image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        (output_base_folder / image_partition / 'base').mkdir(exist_ok = True, parents = True)

        for tile, x_coord, y_coord in tiler.tile(image_data):
            tile_path = output_base_folder / image_partition / 'base' / f'{Path(row.path).stem}_x{x_coord}_y{y_coord}.png'
            cv2.imwrite(tile_path, tile)
            tiling_metadata.append((tile_path.relative_to(output_base_folder),
                                    image_partition,
                                    x_coord,
                                    y_coord,
                                    tile_x_size,
                                    tile_y_size,
                                    x_step,
                                    y_step))

    # Step 3: Storing the metadata file (index.csv)
    pd.DataFrame(tiling_metadata, columns = ['path', 'partition', 'x_coord', 'y_coord',
                                             'tile_x_size', 'tile_y_size', 'x_step', 'y_step']).to_csv(output_base_folder / 'index.csv',
                                                                                                       index = False)
