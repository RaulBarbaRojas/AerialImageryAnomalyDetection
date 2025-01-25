"""
Module to implement the tiling strategy of the LandCover.ai dataset.
"""


import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import rasterio
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.preprocess.norm.functions.unitary_symmetric_interval_norm import UnitarySymmetricIntervalNorm
from aerial_anomaly_detection.preprocess.tile.landcover_ai_tiler import LandCoverAITiler


if __name__ == '__main__':

    # Step 1: Setting up the tiling process
    processed_folder = Path(run_ctx.dataset.params.processed_folder)
    scene_df = pd.read_csv(processed_folder / 'scene_index.csv')
    train_df = scene_df[scene_df['partition'] == 'train']
    test_df = scene_df[scene_df['partition'] == 'test']
    tile_index_data : List[Tuple[str, str, int, int, int, int, str, int]] = []

    norm_fn = UnitarySymmetricIntervalNorm(scale = 255)

    tile_width = run_ctx.params.get('tile_width', 32)
    tile_height = run_ctx.params.get('tile_height', 32)
    tile_x_step = run_ctx.params.get('tile_x_step', 32)
    tile_y_step = run_ctx.params.get('tile_y_step', 32)
    tiler_fn = LandCoverAITiler(tile_width, tile_height, tile_x_step, tile_y_step)

    # Step 2: Tile train scenes
    (out_train_folder := processed_folder / 'train_val').mkdir(exist_ok = True, parents = True)

    for scene_data in tqdm(train_df.itertuples(),
                           total = train_df.shape[0],
                           desc = 'Tiling train/val scenes',
                           unit = 'scene',
                           file = sys.stdout,
                           dynamic_ncols = True):
        scene = rasterio.open(run_ctx.dataset.folder / 'images' / scene_data.scene_id).read()
        mask = rasterio.open(run_ctx.dataset.folder / 'masks' / scene_data.scene_id).read().squeeze()
        (train_scene_out_folder := out_train_folder / Path(scene_data.scene_id).stem).mkdir(exist_ok = True, parents = True)

        for tile, tile_mask, x, y in tiler_fn.tile(scene, mask):
            if np.sum(tile_mask == 3) == tile_width * tile_height:
                norm_tile = norm_fn(tile).astype(np.float32)
                tile_path = (train_scene_out_folder / f'x{x}_y{y}.bin')
                tile_index_data.append((tile_path.relative_to(out_train_folder.parent), scene_data.scene_id,
                                        tile_width, tile_height, tile_x_step, tile_y_step, 'train', 3))
                norm_tile.tofile(tile_path)

    # Step 3: Tile test images
    (out_test_folder := processed_folder / 'test').mkdir(exist_ok = True, parents = True)
    test_tiles_per_type_and_scene = run_ctx.params.get('test_tiles_per_type_and_scene', 100)

    for scene_data in tqdm(test_df.itertuples(),
                           total = test_df.shape[0],
                           desc = 'Tiling test scenes',
                           unit = 'scene',
                           file = sys.stdout,
                           dynamic_ncols = True):
        scene = rasterio.open(run_ctx.dataset.folder / 'images' / scene_data.scene_id).read()
        mask = rasterio.open(run_ctx.dataset.folder / 'masks' / scene_data.scene_id).read().squeeze()
        (test_scene_out_folder := out_test_folder / Path(scene_data.scene_id).stem).mkdir(exist_ok = True, parents = True)

        for idx_tile_type in range(5):
            tile_type_scene_count = test_tiles_per_type_and_scene

            for tile, tile_mask, x, y in tiler_fn.tile(scene, mask):
                if np.sum(tile_mask == idx_tile_type) == tile_width * tile_height:
                    norm_tile = norm_fn(tile).astype(np.float32)
                    tile_path = (test_scene_out_folder / f'x{x}_y{y}.bin')
                    tile_index_data.append((tile_path.relative_to(out_test_folder.parent), scene_data.scene_id,
                                            tile_width, tile_height, tile_x_step, tile_y_step, 'test', idx_tile_type))
                    norm_tile.tofile(tile_path)
                    tile_type_scene_count -= 1

                    if tile_type_scene_count == 0:
                        break


    # Step 4: Storing the tile index file
    pd.DataFrame(tile_index_data,
                 columns = ['tile_path', 'scene_id', 'tile_width', 'tile_height', 'tile_x_step', 'tile_y_step',
                            'preeliminary_partition', 'tile_type']).to_csv(processed_folder / 'tile_index.csv', index = False)
