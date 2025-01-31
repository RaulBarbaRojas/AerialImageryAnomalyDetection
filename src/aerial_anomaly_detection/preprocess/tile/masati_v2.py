"""
Module to tile the MASATI (v2) dataset.
"""


import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.preprocess.norm.functions.unitary_symmetric_interval_norm import UnitarySymmetricIntervalNorm
from aerial_anomaly_detection.preprocess.tile.landcover_ai_tiler import LandCoverAITiler


def tile_partition(scene_df : pd.DataFrame,
                   partition : str,
                   dataset_folder : Path,
                   processed_dataset_folder : Path) -> List[Tuple[str, str, str, int, int, int, int, str]]:
    """
    Function to apply tiling to a given partition.

    Args:
        scene_df (DataFrame): the DataFrame containing information about the scenes and their partitions.
        partition (str): the specific partition to be tiled.
        dataset_folder (Path): the folder containing the scenes to be read.
        processed_dataset_folder (Path): the folder where the tiles and their masks will be stored.

    Returns:
        A tuple containing the path of the tile, the path to its mask, the identifier of the scene it belongs to,\
        its width, height, the x-axis tile step, the y-axis tile step, and the partition it belongs to.
    """
    tile_index_data : List[Tuple[str, str, str, int, int, int, int, str]] = []
    partition_df = scene_df[scene_df['partition'] == partition]

    for scene_data in tqdm(partition_df.itertuples(),
                           total = partition_df.shape[0],
                           desc = f'Tiling {partition} scenes',
                           unit = 'scene',
                           file = sys.stdout,
                           dynamic_ncols = True):
        scene_id = Path(scene_data.img_path).stem

        # Step 1: Converting the scene into RGB-Planar scene
        scene_img = cv2.imread(dataset_folder / scene_data.img_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        scene_img = scene_img.transpose(2, 0, 1)
        scene_img = scene_img[::-1, ...]

        scene_mask_path = processed_dataset_folder / 'scene_test' / 'masks' / f'{scene_id}.tif'
        scene_mask = np.zeros((scene_img.shape[1], scene_img.shape[2]), dtype = np.uint8) if not scene_mask_path.exists() else \
                     rasterio.open(scene_mask_path).read()
        scene_mask = scene_mask.squeeze()

        (partition_scene_out_folder := processed_dataset_folder / partition / scene_id).mkdir(exist_ok = True, parents = True)
        (partition_scene_out_folder / 'images').mkdir(exist_ok = True, parents = True)
        (partition_scene_out_folder / 'masks').mkdir(exist_ok = True, parents = True)

        for tile, tile_mask, x, y in tiler_fn.tile(scene_img, scene_mask):
            norm_tile = norm_fn(tile).astype(np.float32)
            tile_path = (partition_scene_out_folder / 'images' / f'x{x}_y{y}.bin')
            tile_mask_path = (partition_scene_out_folder / 'masks' / f'x{x}_y{y}.bin')

            norm_tile.tofile(tile_path)
            tile_mask.tofile(tile_mask_path)

            tile_index_data.append((tile_path.relative_to(processed_dataset_folder),
                                    tile_mask_path.relative_to(processed_dataset_folder),
                                    scene_id, tile_width, tile_height, tile_x_step, tile_y_step, partition))

    return tile_index_data


if __name__ == '__main__':

    # Step 1: Preparing the tiling process
    processed_folder = Path(run_ctx.dataset.params.processed_folder)
    scene_df = pd.read_csv(processed_folder / 'scene_index.csv')
    tile_index_data : List[Tuple[str, str, str, int, int, int, int, str]] = []

    norm_fn = UnitarySymmetricIntervalNorm(scale = 255)

    tile_width = run_ctx.params.get('tile_width', 32)
    tile_height = run_ctx.params.get('tile_height', 32)
    tile_x_step = run_ctx.params.get('tile_x_step', 32)
    tile_y_step = run_ctx.params.get('tile_y_step', 32)
    tiler_fn = LandCoverAITiler(tile_width, tile_height, tile_x_step, tile_y_step) # LandCoverAI tiler works for MASATI (v2)

    # Step 2: Tiling the different partitions
    for partition in scene_df['partition'].unique():
        tile_index_data += tile_partition(scene_df, partition, run_ctx.dataset.folder, processed_folder)

    # Step 3: Storing the tile index file
    pd.DataFrame(tile_index_data,
                 columns = ['tile_path', 'tile_mask_path', 'scene_id',
                            'tile_width', 'tile_height', 'tile_x_step',
                            'tile_y_step', 'partition']).to_csv(processed_folder / 'tile_index.csv', index = False)
