"""
Module to partition the tiles of the datasets used in the project.
"""


import random
import sys
from pathlib import Path

import pandas as pd
from afml.context import run_ctx
from tqdm.auto import tqdm


if __name__ == '__main__':

    # Step 1: Setting up partitioning
    processed_dataset_folder = Path(run_ctx.dataset.params.processed_folder)
    tile_df = pd.read_csv(processed_dataset_folder / 'tile_index.csv')
    train_tile_df = tile_df[tile_df['preeliminary_partition'] == 'train']
    train_pct = run_ctx.params.get('train_pct', 0.8)

    if 'partition' in tile_df.columns:
        print('Skipping step, dataset already partitioned...')
        exit()

    # Step 2: Creating partition at random per scene
    train_scenes = set(train_tile_df['scene_id'].to_list())
    val_tiles = []

    for train_scene in tqdm(train_scenes,
                            desc = 'Partitioning scenes',
                            unit = 'scene',
                            file = sys.stdout,
                            dynamic_ncols = True):
        scene_df = train_tile_df[train_tile_df['scene_id'] == train_scene]
        val_tiles += random.sample(scene_df['tile_path'].to_list(), round((1.0 - train_pct) * scene_df.shape[0]))

    tile_df['partition'] = tile_df['preeliminary_partition']
    tile_df.loc[tile_df['tile_path'].isin(val_tiles), 'partition'] = 'val'

    # Step 3: Storing tile partitioning results
    tile_df.to_csv(processed_dataset_folder / 'tile_index.csv', index = False)
