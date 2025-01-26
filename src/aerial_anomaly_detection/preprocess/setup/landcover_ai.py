"""
Module to set up the LandCover.ai dataset by indexing its scenes, annotating their size and deciding the ones to be used in the project.
"""


import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import rasterio
from afml.context import run_ctx
from tqdm.auto import tqdm


SCENES_TO_REMOVE = ['N-34-66-C-c-4-3.tif',
                    'N-33-119-C-c-3-3.tif']

if __name__ == '__main__':

    # Step 1: Preparing scene indexation
    scene_data : List[Tuple[str, int, int, str]] = []
    img_dir = Path(run_ctx.dataset.folder) / 'images'
    img_list = list(img_dir.iterdir())

    # Step 2: Indexing scenes
    for img_path in tqdm(img_list,
                         desc = 'Indexing scenes',
                         unit = 'scene',
                         file = sys.stdout,
                         dynamic_ncols = True):
        img = rasterio.open(img_path).read()
        scene_data.append((img_path.name, img.shape[2], img.shape[1], 'train' if (img.shape[2] > 5000) and (img.shape[1] > 5000) else 'test'))

    # Step 3: Discarding images with known data quality issues
    scene_df = pd.DataFrame(scene_data, columns = ['scene_id', 'width', 'height', 'partition'])
    scene_df = scene_df.loc[scene_df['scene_id'].isin(SCENES_TO_REMOVE), 'partition'] = 'discarded'

    # Step 4: Storing results
    (out_folder := Path(run_ctx.dataset.params.processed_folder)).mkdir(exist_ok = True, parents = True)
    scene_df.to_csv(out_folder / 'scene_index.csv', index = False)
