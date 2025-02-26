"""
Module to set up the LandCover.ai dataset by indexing its scenes, annotating their size and deciding the ones to be used in the project.
"""


import random
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import rasterio
from afml.context import run_ctx
from tqdm.auto import tqdm


if __name__ == '__main__':

    # Step 1: Preparing scene indexation
    scene_data : List[Tuple[str, int, int, str]] = []
    img_dir = Path(run_ctx.dataset.folder) / 'HRC_WHU'
    img_list = list(img_dir.iterdir())
    test_pct = run_ctx.params.get('test_pct', 0.2)

    # Step 2: Indexing scenes
    for img_path in tqdm(img_list,
                         desc = 'Indexing scenes',
                         unit = 'scene',
                         file = sys.stdout,
                         dynamic_ncols = True):
        if 'Mask' in str(img_path):
            continue

        img = rasterio.open(img_path).read()
        partition = 'test' if random.uniform(0, 1) < test_pct else 'train'
        scene_data.append((img_path.name, img.shape[2], img.shape[1], partition))

    # Step 3: Storing results
    (out_folder := Path(run_ctx.dataset.params.processed_folder)).mkdir(exist_ok = True, parents = True)
    scene_df = pd.DataFrame(scene_data, columns = ['scene_id', 'width', 'height', 'partition'])
    scene_df.to_csv(out_folder / 'scene_index.csv', index = False)
