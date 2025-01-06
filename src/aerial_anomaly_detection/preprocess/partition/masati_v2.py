"""
A module to implement the image partitioning to be used in the MASATI (v2) dataset.
"""

import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import pandas as pd
from afml.context import run_ctx
from tqdm.auto import tqdm


def process_partition_images(partition_name : str,
                             partition_images : List[Path],
                             dataset_root_folder : Path) -> List[Tuple[str, int, int, str]]:
    """
    Function to process the images of a partition.

    Args:
        partition_name (str): the name of the partition being processed.
        partition_images (List[Path]): a list containing all the paths to the images of the partition.
        dataset_root_folder (Path): the path to the dataset root folder (required for relative path metadata extraction).

    Returns:
        A list of metadata tuples for each image in the partition.
    """
    partition_info : List[Tuple[str, int, int, str]] = []

    for partition_image_path in tqdm(partition_images,
                                     desc = f'Processing {partition_name} images',
                                     dynamic_ncols = True,
                                     file = sys.stdout):
        partition_image = cv2.imread(partition_image_path, cv2.IMREAD_UNCHANGED)
        partition_info.append((partition_image_path.relative_to(dataset_root_folder),
                               partition_image.shape[1],
                               partition_image.shape[0],
                               partition_name))

    return partition_info


if __name__ == '__main__':

    # Step 1: Data partitioning setup
    dataset_folder = run_ctx.dataset.folder
    partition_info : List[Tuple[str, int, int, str]] = []
    train_size_pct = run_ctx.params.get('train_size_pct', 0.85)

    # Step 2: Adding train partition information
    train_val_images = list((dataset_folder / 'water').iterdir())
    random.shuffle(train_val_images)
    train_images = train_val_images[:round(train_size_pct * len(train_val_images))]
    val_images = train_val_images[round(train_size_pct * len(train_val_images)):]

    partition_info += process_partition_images('train', train_images, dataset_folder)

    # Step 3: Adding validation partition information
    partition_info += process_partition_images('val', val_images, dataset_folder)

    # Step 4: Adding test partition information
    test_images = list((dataset_folder / 'ship').iterdir())
    test_images += [test_image_path for test_image_path in list((dataset_folder / 'multi').iterdir())
                    if test_image_path.name[0] == 'm'] # Only water images, ignore land images (with ships).
    partition_info += process_partition_images('test', test_images, dataset_folder)

    # Step 5: Storing the partitions.csv file
    (output_dataset_folder := Path(run_ctx.params.output_dataset_folder)).mkdir(exist_ok = True, parents = True)
    pd.DataFrame(partition_info,
                 columns = ['path', 'width', 'height', 'partition']).to_csv(output_dataset_folder / 'partitions.csv', index = False)
