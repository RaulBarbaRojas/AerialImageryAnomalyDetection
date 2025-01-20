"""
Module to implement image normalization for the calculated patches of the MASATI (v2) dataset.
"""


import sys
from configparser import ConfigParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.preprocess.norm.functions.unitary_symmetric_interval_norm import UnitarySymmetricIntervalNorm


def normalize_partition(partition : str, index_df : pd.DataFrame, processed_dataset_folder : Path) -> None:
    """
    Method to normalize a given partition.

    Args:
        partition (str): the partition to be normalized.
        index_df (DataFrame): the index dataframe of the dataset.
        processed_dataset_folder (Path): the path to the processed dataset.
    """
    norm_metadata = []
    index_df = index_df[index_df['partition'] == partition]



if __name__ == '__main__':

    # Step 1: Reading the mean-scale information from the .ini file
    processed_dataset_folder = Path(run_ctx.params.output_dataset_folder)
    index_df = pd.read_csv(processed_dataset_folder / 'index.csv')
    norm_fn = UnitarySymmetricIntervalNorm()

    # Step 2: Checking if normalized tiles have already been generated
    if 'norm_path' in index_df.columns:
        print('Skipping step, normalized information already generated...')
        exit()

    # Step 3: Applying MeanScale normalization
    norm_paths = []
    for row in tqdm(index_df.itertuples(),
                    desc = 'Applying normalization',
                    unit = 'tile',
                    total = index_df.shape[0],
                    file = sys.stdout):
        norm_path = row.path.replace('base', 'norm').replace('.png', '.bin')
        (processed_dataset_folder / norm_path).parent.mkdir(exist_ok = True, parents = True)

        img = cv2.imread(processed_dataset_folder / row.path, cv2.IMREAD_UNCHANGED)
        img = img[...,::-1].transpose(2, 0, 1).astype(np.float32)
        norm_img = norm_fn(img)
        norm_img.tofile(processed_dataset_folder / norm_path)
        norm_paths.append(norm_path)

    # Step 4: Storing normalization metadata
    index_df['norm_path'] = norm_paths
    index_df.to_csv(processed_dataset_folder / 'index.csv', index = False)
