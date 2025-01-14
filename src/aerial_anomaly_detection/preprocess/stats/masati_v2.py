"""
Module to calculate certain stats related to the MASATI (v2) dataset.
"""


import sys
from configparser import ConfigParser
from pathlib import Path

import cv2
import pandas as pd
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.preprocess.stats import StatsCalculator


if __name__ == '__main__':

    # Step 1: Setting up stat calculation.
    processed_dataset_folder = Path(run_ctx.params.output_dataset_folder)
    index_df = pd.read_csv(processed_dataset_folder / 'index.csv')
    index_df = index_df[index_df['partition'] == 'train']
    stats_calculator = StatsCalculator()

    # Step 2: Iterating through all train images
    for row in tqdm(index_df.itertuples(),
                    desc = 'Calculating stats through all train images',
                    unit = 'image',
                    total = index_df.shape[0],
                    file = sys.stdout):
        img = cv2.imread(processed_dataset_folder / row.path)[...,::-1].transpose(2, 0, 1)
        stats_calculator.update(img)

    # Step 3: Calculating the mean and the standard deviation and saving the stats in the info.ini file.
    mean = stats_calculator.mean
    scale = stats_calculator.std

    info_file = ConfigParser()
    info_file.add_section('stats')
    info_file['stats']['color_channel_order'] = 'RGB'
    info_file['stats']['format'] = 'Planar'
    info_file['stats']['mean'] = str(mean.tolist())
    info_file['stats']['scale'] = str(scale.tolist())

    with open(processed_dataset_folder / 'info.ini', mode = 'w', encoding = 'utf-8') as info_file_pointer:
        info_file.write(info_file_pointer)
