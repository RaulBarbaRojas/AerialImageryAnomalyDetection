"""
Module to set up the MASATI (v2) dataset, so that it can be used for anomaly detection.
"""


import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
from afml.context import run_ctx
from numpy.typing import NDArray
from tqdm.auto import tqdm


def generate_mask_from_xml_annotation(xml_filepath : str | Path, mask_width : int, mask_height : int) -> NDArray:
    """
    Method to generate a mask from a given XML annotation.

    Args:
        xml_filepath (str | Path): the path to the XML annotation file.
        mask_width (int): the width of the mask.
        mask_height (int): the height of the mask.

    Returns:
        The uint8 mask with sea pixels as 0 and ship pixels as 255.
    """

    # Step 1: Setting up mask generation
    mask = np.zeros((mask_height, mask_width), dtype = np.uint8)

    if xml_filepath is not None:
        xml_filepath = Path(xml_filepath)

        # Step 2: Parsing XML
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        raw_bboxes = root.findall('.//bndbox')

        # Step 3: Parsing bbox data
        processed_bboxes = [(int(raw_bbox.find('xmin').text),
                            int(raw_bbox.find('xmax').text),
                            int(raw_bbox.find('ymin').text),
                            int(raw_bbox.find('ymax').text)) for raw_bbox in raw_bboxes]

        # Step 4: Adding ship pixels to mask
        for processed_bbox in processed_bboxes:
            mask[processed_bbox[2]:processed_bbox[3] + 1, processed_bbox[0]:processed_bbox[1] + 1] = 255

    return mask


if __name__ == '__main__':

    # Step 1: Preparing scene-level partitioning
    (processed_folder := Path(run_ctx.dataset.params.processed_folder)).mkdir(exist_ok = True, parents = True)
    scene_partition_metadata : List[Tuple[str, str]] = []
    train_pct  = run_ctx.params.get('train_pct', 0.8)

    # Step 2: Adding information from the train and validation partitions
    train_val_img_paths = list((run_ctx.dataset.folder / 'water').iterdir())
    random.shuffle(train_val_img_paths)
    train_img_paths = train_val_img_paths[:round(len(train_val_img_paths) * train_pct)]
    val_img_paths = train_val_img_paths[round(len(train_val_img_paths) * train_pct):]

    for train_img in tqdm(train_img_paths,
                          desc = 'Adding train images metadata',
                          unit = 'scene',
                          dynamic_ncols = True,
                          file = sys.stdout):
        scene_partition_metadata.append((train_img.relative_to(run_ctx.dataset.folder), 'train'))

    for val_img in tqdm(val_img_paths,
                          desc = 'Adding validation images metadata',
                          unit = 'scene',
                          dynamic_ncols = True,
                          file = sys.stdout):
        scene_partition_metadata.append((val_img.relative_to(run_ctx.dataset.folder), 'val'))

    # Step 3: Adding information from the test partition
    test_img_paths = list((run_ctx.dataset.folder / 'multi').iterdir())
    test_annotation_folder = run_ctx.dataset.folder / 'multi_labels'
    out_test_folder = processed_folder / 'scene_test'
    (out_test_folder / 'masks').mkdir(exist_ok = True, parents = True)
    (out_test_folder / 'images').mkdir(exist_ok = True, parents = True)

    for test_img in tqdm(test_img_paths,
                         desc = 'Processing test scenes and adding test scene metadata',
                         unit = 'scene',
                         dynamic_ncols = True,
                         file = sys.stdout):

        # Step 3.1: Processing scene images
        scene_img = cv2.imread(test_img, cv2.IMREAD_UNCHANGED)
        if scene_img.shape[-1] != 3 or test_img.name[0] == 'y':
            continue

        scene_img = scene_img.transpose(2, 0, 1)
        scene_img = scene_img[::-1,...]

        with rasterio.open(out_test_folder / 'images' / f'{test_img.stem}.tif', mode = 'w', driver = 'GTiff',
                           count = 3, dtype = np.uint8, width = 512, height = 512) as scene_img_fp:
            scene_img_fp.write(scene_img)

        # Step 3.2: Processing scene masks
        mask = generate_mask_from_xml_annotation(test_annotation_folder / f'{test_img.stem}.xml',
                                                 mask_width = 512,
                                                 mask_height = 512)
        with rasterio.open(out_test_folder / 'masks' / f'{test_img.stem}.tif', mode = 'w', driver = 'GTiff',
                           count = 1, dtype = np.uint8, width = 512, height = 512) as mask_fp:
            mask_fp.write(mask, 1)

        # Step 3.3: Adding metadata to the scene index file
        scene_partition_metadata.append((test_img.relative_to(run_ctx.dataset.folder), 'test'))

    # Step 4: Storing scene partition file
    pd.DataFrame(scene_partition_metadata, columns = ['img_path', 'partition']).to_csv(processed_folder / 'scene_index.csv',
                                                                                       index = False)
