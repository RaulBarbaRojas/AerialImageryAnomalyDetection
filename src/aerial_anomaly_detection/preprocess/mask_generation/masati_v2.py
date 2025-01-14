"""
Module to generate ship masks from the XML annotations.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import pandas as pd
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


def generate_partition_masks(base_output_folder : str | Path,
                             partition : str,
                             partitions_df : pd.DataFrame,
                             img_to_annotation_fn : Callable[[str], str]) -> pd.DataFrame:
    """
    Method to generate all the masks of a given partition.

    Args:
        base_output_folder (str | Path): the base folder where the masks will be output'd.
        partition (str): the name of the partition whose masks will be generated.
        partitions_df (DataFrame): the dataframe containing image partition assignments.
        img_to_annotation_fn (Callable[[str], str]): a function that converts the image path into the XML ship annotation path.
    """
    mask_metadata : List[Tuple[str, str]] = []
    base_output_folder = Path(base_output_folder)
    partitions_df = partitions_df[partitions_df['partition'] == partition]
    (base_output_folder / partition).mkdir(exist_ok = True, parents = True)

    for row in tqdm(partitions_df.itertuples(),
                    total = partitions_df.shape[0],
                    desc = f'Generating {partition} masks',
                    unit = 'mask',
                    file = sys.stdout):
        mask = generate_mask_from_xml_annotation(img_to_annotation_fn(row.path),
                                                 row.width,
                                                 row.height)
        mask_path = base_output_folder / partition / Path(row.path).name
        cv2.imwrite(mask_path, mask)
        mask_metadata.append((row.path, mask_path.relative_to(base_output_folder.parent)))

    return pd.DataFrame(mask_metadata, columns = ['path', 'mask_path'])


if __name__ == '__main__':

    # Step 1: Reading partitioning data
    partition_df = pd.read_csv(Path(run_ctx.params.output_dataset_folder) / 'partitions.csv')
    (mask_output_folder := Path(run_ctx.params.mask_output_folder)).mkdir(exist_ok = True, parents = True)

    # Step 2: Generating masks
    pd.concat([generate_partition_masks(mask_output_folder, 'train', partition_df, lambda _: None),
               generate_partition_masks(mask_output_folder, 'val', partition_df, lambda _: None),
               generate_partition_masks(mask_output_folder, 'test', partition_df, lambda img_name: run_ctx.dataset.folder / Path(Path(img_name).parent.name + '_labels') / (Path(img_name).stem + '.xml'))]).to_csv(Path(run_ctx.params.output_dataset_folder) / 'masks.csv', index = False)
