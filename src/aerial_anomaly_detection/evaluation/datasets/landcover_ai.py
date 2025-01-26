"""
Generic module to evaluate a model over the LandCover.ai dataset.
"""


import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from pandas import DataFrame
from tqdm.auto import tqdm

from aerial_anomaly_detection.evaluation.datasets import ModelEvaluator
from aerial_anomaly_detection.evaluation.metrics import accuracy, f1, precision, recall
from aerial_anomaly_detection.preprocess.norm.functions import Norm


class LandCoverAIModelEvaluator(ModelEvaluator):
    """
    A class to evaluate a given model with the LandCover.ai dataset.
    """


    def __init__(self, norm_fn : Norm, tile_width : int, tile_height : int, tile_x_step : int,
                 tile_y_step : int, input_folder : str | Path, output_folder : str | Path,
                 scene_df : DataFrame, *args : Tuple[Any, ...], **kwargs : Dict[str, Any]):
        """
        Constructor method for the LandCoverAIModelEvaluator class.

        Args:
            norm_fn (Norm): the normalization function to be applied on the tiles.
            tile_width (int): the width of the LandCover.ai tiles to be used.
            tile_height (int): the height of the LandCover.ai tiles to be used.
            tile_x_step (int): the x-axis step to be used when tiling.
            tile_y_step (int): the y-axis step to be used when tiling.
            input_folder (str | Path): the path to the base folder of the downloaded LandCover.ai dataset with images and masks.
            output_folder (str | Path): the path where predicted masks will be stored for visual model evaluation.
            scene_df (DataFrame): the scene partition dataframe containing the identifier of the test scenes.
            *args and **kwargs: the required base arguments of a ModelEvaluator instance.
        """

        # Step 1: Setting up the base attributes
        super().__init__(*args, **kwargs)

        # Step 2: Setting up the new attributes
        self.norm_fn = norm_fn
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_x_step = tile_x_step
        self.tile_y_step = tile_y_step
        self.input_scene_folder = Path(input_folder) / 'images'
        self.input_mask_folder = Path(input_folder) / 'masks'
        self.output_folder = Path(output_folder)
        self.scene_df = scene_df[scene_df['partition'] == 'test']


    def evaluate(self) -> Tuple[DataFrame, DataFrame]:
        """
        Method to evaluate a model on the LandCover.ai dataset.

        Returns:
            A tuple with a DataFrame holding per-scene metrics and a DataFrame holding\
            averaged metrics across all scenes. 
        """

        # Step 1: Setting up model evaluation
        global_confussion_matrix = np.zeros((2, 2))
        scene_metric_data : List[Tuple[str, int, int, int, int, float, float, float, float]] = []
        metric_summary_data : List[Tuple[str, int, int, int, int, float, float, float, float]] = []
        (out_pred_masks_folder := self.output_folder / 'masks' / 'pred').mkdir(exist_ok = True, parents = True)
        (out_true_masks_folder := self.output_folder / 'masks' / 'true').mkdir(exist_ok = True, parents = True)

        # Step 2: Calculating the reconstruction error threshold
        reconstruction_error_threshold = self._calculate_reconstruction_error_treshold()

        # Step 3: Evaluating the model over the test LandCover.ai scenes
        self.model.eval()
        with torch.inference_mode():
            for scene_id in tqdm(self.scene_df['scene_id'].to_list(),
                                desc = 'Running inference over test scenes',
                                unit = 'scene',
                                file = sys.stdout,
                                dynamic_ncols = True):

                # Step 2.1: Reading scene information
                scene = rasterio.open(self.input_scene_folder / scene_id).read().astype(np.uint8)
                mask = rasterio.open(self.input_mask_folder / scene_id).read().squeeze()
                pred_mask = np.zeros_like(mask)
                scene_confussion_matrix = np.zeros_like(global_confussion_matrix)

                # Step 2.2: Tiling and running inference over tiles
                for y_coord in tqdm(range(0, scene.shape[1], self.tile_y_step),
                                    desc = 'Running inference over tile-rows',
                                    dynamic_ncols = True,
                                    leave = True,
                                    file = sys.stdout,
                                    unit = 'tile'):
                    y_coord = y_coord if y_coord + self.tile_height < scene.shape[1] else scene.shape[1] - self.tile_height

                    for x_coord in range(0, scene.shape[2], self.tile_x_step):
                        x_coord = x_coord if x_coord + self.tile_width < scene.shape[2] else scene.shape[2] - self.tile_width

                        tile = scene[:, y_coord : y_coord + self.tile_height, x_coord : x_coord + self.tile_width]
                        tile = self.norm_fn(tile)
                        tile = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                        tile_mask = mask[y_coord : y_coord + self.tile_height, x_coord : x_coord + self.tile_width]
                        ground_truth_anomaly = int(int(np.sum(tile_mask != 3)) > (self.tile_width * self.tile_height / 2))

                        y_pred = self.model(tile)

                        reconstruction_error = self.reconstruction_error_fn(self.model, y_pred.to(self.device),
                                                                            tile, torch.Tensor([ground_truth_anomaly]))

                        pred_anomaly = int(reconstruction_error > reconstruction_error_threshold)
                        pred_mask[y_coord : y_coord + self.tile_height, x_coord : x_coord + self.tile_width] |= pred_anomaly * 255

                        global_confussion_matrix[ground_truth_anomaly, pred_anomaly] += 1
                        scene_confussion_matrix[ground_truth_anomaly, pred_anomaly] += 1

                # Step 2.3: Storing mask and metrics per scene
                with rasterio.open(out_pred_masks_folder / scene_id,
                                    mode = 'w',
                                    driver='GTiff',
                                    height=pred_mask.shape[0],
                                    width=pred_mask.shape[1],
                                    count=1,
                                    dtype=np.uint8) as mask_writer:
                    mask_writer.write(pred_mask, 1)
                    plt.figure(figsize = (10, 7))
                    plt.imshow(pred_mask)
                    plt.title('Predicted scene')
                    plt.axis(False)
                    plt.tight_layout()
                    plt.savefig(out_pred_masks_folder /f'{Path(scene_id).stem}.png')
                    plt.close()

                with rasterio.open(out_true_masks_folder / scene_id,
                                    mode = 'w',
                                    driver='GTiff',
                                    height=mask.shape[0],
                                    width=mask.shape[1],
                                    count=1,
                                    dtype=np.uint8) as mask_writer:
                    processed_mask = ((mask != 3) * 255).astype(np.uint8)
                    mask_writer.write(processed_mask, 1)
                    plt.figure(figsize = (10, 7))
                    plt.imshow(processed_mask)
                    plt.title('Ground truth scene')
                    plt.axis(False)
                    plt.tight_layout()
                    plt.savefig(out_true_masks_folder /f'{Path(scene_id).stem}.png')
                    plt.close()

                scene_metric_data.append((scene_id, scene_confussion_matrix[0, 0], scene_confussion_matrix[0, 1],
                                          scene_confussion_matrix[1, 0], scene_confussion_matrix[1, 1],
                                          accuracy(scene_confussion_matrix), precision(scene_confussion_matrix),
                                          recall(scene_confussion_matrix), f1(scene_confussion_matrix)))
                print('Scene confussion matrix: \n', scene_confussion_matrix)
                print(f'Scene Acc: {scene_metric_data[-1][5]:.6f}')
                print(f'Scene Precision: {scene_metric_data[-1][6]:.6f}')
                print(f'Scene Recall: {scene_metric_data[-1][7]:.6f}')
                print(f'Scene F1: {scene_metric_data[-1][8]:.6f}')

            # Step 2.4: Storing metrics across scenes
            metric_summary_data.append((global_confussion_matrix[0, 0], global_confussion_matrix[0, 1],
                                        global_confussion_matrix[1, 0], global_confussion_matrix[1, 1],
                                        accuracy(global_confussion_matrix), precision(global_confussion_matrix),
                                        recall(global_confussion_matrix), f1(global_confussion_matrix)))

        # Step 3: Storing the inference metadata
        scene_metric_df = DataFrame(scene_metric_data,
                                    columns = ['scene_id', 'true_negatives', 'false_positives', 'false_negatives',
                                               'true_positives', 'accuracy', 'precision', 'recall', 'f1'])
        global_metric_df = DataFrame(metric_summary_data,
                                     columns = ['true_negatives', 'false_positives', 'false_negatives',
                                                'true_positives', 'accuracy', 'precision', 'recall', 'f1'])

        return (scene_metric_df, global_metric_df)
