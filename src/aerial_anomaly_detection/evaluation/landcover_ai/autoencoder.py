"""
Module to implement a scene-based evaluation of the models.
"""


import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.datasets import DataLoader
from aerial_anomaly_detection.datasets.landcover_ai import LandCoverAI
from aerial_anomaly_detection.evaluation import accuracy, f1, precision, recall
from aerial_anomaly_detection.models.autoencoder import AutoEncoder
from aerial_anomaly_detection.preprocess.norm.functions.unitary_symmetric_interval_norm import UnitarySymmetricIntervalNorm


if __name__ == '__main__':

    # Step 1: Preparing scene-based evaluation
    processed_dataset_folder = Path(run_ctx.dataset.params.processed_folder)
    (out_folder := Path(run_ctx.params.out_folder)).mkdir(exist_ok = True, parents = True)
    scene_folder = run_ctx.dataset.folder / 'images'
    mask_folder = run_ctx.dataset.folder / 'masks'

    scene_df = pd.read_csv(processed_dataset_folder / 'scene_index.csv')
    scene_df = scene_df[scene_df['partition'] == 'test']

    norm_fn = UnitarySymmetricIntervalNorm(scale = 255)

    tile_width = run_ctx.params.get('tile_width', 32)
    tile_height = run_ctx.params.get('tile_height', 32)
    tile_x_step = run_ctx.params.get('tile_x_step', 32)
    tile_y_step = run_ctx.params.get('tile_y_step', 32)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoEncoder(latent_dimension = run_ctx.model.params.get('latent_dimension', 1000),
                        img_width = run_ctx.model.params.get('img_width', 32),
                        img_height = run_ctx.model.params.get('img_height', 32)).to(device)
    model.load_state_dict(torch.load(run_ctx.model.params.model_weights, weights_only = True))

    # Step 2: Calculating threshold
    val_loss = 0
    loss_fn = torch.nn.MSELoss()
    batch_size = run_ctx.params.get('batch_size', 256)
    val_dataset = LandCoverAI.load(processed_dataset_folder, partition = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count())

    model.eval()
    with torch.inference_mode():
        for _, X, _ in tqdm(val_dataloader,
                            desc = 'Validation step',
                            unit = 'batch',
                            dynamic_ncols = True,
                            file = sys.stdout,
                            leave = True):
            X = X.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, X)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    error_threshold = val_loss # Using this tactic (other methods could be used/explored)
    print(f'Threshold: {error_threshold:.6f}')

    # Step 3: Running inference over
    global_confussion_matrix = np.zeros((2, 2))
    scene_metric_data : List[Tuple[str, int, int, int, int, float, float, float, float]] = []
    metric_summary_data : List[Tuple[str, int, int, int, int, float, float, float, float]] = []
    (out_pred_masks_folder := out_folder / 'masks' / 'pred').mkdir(exist_ok = True, parents = True)
    (out_true_masks_folder := out_folder / 'masks' / 'true').mkdir(exist_ok = True, parents = True)

    model.eval()
    with torch.inference_mode():
        for scene_id in tqdm(scene_df['scene_id'].to_list(),
                            desc = 'Running inference over test scenes',
                            unit = 'scene',
                            file = sys.stdout,
                            dynamic_ncols = True):

            # Step 2.1: Reading scene information
            scene = rasterio.open(scene_folder / scene_id).read().astype(np.uint8)
            mask = rasterio.open(mask_folder / scene_id).read().squeeze()
            pred_mask = np.zeros_like(mask)
            scene_confussion_matrix = np.zeros_like(global_confussion_matrix)

            # Step 2.2: Tiling and running inference over tiles
            for y_coord in tqdm(range(0, scene.shape[1], tile_y_step),
                                desc = 'Running inference over tile-rows',
                                dynamic_ncols = True,
                                leave = True,
                                file = sys.stdout,
                                unit = 'tile'):
                y_coord = y_coord if y_coord + tile_height < scene.shape[1] else scene.shape[1] - tile_height

                for x_coord in range(0, scene.shape[2], tile_x_step):
                    x_coord = x_coord if x_coord + tile_width < scene.shape[2] else scene.shape[2] - tile_width

                    tile = scene[:, y_coord : y_coord + tile_height, x_coord : x_coord + tile_width]
                    tile = norm_fn(tile)
                    tile = torch.from_numpy(tile).unsqueeze(0).to(device)
                    tile_mask = mask[y_coord : y_coord + tile_height, x_coord : x_coord + tile_width]

                    y_pred = model(tile)

                    reconstruction_error = loss_fn(y_pred.to(device), tile).item()

                    pred_anomaly = int(reconstruction_error > error_threshold)
                    ground_truth_anomaly = int(int(np.sum(tile_mask != 3)) > (tile_width * tile_height / 2))
                    pred_mask[y_coord : y_coord + tile_height, x_coord : x_coord + tile_width] |= pred_anomaly * 255

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
    pd.DataFrame(scene_metric_data, columns = ['scene_id', 'true_negatives', 'false_positives', 'false_negatives', 'true_positives',
                                               'accuracy', 'precision', 'recall', 'f1']).to_csv(out_folder / 'scene_metrics.csv', index = False)
    pd.DataFrame(metric_summary_data, columns = ['true_negatives', 'false_positives', 'false_negatives', 'true_positives',
                                                 'accuracy', 'precision', 'recall', 'f1']).to_csv(out_folder / 'global_metrics.csv', index = False)
