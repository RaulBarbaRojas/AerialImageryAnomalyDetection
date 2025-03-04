"""
Module to implement a generic model evaluation over the desired dataset.
"""


import os
from pathlib import Path

import pandas as pd
import torch
from afml.context import run_ctx

from aerial_anomaly_detection.datasets import DataLoader, Dataset
from aerial_anomaly_detection.evaluation.datasets.landcover_ai import LandCoverAIModelEvaluator
from aerial_anomaly_detection.evaluation.datasets.hrc_whu import HRC_WHUModelEvaluator
from aerial_anomaly_detection.preprocess.norm.functions.unitary_symmetric_interval_norm import UnitarySymmetricIntervalNorm


if __name__ == '__main__':

    # Step 1: Setting up the model
    model : torch.nn.Module = run_ctx.model.build()
    pretrained_weights_filepath = Path(model.model_settings.get('trained_weights', ''))
    if pretrained_weights_filepath.is_file():
        loaded_keys = model.load_state_dict(torch.load(pretrained_weights_filepath, weights_only = True))
        print(f'[ModelEvaluation] {run_ctx.model.name} weights loaded: {loaded_keys}')

    # Step 2: Setting up validation data for reconstruction error threshold calculation
    batch_size = run_ctx.params.get('batch_size', 256)

    match run_ctx.dataset.name:
        case 'LandCoverAI' | 'HRC_WHU':
            val_dataset = Dataset.load(run_ctx.dataset.params.processed_folder, partition = 'val')
        case _:
            raise ValueError(f'[ModelEvaluation] Unknown dataset "{run_ctx.dataset.name}"')

    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count())

    match run_ctx.model.name:
        case 'AutoEncoder' | 'Izi' | 'Ziz' | 'LowLevelAnoDAE' | 'HighLevelAnoDAE':
            loss_fn = torch.nn.MSELoss()
            reconstruction_error_fn = lambda model, X, y_pred, y: loss_fn(y_pred, X).item()
        case 'DCGANDiscriminator':
            reconstruction_error_fn = lambda model, X, y_pred, y: torch.mean(1.0 - y_pred).item()
        case 'BiGAN':
            loss_fn = torch.nn.L1Loss()
            reconstruction_error_fn = lambda model, X, y_pred, y: 0.9 * loss_fn(X, y_pred[0]).item() + 0.1 * loss_fn(y_pred[1], y_pred[2]).item()
        case 'f-AnoGAN':
            loss_fn = torch.nn.MSELoss()
            reconstruction_error_fn = lambda model, X, y_pred, y: loss_fn(X, y_pred[0]).item() + loss_fn(y_pred[1], y_pred[2]).item()
        case 'DualAnoDAE':
            loss_fn = torch.nn.MSELoss()
            reconstruction_error_fn = lambda model, X, y_pred, y: loss_fn(y_pred[0], X).item() + loss_fn(y_pred[1], X).item()
        case _:
            raise ValueError(f'[ModelEvaluation] Unknown model "{run_ctx.model.name}"')

    # Step 3: Setting up model evaluator
    num_errors_per_scene = run_ctx.params.get('num_errors_per_scene', 48)

    match run_ctx.dataset.name:
        case 'LandCoverAI':
            model_evaluator = LandCoverAIModelEvaluator(model = model,
                                                        validation_dataloader = val_dataloader,
                                                        reconstruction_error_fn = reconstruction_error_fn,
                                                        norm_fn = UnitarySymmetricIntervalNorm(scale = 255),
                                                        tile_width = run_ctx.params.get('tile_width', 32),
                                                        tile_height = run_ctx.params.get('tile_height', 32),
                                                        tile_x_step = run_ctx.params.get('tile_x_step', 32),
                                                        tile_y_step = run_ctx.params.get('tile_y_step', 32),
                                                        input_folder = run_ctx.dataset.folder,
                                                        output_folder = run_ctx.params.out_folder,
                                                        scene_df = pd.read_csv(Path(run_ctx.dataset.params.processed_folder) / 'scene_index.csv'),
                                                        num_errors_per_scene = num_errors_per_scene)
        case 'HRC_WHU':
            model_evaluator = HRC_WHUModelEvaluator(model = model,
                                                    validation_dataloader = val_dataloader,
                                                    reconstruction_error_fn = reconstruction_error_fn,
                                                    norm_fn = UnitarySymmetricIntervalNorm(scale = 255),
                                                    tile_width = run_ctx.params.get('tile_width', 32),
                                                    tile_height = run_ctx.params.get('tile_height', 32),
                                                    tile_x_step = run_ctx.params.get('tile_x_step', 32),
                                                    tile_y_step = run_ctx.params.get('tile_y_step', 32),
                                                    input_folder = run_ctx.dataset.folder,
                                                    output_folder = run_ctx.params.out_folder,
                                                    scene_df = pd.read_csv(Path(run_ctx.dataset.params.processed_folder) / 'scene_index.csv'),
                                                    num_errors_per_scene = num_errors_per_scene)
        case _:
            raise ValueError(f'[ModelEvaluation] Unknown dataset "{run_ctx.dataset.name}"')

    match run_ctx.model.name:
        case 'DCGANDiscriminator':
            model_evaluator._calculate_reconstruction_error_treshold = lambda: 0.5

    # Step 4: Running evaluation
    scene_metric_df, global_metric_df = model_evaluator.evaluate()
    scene_metric_df.to_csv(Path(run_ctx.params.out_folder) / 'scene_metrics.csv', index = False)
    global_metric_df.to_csv(Path(run_ctx.params.out_folder) / 'global_metrics.csv', index = False)
