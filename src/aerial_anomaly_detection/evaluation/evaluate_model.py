"""
Module to implement a generic model evaluation over the desired dataset.
"""


import os
from pathlib import Path

import pandas as pd
import torch
from afml.context import run_ctx

from aerial_anomaly_detection.datasets import DataLoader
from aerial_anomaly_detection.datasets.landcover_ai import LandCoverAI
from aerial_anomaly_detection.evaluation.datasets.landcover_ai import LandCoverAIModelEvaluator
from aerial_anomaly_detection.preprocess.norm.functions.unitary_symmetric_interval_norm import UnitarySymmetricIntervalNorm


if __name__ == '__main__':

    # Step 1: Setting up the model
    model : torch.nn.Module = run_ctx.model.build()
    pretrained_weights_filepath = Path(model.model_settings.get('pretrained_weights', ''))
    if pretrained_weights_filepath.is_file():
        loaded_keys = model.load_state_dict(torch.load(pretrained_weights_filepath, weights_only = True))
        print(f'[ModelEvaluation] {run_ctx.model.name} weights loaded: {loaded_keys}')

    # Step 2: Setting up validation data for reconstruction error threshold calculation
    batch_size = run_ctx.params.get('batch_size', 256)

    match run_ctx.dataset.name:
        case 'LandCoverAI':
            val_dataset = LandCoverAI.load(run_ctx.dataset.params.processed_folder, partition = 'val')
            loss_fn = torch.nn.MSELoss()
            reconstruction_error_fn = lambda model, X, y_pred, y: loss_fn(y_pred, X).item()
        case _:
            raise ValueError(f'Unknown dataset "{run_ctx.dataset.name}"')

    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count())

    # Step 3: Setting up model evaluator
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
                                                        scene_df = pd.read_csv(Path(run_ctx.dataset.params.processed_folder) / 'scene_index.csv'))
        case _:
            raise ValueError(f'Unknown dataset "{run_ctx.dataset.name}"')

    # Step 4: Running evaluation
    scene_metric_df, global_metric_df = model_evaluator.evaluate()
    scene_metric_df.to_csv(Path(run_ctx.params.out_folder) / 'scene_metrics.csv', index = False)
    global_metric_df.to_csv(Path(run_ctx.params.out_folder) / 'global_metrics.csv', index = False)
