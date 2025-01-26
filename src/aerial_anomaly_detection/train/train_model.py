"""
Script to train a generic model.
"""


import os
from pathlib import Path

import torch
from afml.context import run_ctx

from aerial_anomaly_detection.datasets import DataLoader
from aerial_anomaly_detection.datasets.landcover_ai import LandCoverAI
from aerial_anomaly_detection.train import ModelTrainer


if __name__ == '__main__':

    # Step 1: Setting up model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model : torch.nn.Module = run_ctx.model.build()
    model_pretrained_weights_filepath = Path(model.model_settings.get('pretrained_weights', ''))

    if model_pretrained_weights_filepath.is_file():
        loaded_keys = model.load_state_dict(torch.load(model_pretrained_weights_filepath, weights_only = True))
        print(f'[ModelTraining] {run_ctx.model.name} weights loaded: {loaded_keys}')

    model = model.to(device)

    # Step 2: Setting up data
    processed_dataset_folder = Path(run_ctx.dataset.params.processed_folder)
    batch_size = run_ctx.params.batch_size

    match run_ctx.dataset.name:
        case 'LandCoverAI':
            train_dataset = LandCoverAI.load(processed_dataset_folder, partition = 'train')
            train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = os.cpu_count())

            val_dataset = LandCoverAI.load(processed_dataset_folder, partition = 'val')
            val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count())

            test_dataset = LandCoverAI.load(processed_dataset_folder, partition = 'test')
            test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
        case _:
            raise ValueError(f'[ModelTraining] Unknown dataset "{run_ctx.dataset.name}"')

    # Step 3: Setting up training parameters
    n_epochs = run_ctx.params.get('n_epochs', 200)
    learning_rate = run_ctx.params.get('learning_rate', 0.0001)

    match run_ctx.model.name:
        case 'AutoEncoder':
            loss_fn = torch.nn.MSELoss()
            loss_calculation_fn = lambda model, X, y_pred, y: loss_fn(y_pred, X)
            optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
        case _:
            raise ValueError(f'[ModelTraining] Unknown model "{run_ctx.dataset.name}"')

    # Step 4: Running model training
    (output_folder := Path(run_ctx.params.out_folder)).mkdir(exist_ok = True, parents = True)
    model_trainer = ModelTrainer(model = model,
                                 train_dataloader = train_dataloader,
                                 val_dataloader = val_dataloader,
                                 test_dataloader = test_dataloader,
                                 loss_calculation_fn = loss_calculation_fn,
                                 optimizer = optimizer,
                                 device = device,
                                 output_folder = output_folder)
    model_trainer.train_n_epochs(n_epochs)
    model_trainer.save()
