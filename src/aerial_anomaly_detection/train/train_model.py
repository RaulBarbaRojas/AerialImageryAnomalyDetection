"""
Script to train a generic model.
"""


import os
from pathlib import Path

import torch
from afml.context import run_ctx

from aerial_anomaly_detection.datasets import DataLoader, Dataset
from aerial_anomaly_detection.train import ModelTrainer, ZizModelTrainer


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
        case 'LandCoverAI' | 'HRC_WHU': # Same dataset interface works for both
            train_dataset = Dataset.load(processed_dataset_folder, partition = 'train')
            train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = os.cpu_count())

            val_dataset = Dataset.load(processed_dataset_folder, partition = 'val')
            val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count())
        case _:
            raise ValueError(f'[ModelTraining] Unknown dataset "{run_ctx.dataset.name}"')

    # Step 3: Setting up training parameters
    n_epochs = run_ctx.params.get('n_epochs', 200)
    learning_rate = run_ctx.params.get('learning_rate', 0.0001)

    match run_ctx.model.name:
        case 'AutoEncoder' | 'Izi':
            loss_fn = torch.nn.MSELoss()
            loss_calculation_fn = lambda model, X, y_pred, y: loss_fn(y_pred, X)
            optimizer = torch.optim.Adam(params = filter(lambda param: param.requires_grad, model.parameters()), lr = learning_rate)
            model_trainer_class = ModelTrainer
        case 'Ziz':
            loss_fn = torch.nn.MSELoss()
            loss_calculation_fn = lambda model, X, y_pred, y: loss_fn(y_pred, X)
            optimizer = torch.optim.Adam(params = filter(lambda param: param.requires_grad, model.parameters()), lr = learning_rate)
            model_trainer_class = ZizModelTrainer
        case 'f-AnoGAN':
            loss_fn = torch.nn.MSELoss()
            loss_calculation_fn = lambda model, X, y_pred, y: loss_fn(X, y_pred[0]) + loss_fn(y_pred[1], y_pred[2])
            optimizer = torch.optim.Adam(params = filter(lambda param: param.requires_grad, model.parameters()), lr = learning_rate)
            model_trainer_class = ModelTrainer
        case _:
            raise ValueError(f'[ModelTraining] Unknown model "{run_ctx.dataset.name}"')

    # Step 4: Running model training
    (output_folder := Path(run_ctx.params.out_folder)).mkdir(exist_ok = True, parents = True)
    model_trainer = model_trainer_class(model = model,
                                        train_dataloader = train_dataloader,
                                        val_dataloader = val_dataloader,
                                        loss_calculation_fn = loss_calculation_fn,
                                        optimizer = optimizer,
                                        device = device,
                                        output_folder = output_folder)
    model_trainer.train_n_epochs(n_epochs)
    model_trainer.save()
