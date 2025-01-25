"""
Module to implement the training script of the AutoEncoder.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.datasets.landcover_ai import LandCoverAI
from aerial_anomaly_detection.datasets import DataLoader
from aerial_anomaly_detection.models.autoencoder import AutoEncoder



if __name__ == '__main__':

    # Step 1: Preparing the training process
    device = torch.device('cuda')
    batch_size = run_ctx.params.get('batch_size', 256)

    (out_folder := Path(run_ctx.params.out_folder) / run_ctx.model.name).mkdir(exist_ok = True, parents = True)
    processed_dataset_folder = Path(run_ctx.dataset.params.processed_folder)

    train_dataset = LandCoverAI.load(processed_dataset_folder, partition = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = os.cpu_count())

    val_dataset = LandCoverAI.load(processed_dataset_folder, partition = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = os.cpu_count())

    test_dataset = LandCoverAI.load(processed_dataset_folder, partition = 'test')
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # Step 2: Setting up the model to be trained
    model = AutoEncoder(run_ctx.model.params.latent_dimension,
                        run_ctx.model.params.img_width,
                        run_ctx.model.params.img_height).to(device)

    # Step 3: Defining training parameters
    learning_rate = run_ctx.params.get('learning_rate', 0.001)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
    n_epoch = run_ctx.params.get('n_epochs', 100)
    min_val_loss = np.inf

    # Step 4: Training the model
    for epoch in tqdm(range(n_epoch),
                      desc = 'Training convolutional autoencoder',
                      unit = 'epoch',
                      file = sys.stdout,
                      dynamic_ncols = True):
        model.train()
        epoch_train_loss = 0
        epoch_val_loss = 0

        # Step 4.1: Train step
        for _, X, _ in tqdm(train_dataloader,
                            desc = 'Train step',
                            unit = 'epoch',
                            file = sys.stdout,
                            dynamic_ncols = True,
                            leave = True):
            X = X.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)

        # Step 4.2: Validation step
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
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_dataloader)

        # Step 4.3: Checking test accuracy (unused for model selection, simply informative)
        confussion_matrix = np.zeros((2, 2))

        model.eval()
        with torch.inference_mode():
            for _, X, y in tqdm(test_dataloader,
                                desc = 'Test accuracy check',
                                unit = 'batch',
                                dynamic_ncols = True,
                                file = sys.stdout,
                                leave = True):
                X = X.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, X)
                pred_anomaly = int(loss > epoch_val_loss)
                ground_truth_anomaly = int(y != 3)
                confussion_matrix[ground_truth_anomaly, pred_anomaly] += 1

        accuracy = (confussion_matrix[0, 0] + confussion_matrix[1, 1]) / confussion_matrix.sum()
        precision = confussion_matrix[1, 1] / (confussion_matrix[:, 1].sum())
        recall = confussion_matrix[1, 1] / (confussion_matrix[1, :].sum())
        f1 = 2 * precision * recall / (precision + recall)

        print(f'Epoch {epoch} | Train loss: {epoch_train_loss:.6f} | Val loss: {epoch_val_loss:.6f}')
        print('Confussion matrix: \n', confussion_matrix)
        print(f'Accuracy: {accuracy:.6f}')
        print(f'Precision: {precision:.6f}')
        print(f'Recall: {recall:.6f}')
        print(f'F1: {f1:.6f}')

        # Step 4.4: Saving the model if the result improves the current best model (validation-wise)
        if min_val_loss > epoch_val_loss:
            min_val_loss = epoch_val_loss
            torch.save(model.state_dict(), out_folder / 'model.pth')
