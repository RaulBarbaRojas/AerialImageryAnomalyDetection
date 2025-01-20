"""
Module to implement the training script of the AutoEncoder.
"""

import os
from pathlib import Path

import numpy as np
import torch
from afml.context import run_ctx
from tqdm.auto import tqdm

from aerial_anomaly_detection.datasets.masati_v2 import MASATIv2
from aerial_anomaly_detection.datasets import DataLoader
from aerial_anomaly_detection.models.autoencoder import AutoEncoder



if __name__ == '__main__':
    device = torch.device('cuda')
    model = AutoEncoder(run_ctx.model.params.latent_dimension,
                        run_ctx.model.params.img_width,
                        run_ctx.model.params.img_height).to(device)
    (out_folder := Path(run_ctx.params.out_folder) / model.name).mkdir(exist_ok = True, parents = True)
    train_dataset = MASATIv2.load(r'data\processed\MASATI-v2', partition = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = run_ctx.params.batch_size, shuffle = True, num_workers = os.cpu_count())
    val_dataset = MASATIv2.load(r'data\processed\MASATI-v2', partition = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = run_ctx.params.batch_size, shuffle = False, num_workers = os.cpu_count())

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(params = model.parameters(), lr = 0.001)
    n_epoch = run_ctx.params.n_epochs
    min_val_loss = np.inf

    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0
        epoch_val_loss = 0

        for _, X, _ in tqdm(train_dataloader, desc = 'Training model', unit = 'epoch'):
            X = X.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        with torch.inference_mode():
            for _, X, _ in tqdm(val_dataloader, desc = 'Validating model', unit = 'epoch'):
                X = X.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, X)
                epoch_val_loss += loss.item()

        if min_val_loss > (epoch_val_loss / len(val_dataloader)):
            min_val_loss = epoch_val_loss / len(val_dataloader)
            torch.save(model.state_dict(), out_folder / 'model.pth')

        print(f'Epoch {epoch} | Train loss: {epoch_train_loss / len(train_dataloader):.6f} | Val loss: {epoch_val_loss / len(val_dataloader):.6f}')
