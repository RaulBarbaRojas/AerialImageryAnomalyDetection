"""
Module to implement a generic training functionality.
"""


import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from aerial_anomaly_detection.datasets import DataLoader
from aerial_anomaly_detection.evaluation.metrics import accuracy, precision, recall, f1


class ModelTrainer:
    """
    Class to implement the generic training functionality.
    """


    def __init__(self, model : torch.nn.Module, train_dataloader : DataLoader, val_dataloader : DataLoader,
                 loss_calculation_fn : Callable[[torch.nn.Module, NDArray, NDArray, NDArray], float],
                 optimizer : torch.optim.Optimizer, device : torch.device, output_folder : str | Path) -> None:
        """
        Constructor method for the ModelTrainer class.

        Args:
            model (torch.nn.Module): the model to be trained.
            train_dataloader (DataLoader): the train dataloader containing training data.
            train_dataloader (DataLoader): the train dataloader containing validation data.
            loss_calculation_fn (Callable[[torch.nn.Module, NDArray, NDArray, NDArray], float]): a function to calculate\
                the loss based on the model, input data, predicted data and input data ground truth (typically anomaly class).
            optimizer (torch.nn.Optimizer): the optimizer to be used for improving the model and reducing the loss.
            device (torch.device): the device to be used for training the model.
            output_folder (str | Path): the output folder where training metadata will be stored.
        """

        # Step 1: Setting up training attributes
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_calculation_fn = loss_calculation_fn
        self.optimizer = optimizer
        self.device = device
        self.output_folder = Path(output_folder)

        # Step 2: Preparing training
        self.output_folder.mkdir(exist_ok = True, parents = True)
        self.train_history : Dict[int, Dict[str, Any]] = {}


    def _train_step(self) -> Dict[str, Any]:
        """
        Method to carry out the train step during the epoch.

        Returns:
            A dictionary containing relevant training metadata, such as the epoch train loss.
        """
        epoch_train_loss = 0.0

        self.model.train()
        for _, X, y in tqdm(self.train_dataloader,
                            desc = 'Train step',
                            unit = 'epoch',
                            file = sys.stdout,
                            dynamic_ncols = True,
                            leave = True):
            X = X.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_calculation_fn(self.model, X, y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(self.train_dataloader)

        return {'train_loss' : epoch_train_loss}


    def _validation_step(self) -> Dict[str, Any]:
        """
        Method to carry out the validation step during the epoch.

        Returns:
            A dictionary containing relevant validation metadata, such as the epoch validation loss.
        """
        epoch_val_loss = 0.0

        self.model.eval()
        with torch.inference_mode():
            for _, X, y in tqdm(self.val_dataloader,
                                desc = 'Validation step',
                                unit = 'batch',
                                dynamic_ncols = True,
                                file = sys.stdout,
                                leave = True):
                X = X.to(self.device)
                y_pred = self.model(X, mode = 'inference')
                loss = self.loss_calculation_fn(self.model, X, y_pred, y)
                epoch_val_loss += loss.item()

            epoch_val_loss /= len(self.val_dataloader)

        return {'val_loss' : epoch_val_loss}


    def _train_one_epoch(self) -> Dict[str, Any]:
        """
        Method to train the given model during an epoch.
        """
        train_metadata = self._train_step()
        val_metadata = self._validation_step()

        return train_metadata | val_metadata


    def train_n_epochs(self, n_epochs : int) -> Dict[int, Dict[str, Any]]:
        """
        Method to train the given model during N epochs.

        Args:
            n_epochs (int): the number of epochs to train the model.

        Returns:
            A dictionary containing the training history metadata with loss values and performance metrics per epoch.
        """
        best_val_loss = np.inf

        for epoch in tqdm(range(n_epochs),
                          desc = 'Training the model',
                          unit = 'epoch',
                          file = sys.stdout,
                          dynamic_ncols = True):
            self.train_history[epoch] = self._train_one_epoch()
            if best_val_loss > self.train_history[epoch]['val_loss']:
                best_val_loss = self.train_history[epoch]['val_loss']
                self._save_model('best')

            print(f'Epoch {epoch} | Train loss: {self.train_history[epoch]['train_loss']:.6f} | '
                  f'Validation loss: {self.train_history[epoch]['val_loss']:.6f}')

        return self.train_history


    def _plot_loss_curve(self) -> None:
        """
        Method to plot the loss curve of the trained model storing the plot in the output folder.
        """
        plt.figure(figsize = (10, 7))
        plt.style.use('ggplot')

        if len(self.train_history.keys()) != 0:
            epochs = list(self.train_history.keys())
            train_loss = [self.train_history[epoch]['train_loss'] for epoch in range(len(self.train_history.keys()))]
            val_loss = [self.train_history[epoch]['val_loss'] for epoch in range(len(self.train_history.keys()))]

            plt.plot(epochs, train_loss, label = 'Train loss', color = 'blue')
            plt.plot(epochs[-1], train_loss[-1], 'o', color = 'blue', markersize = 10)        
            plt.plot(epochs, val_loss, label = 'Validation loss', color = 'red')
            plt.plot(epochs[-1], val_loss[-1], 'o', color = 'red', markersize = 10)

            plt.title('Train and validation loss curves comparison', fontsize = 18, fontweight = 'bold')
            plt.xlabel('Epoch', fontsize = 16, fontweight = 'bold')
            plt.ylabel('Loss', fontsize = 16, fontweight = 'bold')
            plt.tight_layout()

        plt.savefig(self.output_folder / 'loss_curve.png')
        plt.close()


    def _save_train_history(self) -> None:
        """
        Method to save the train history as a pickle file in the output folder.
        """
        with open(self.output_folder / 'train_history.pkl', mode = 'wb') as train_history_fp:
            pickle.dump(self.train_history, train_history_fp)


    def _save_model(self, model_name : str = 'best') -> None:
        """
        Method to save the model weights in the output folder.
        """
        torch.save(self.model.state_dict(), self.output_folder / f'{model_name}.pth')


    def save(self) -> None:
        """
        Method to save model training results (model weights, loss curve and train history) in the output folder.
        """
        self._save_model('last')
        self._save_train_history()
        self._plot_loss_curve()


class ZizModelTrainer(ModelTrainer):
    """
    A class to implement the training of the ziz architecture.
    """


    def _train_step(self) -> Dict[str, Any]:
        """
        Method to carry out the train step during the epoch.

        Returns:
            A dictionary containing relevant training metadata, such as the epoch train loss.
        """
        epoch_train_loss = 0.0

        self.model.train()
        for _, X, y in tqdm(self.train_dataloader,
                            desc = 'Train step',
                            unit = 'epoch',
                            file = sys.stdout,
                            dynamic_ncols = True,
                            leave = True):
            sample_noise_vector = torch.randn((X.shape[0], self.model.latent_dimension), device = self.device)
            y_pred = self.model(sample_noise_vector, mode = 'train')
            loss = self.loss_calculation_fn(self.model, sample_noise_vector, y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(self.train_dataloader)

        return {'train_loss' : epoch_train_loss}
