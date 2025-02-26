"""
Module to provide general utility functions to be used for any model evaluation.
"""


import sys
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
from pandas import DataFrame
from tqdm.auto import tqdm

from aerial_anomaly_detection.datasets import DataLoader


class ModelEvaluator(ABC):
    """
    Abstract class to implement generic model evaluation over a dataset.
    """


    def __init__(self, model : torch.nn.Module, validation_dataloader : DataLoader,
                 reconstruction_error_fn : Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], float],
                 num_errors_per_scene : int = 48) -> None:
        """
        Constructor method for the model evaluator class.

        Args:
            model (torch.nn.Module): the model to be evaluated.
            validation_dataloader (DataLoader): the validation dataloader containing data to calculate the reconstruction error threshold.
            reconstruction_error_fn (Callable[[Any], float]): the function to calculate the reconstruction error threshold,\
                from the model, the input tensor X, the predicted output y_pred, and the ground truth label of the input y.
            num_errors_per_scene (int): the number of wrong predictions per scene to be visualized (by default 16).
        """

        # Step 1: Setting the base attributes
        self.model = model
        self.validation_dataloader = validation_dataloader
        self.reconstruction_error_fn = reconstruction_error_fn
        self.num_errors_per_scene = num_errors_per_scene

        # Step 2: Setting up device agnostic code
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)


    def _calculate_reconstruction_error_treshold(self) -> float:
        """
        Method to calculate the reconstruction error threshold to be used.

        Returns:
            The reconstruction error threshold.
        """
        reconstruction_error_threshold = 0

        self.model.eval()
        with torch.inference_mode():
            for _, X, y in tqdm(self.validation_dataloader,
                                desc = 'Validation step',
                                unit = 'batch',
                                dynamic_ncols = True,
                                file = sys.stdout,
                                leave = True):
                X = X.to(self.device)
                y_pred = self.model(X, mode = 'inference')
                reconstruction_error_threshold += self.reconstruction_error_fn(self.model, X, y_pred, y)

        reconstruction_error_threshold /= len(self.validation_dataloader)
        return reconstruction_error_threshold


    @abstractmethod
    def evaluate(self) -> Tuple[DataFrame, DataFrame]:
        """
        Method to evaluate the model over the given dataset under the given settings.

        Returns:
            A tuple with local-based metric results DataFrame (e.g., per scene) and a global-based results\
            DataFrame (e.g., across all scenes).
        """
