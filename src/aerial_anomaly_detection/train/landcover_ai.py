"""
Module to implement a model trainer related to the LandCover.ai dataset.
"""


import sys
from typing import Dict, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from aerial_anomaly_detection.train import ModelTrainer
from aerial_anomaly_detection.evaluation.metrics import accuracy, precision, recall, f1


class LandCoverAIModelTrainer(ModelTrainer):
    """
    A class to implement a model trainer related to the LandCover.ai dataset.
    """


    def _test_step(self, reconstruction_error_threshold : float, epoch : int) -> Dict[str, Any]:
        """
        Method to carry out the test step during the epoch.
        NOTE: the test step is simply used for checking model performance in real time.\
              Test information is NEVER used for model selection as can be seen from the code.

        Args:
            reconstruction_error_threshold (float): the reconstruction error threshold used for anomaly prediction.
            epoch (int): the epoch number that launched the test step.

        Returns:
            A dictionary containing relevant test metadata, such as model performance metrics.
        """
        test_metadata : Dict[str, Any] = {}
        confussion_matrix = np.zeros((2, 2))

        self.model.eval()
        with torch.inference_mode():
            for _, X, y in tqdm(self.test_dataloader,
                                desc = 'Testing model performance on dummy test dataset',
                                unit = 'batch',
                                dynamic_ncols = True,
                                file = sys.stdout,
                                leave = True):
                X = X.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_calculation_fn(self.model, X, y_pred, y).item()
                pred_anomaly = int(loss > reconstruction_error_threshold)
                ground_truth_anomaly = int(y != 3)
                confussion_matrix[ground_truth_anomaly, pred_anomaly] += 1

        test_metadata['confussion_matrix'] = confussion_matrix
        test_metadata['test_accuracy'] = accuracy(confussion_matrix)
        test_metadata['test_precision'] = precision(confussion_matrix)
        test_metadata['test_recall'] = recall(confussion_matrix)
        test_metadata['test_f1'] = f1(confussion_matrix)

        return test_metadata
