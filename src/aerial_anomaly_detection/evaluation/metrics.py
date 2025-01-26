"""
Module to provide with functions for calculating different ML metrics.
"""


from numpy.typing import NDArray


def accuracy(confussion_matrix : NDArray) -> float:
    """
    Function to calculate accuracy given a confussion matrix.

    Args:
        confussion_matrix (NDArray): the confussion matrix to be used for calculating model accuracy.

    Returns:
        Model accuracy based on the given confussion matrix.
    """
    return (confussion_matrix[0, 0] + confussion_matrix[1, 1]) / confussion_matrix.sum()


def precision(confussion_matrix : NDArray) -> float:
    """
    Function to calculate precision given a confussion matrix.

    Args:
        confussion_matrix (NDArray): the confussion matrix to be used for calculating model precision.

    Returns:
        Model precision based on the given confussion matrix.
    """
    return confussion_matrix[1, 1] / (confussion_matrix[:, 1].sum())


def recall(confussion_matrix : NDArray) -> float:
    """
    Function to calculate recall given a confussion matrix.

    Args:
        confussion_matrix (NDArray): the confussion matrix to be used for calculating model recall.

    Returns:
        Model recall based on the given confussion matrix.
    """
    return confussion_matrix[1, 1] / (confussion_matrix[1, :].sum())


def f1(confussion_matrix : NDArray) -> float:
    """
    Function to calculate f1 given a confussion matrix.

    Args:
        confussion_matrix (NDArray): the confussion matrix to be used for calculating model f1.

    Returns:
        Model f1 based on the given confussion matrix.
    """
    precision_val = precision(confussion_matrix)
    recall_val = recall(confussion_matrix)
    return (2 * precision_val * recall_val) / (precision_val + recall_val)
