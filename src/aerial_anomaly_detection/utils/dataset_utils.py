"""
A module containing general utilities common to the usage of all datasets.
"""


from typing import Tuple

def str_to_value_list(str_tuple : str) -> Tuple[float, ...]:
    """
    Method to convert a string into a value tuple.

    Args:
        str_tuple (str): the string representation of the tuple with values.

    Returns:
        The real tuple with float values in it.
    """
    return [float(value) for value in str_tuple[1:-1].replace(' ', '').split(',')]
