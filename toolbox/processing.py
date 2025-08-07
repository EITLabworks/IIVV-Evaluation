import numpy as np
from typing import Union


def assessment_wrapping(lesion_assessment: Union[list, np.ndarray], mapping: dict):
    """
    Applies a mapping to a list or array of lesion assessments and returns the new labeled array.

    Parameters:
        lesion_assessment (list or np.ndarray): List or array of original lesion assessment labels.
        mapping (dict): Dictionary mapping original labels to new labels (e.g., 'Cancer' -> 1).

    Returns:
        np.ndarray: A new NumPy array with the mapped labels.
    """
    # create a copy of e.g., lesion assessment
    lesion_assessment = np.array(lesion_assessment)  # ensure it's a NumPy array
    labeled = np.array([mapping[label] for label in lesion_assessment])
    return labeled


def find_all_indices(array, target_values, split=False):
    """
    Returns the indices of all occurrences of specified target values in a list.

    Parameters:
        array (list): The list to search through.
        target_values (str or list): A single string or a list of strings to search for.
        split (bool, optional): If True, returns a dictionary mapping each target value
                                to a list of its indices. If False (default), returns a
                                flat list of all matching indices.

    Returns:
        list or dict:
            - If split is False: a list of indices where any target value occurs in the array.
            - If split is True: a dictionary where each key is a target value and the value
              is a list of indices where that value appears in the array.

    Example:
        >>> array = ['a', 'b', 'c', 'a', 'b']
        >>> find_all_indices(array, 'a')
        [0, 3]
        >>> find_all_indices(array, ['a', 'b'])
        [0, 1, 3, 4]
        >>> find_all_indices(array, ['a', 'b'], split=True)
        {'a': [0, 3], 'b': [1, 4]}
    """

    if isinstance(target_values, str):
        target_values = [target_values]
    result = {val: [] for val in target_values}

    for i, value in enumerate(array):
        for target in target_values:
            if value == target:
                result[target].append(i)

    for key, value in result.items():
        print(f"{key}: {len(value)} entries")
    if split:
        return result
    else:
        return [i for i, value in enumerate(array) if value in target_values]
