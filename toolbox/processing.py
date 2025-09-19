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


def find_all_indices(array, target_values, split=False, prints=True):
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
    if prints:
        for key, value in result.items():
            print(f"{key}: {len(value)} entries")
    if split:
        return result
    else:
        return [i for i, value in enumerate(array) if value in target_values]


def augment_patient_and_cat_specific(
    X, Y_labels, labels, Case, aug_dict: dict, snr_db=20, random_state=42, prints=True,
):
    # create augmented data arrays
    X_aug = [x for x in X]
    Y_labels_aug = [y for y in Y_labels]  # shape () pro Sample
    labels_aug = [l for l in labels]  # shape () pro Sample
    Case_aug = [c for c in Case]

    rng = np.random.default_rng(random_state)
    for aug_cat, n_aug in aug_dict.items():
        if prints:
            print(aug_cat, "to be created:", n_aug)
        select_idx = find_all_indices(labels, [aug_cat], split=False, prints=False)
        if prints:
            print("")
        cat_num = len(select_idx)
        # select all X,Y,... for specific category
        X_in_selcat = X[select_idx]
        Y_labels_in_selcat = Y_labels[select_idx]
        labels_in_selcat = labels[select_idx]
        Case_in_selcat = Case[select_idx]
        # Create a vector with unique patients
        unique_patients = np.unique(Case_in_selcat)

        if prints:
            print("---")
        pat_count_dict = {
            int(num): int(np.sum(Case_in_selcat == num)) for num in unique_patients
        }
        if prints:
            print(f"{aug_cat} {pat_count_dict=}")
            print("---")
        # now: draw as often as needed from the patient list:
        while cat_num <= n_aug:
            # New Case IDs after last in Case_aug
            new_Case = np.max(Case_aug, axis=0) + 1
            # select a random case
            sel_case = np.random.choice(Case_in_selcat)
            for chosen_idx in np.where(Case_in_selcat == sel_case)[0]:
                # append the aug lists with ...[sel_case]
                X_sel = X_in_selcat[chosen_idx]
                # Calculate signal power
                signal_power = np.mean(np.abs(X_sel) ** 2)
                # Convert SNR from dB to linear
                snr_linear = 10 ** (snr_db / 10.0)
                # Noise power per sample
                noise_power = signal_power / snr_linear
                # Gaussian noise
                noise = rng.normal(0, np.sqrt(noise_power), X_sel.shape)
                X_noisy = X_sel + noise
                # Update containers
                X_aug.append(X_noisy)
                Y_labels_aug.append(Y_labels_in_selcat[chosen_idx])
                labels_aug.append(labels_in_selcat[chosen_idx])
                Case_aug.append(new_Case)
                cat_num = len(
                    find_all_indices(labels_aug, [aug_cat], split=False, prints=False)
                )

    X_aug = np.array(X_aug)
    Y_labels_aug = np.array(Y_labels_aug)
    labels_aug = np.array(labels_aug)
    Case_aug = np.array(Case_aug)

    find_all_indices(labels_aug, np.unique(labels_aug), split=False)

    return X_aug, Y_labels_aug, labels_aug, Case_aug
