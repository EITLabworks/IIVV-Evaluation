from . import z_score
from glob import glob
from os.path import join

import numpy as np


def load_data(path: str = "data/", Cutpats_threshold=60, norm="freq", print_info=True):
    """
    Loads, filters, and normalizes impedance data from .npz files.

    Parameters
    ----------
    path : str, optional
        Path to the directory containing the `.npz` files. Default is "data/".
    Cutpats_threshold : float, optional
        Threshold for filtering samples based on the percentage of cut pixels (`Avg_Cutpats`).
        Only samples with `Avg_Cutpats` less than this value will be retained. Default is 60.
    norm : str, optional
        Specifies the normalization mode for `Zdata`:
        - "global" : Normalize over all data globally.
        - "freq" : Normalize along each frequency (axis 1).
        Default is "freq".
    print_info : bool, optional
        If True, prints information during processing. Default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - py_idx : np.ndarray of int
        - patient_no : np.ndarray of int
        - patient_sample : np.ndarray of int
        - Zdata : np.ndarray of float
            Normalized and filtered impedance data (absolute values).
        - Avg_Cutpats : np.ndarray of float
            Average cut pixel percentages.
        - LesionAssessment : np.ndarray of str
            Labels related to lesion assessment.
        - label_2class : np.ndarray of str
        - labels_4class : np.ndarray of str
        - labels_WHOclass : np.ndarray of str

    Notes
    -----
    - Only `.npz` files within the specified directory are loaded.
    - Zdata is converted to absolute values before normalization.
    - NaNs in Zdata are replaced with zeros after normalization.
    - The function assumes each `.npz` file contains the same set of expected keys.

    """

    def remaining_sample_count(Zdata):
        print("\nRemaining data samples:", Zdata.shape[0], "")

    py_idx = []
    patient_no = []
    patient_sample = []
    Zdata = []
    Avg_Cutpats = []
    LesionAssessment = []
    label_2class = []
    labels_4class = []
    labels_WHOclass = []

    files = glob(join(path, "*.npz"))
    for file in files:
        tmp = np.load(file, allow_pickle=True)
        py_idx.append(tmp["py_idx"])
        patient_no.append(tmp["patient_no"])
        patient_sample.append(tmp["patient_sample"])
        Zdata.append(tmp["Zdata"])
        Avg_Cutpats.append(tmp["Avg_Cutpats"])
        LesionAssessment.append(tmp["LesionAssessment"])
        label_2class.append(tmp["label_2class"])
        labels_4class.append(tmp["labels_4class"])
        labels_WHOclass.append(tmp["labels_WHOclass"])

    py_idx = np.array(py_idx, dtype=int)
    patient_no = np.array(patient_no, dtype=int)
    patient_sample = np.array(patient_sample, dtype=int)
    Zdata = np.array(Zdata, dtype=complex)
    Avg_Cutpats = np.array(Avg_Cutpats, dtype=float)
    LesionAssessment = np.array(LesionAssessment, dtype=str)
    label_2class = np.array(label_2class, dtype=str)
    labels_4class = np.array(labels_4class, dtype=str)
    labels_WHOclass = np.array(labels_WHOclass, dtype=str)

    # Zdata to absolute values
    print("\n\tInfo:\tnp.abs(Zdata)")
    Zdata = np.abs(Zdata)

    if print_info:
        remaining_sample_count(Zdata)

    # Threshold selection
    print(f"\tInfo:\tSelect only data Avg_Cutpats <{Cutpats_threshold}%")
    dropout = np.where(Avg_Cutpats < Cutpats_threshold)[0]

    py_idx = py_idx[dropout]
    patient_no = patient_no[dropout]
    patient_sample = patient_sample[dropout]
    Zdata = Zdata[dropout]
    Avg_Cutpats = Avg_Cutpats[dropout]
    LesionAssessment = LesionAssessment[dropout]
    label_2class = label_2class[dropout]
    labels_4class = labels_4class[dropout]
    labels_WHOclass = labels_WHOclass[dropout]

    if print_info:
        remaining_sample_count(Zdata)

    # Normalization
    print("\n\tInfo:\tz-score Zdata")
    Zdata = z_score(Zdata, norm, print_info)

    # Replace nan with 0
    print("\n\tInfo:\tnp.nan_to_num(Zdata)")
    Zdata = np.nan_to_num(Zdata)

    if print_info:
        remaining_sample_count(Zdata)

    return (
        py_idx,
        patient_no,
        patient_sample,
        Zdata,
        Avg_Cutpats,
        LesionAssessment,
        label_2class,
        labels_4class,
        labels_WHOclass,
    )
