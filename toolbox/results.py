import numpy as np
from .loading import load_data
from .processing import find_all_indices, augment_patient_and_cat_specific
from .decoding_encoding import hot_coding
from .visualization import visualize_Category_hist


def Results_data_loading(class_task, data_augmentation=True):
    """
    return X, Y_labels, labels, Case
    """
    (
        py_idx,
        patient_no,
        patient_sample,
        Zdata,
        Avg_Cutpats,
        LesionAssessment,
        labels_healthy_vs_cancer,
        labels_cancer_vs_hbd,
        labels_MC,
    ) = load_data(path="data", Cutpats_threshold=60, norm="freq")
    if class_task == "healthy_vs_cancer":
        print("Healthy vs. cancer\n")
        select_idx = find_all_indices(
            labels_healthy_vs_cancer, ["Healthy", "Cancer"], split=False
        )
        labels = labels_healthy_vs_cancer[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        # for SVD kernel boundary
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            1: "#003F5C",  # healthy
            0: "#FFA600",  # cancer
        }
        cp_l = {
            1: "healthy",  # healthy
            0: "cancer",  # cancer
        }
        visualize_Category_hist(labels)

    elif class_task == "cancer_vs_hbd":
        print("\nHealthy vs. HDB\n")
        select_idx = find_all_indices(
            labels_cancer_vs_hbd, ["Non-cancer", "Cancer"], split=False
        )
        labels = labels_cancer_vs_hbd[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            1: "#DD5182",  # non-cancer
            0: "#FFA600",  # cancer
        }
        cp_l = {
            1: "healthy",  # non-cancer
            0: "cancer",  # cancer
        }
        visualize_Category_hist(labels)

    elif class_task == "MC":
        print("\nMC\n")
        select_idx = find_all_indices(
            labels_MC,
            ["Healthy", "Cancer", "Severe dysplasia", "Benign"],
            split=False,
        )
        labels = labels_MC[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            0: "#58508D",  # Benign
            1: "#FFA600",  # Cancer
            2: "#003F5C",  # Healthy
            3: "#FF6361",  # Severe dysplasia
        }
        cp_l = {
            0: "Benign",
            1: "Cancer",
            2: "Healthy",
            3: "Severe dysplasia",
        }
        aug_dict = {
            "Severe dysplasia": 70,
            "Benign": 70,
        }

    Case = patient_no[select_idx]
    X = Zdata[select_idx, ...]
    print("\nInverse mapping:", inverse_mapping)
    print(X.shape, Y.shape, Case.shape)

    if class_task == "MC" and data_augmentation:
        print("\n\t\tBefore augmentation")
        visualize_Category_hist(labels)
        X, Y_labels, labels, Case = augment_patient_and_cat_specific(
            X, Y_labels, labels, Case, aug_dict
        )
        print("\n\t\tAfter augmentation")
        visualize_Category_hist(labels)
    return X, Y_labels, labels, Case
