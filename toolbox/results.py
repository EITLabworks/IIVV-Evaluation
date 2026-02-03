import numpy as np
from .loading import load_data
from .processing import find_all_indices, augment_patient_and_cat_specific
from .decoding_encoding import hot_coding
from .visualization import visualize_Category_hist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize


def Results_data_loading(class_task, data_augmentation=True):
    """
    return X, Y_labels, labels, Case, inverse_mapping, Y
    """
    (
        py_idx,
        patient_no,
        patient_sample,
        Zdata,
        Avg_Cutpats,
        LesionAssessment,
        labels_healthy_vs_cancer,
        labels_cancer_vs_lr_hr,
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

    elif class_task == "cancer_vs_lr_hr":
        print("\nHealthy vs. LR vs. HR\n")
        select_idx = find_all_indices(
            labels_cancer_vs_lr_hr, ["Cancer", "High risk", "Low risk"], split=False
        )
        labels = labels_cancer_vs_lr_hr[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            0: "#FFA600",  # cancer
            1: "#FF6361",  # "High risk
            2: "#58508D",  # "Low risk
        }
        cp_l = {
            0: "cancer",  # cancer
            1: "High risk",  # High risk
            2: "Low risk",  # Low risk
        }
        visualize_Category_hist(labels)

    elif class_task == "MC":
        print("\nMC\n")
        select_idx = find_all_indices(
            labels_MC,
            ["Healthy", "Cancer", "High-grade dysplasia", "Non-malignant"],
            split=False,
        )
        labels = labels_MC[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            3: "#58508D",  # Non-malignant
            0: "#FFA600",  # Cancer
            1: "#003F5C",  # Healthy
            2: "#FF6361",  # High-grade dysplasia
        }
        cp_l = {
            0: "Cancer",
            1: "Healthy",
            2: "High-grade dysplasia",
            3: "Non-malignant",
        }

        aug_dict = {
            "High-grade dysplasia": 70,
            "Non-malignant": 70,
        }
        visualize_Category_hist(labels)
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
    return X, Y_labels, labels, Case, inverse_mapping, Y


def ResultsROCAUC(class_task, y_true, y_score, Y_labels, s_name=None):
    # Color + label maps
    label_roc_plot = {
        "healthy_vs_cancer": "Healthy vs. cancer",
        "cancer_vs_lr_hr": "Cancer vs. low risk vs. high risk",
    }

    if class_task == "healthy_vs_cancer":
        cp = {
            1: "#003F5C",  # healthy
            0: "#FFA600",  # cancer
        }
        cp_l = {
            1: "healthy",  # healthy
            0: "cancer",  # cancer
        }
    elif class_task == "cancer_vs_lr_hr":
        cp = {
            0: "#FFA600",  # cancer
            1: "#FF6361",  # "High risk
            2: "#58508D",  # "Low risk
        }
        cp_l = {
            0: "Cancer",  # cancer
            1: "High risk",  # High risk
            2: "Low risk",  # Low risk
        }
    elif class_task == "MC":
        cp = {
            3: "#58508D",  # Non-malignant
            0: "#FFA600",  # Cancer
            1: "#003F5C",  # Healthy
            2: "#FF6361",  # High-grade dysplasia
        }
        cp_l = {
            0: "Cancer",
            1: "Healthy",
            2: "High-grade dysplasia",
            3: "Non-malignant",
        }

    # ----- ROC Curve plotting -----
    classes = np.unique(Y_labels)
    n_classes = len(classes)

    plt.figure(figsize=(5.5, 3))

    if n_classes == 2:
        # Binary case
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=cp[1],
            label=f"{label_roc_plot[class_task]} (AUC = {roc_auc:.2f})",
        )
        print("AUC = ", roc_auc)
        np.savez(
            s_name.split(".pdf")[0] + ".npz",
            fpr=fpr,
            tpr=tpr,
            color=cp[1],
            roc_auc=roc_auc,
            label=f"{label_roc_plot[class_task]} (AUC = {roc_auc:.2f})",
            class_task=class_task,
        )
    else:
        # Multi-class case: binarize labels
        y_bin = label_binarize(y_true, classes=classes)
        MC_save_npz = {}
        # One-vs-rest ROC for each class
        for i in classes:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)

            # Flip AUC if < 0.5
            if roc_auc < 0.5:
                fpr, tpr = 1 - fpr, 1 - tpr
                roc_auc = 1 - roc_auc

            plt.plot(fpr, tpr, color=cp[i], label=f"{cp_l[i]} (AUC = {roc_auc:.2f})")
            # prepare saving
            MC_save_npz[cp_l[i]] = (fpr, tpr, cp[i], f"{cp_l[i]} (AUC = {roc_auc:.2f})")
        # Micro-average ROC
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        # Flip AUC if < 0.5
        if roc_auc < 0.5:
            fpr, tpr = 1 - fpr, 1 - tpr
            roc_auc = 1 - roc_auc
        plt.plot(
            fpr,
            tpr,
            color="black",
            linestyle="--",
            label=f"Micro-average (AUC = {roc_auc:.2f})",
        )
        print("Micro-average AUC = ", roc_auc)
        MC_save_npz["Micro-average"] = (
            fpr,
            tpr,
            "black",
            "Micro-average (AUC = {roc_auc:.2f})",
        )
        np.savez(
            s_name.split(".pdf")[0] + ".npz",
            info="(fpr, tpr, color, label)",
            MC_save_npz=MC_save_npz,
            class_task=class_task,
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([-0.03, 1.03])
    plt.ylim([0.0, 1.03])
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if s_name != None:
        plt.savefig(s_name, bbox_inches="tight")
    plt.show()
