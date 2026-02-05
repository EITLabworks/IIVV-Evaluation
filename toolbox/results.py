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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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


def ResultsROCAUC(
    class_task, y_true, y_score, Y_labels, s_name=None, score_classes=None
):
    """
    score_classes: array-like of shape (n_classes,)
        The class labels corresponding to columns in y_score.
        If you have a sklearn classifier, pass clf.classes_ here.
        If None, we assume columns correspond to sorted unique labels from Y_labels.
    """

    # Color + label maps
    label_roc_plot = {
        "healthy_vs_cancer": "Healthy vs. cancer",
        "cancer_vs_lr_hr": "Cancer vs. low risk vs. high risk",
    }

    if class_task == "healthy_vs_cancer":
        cp = {1: "#003F5C", 0: "#FFA600"}
        cp_l = {1: "healthy", 0: "cancer"}
    elif class_task == "cancer_vs_lr_hr":
        cp = {0: "#FFA600", 1: "#FF6361", 2: "#58508D"}
        cp_l = {0: "Cancer", 1: "High risk", 2: "Low risk"}
    elif class_task == "MC":
        cp = {3: "#58508D", 0: "#FFA600", 1: "#003F5C", 2: "#FF6361"}
        cp_l = {
            0: "Cancer",
            1: "Healthy",
            2: "High-grade dysplasia",
            3: "Non-malignant",
        }
    else:
        raise ValueError(f"Unknown class_task={class_task}")

    # Define the label set we want to evaluate (and plot) in a stable order
    eval_classes = np.array(sorted(np.unique(Y_labels)))

    # Define the class order for the score columns
    if score_classes is None:
        score_classes = eval_classes
    else:
        score_classes = np.array(score_classes)

    if y_score.ndim == 1:
        # If someone passed probability of positive class only, make it (n,2)
        y_score = np.vstack([1 - y_score, y_score]).T

    n_score_classes = y_score.shape[1]
    if n_score_classes != len(score_classes):
        raise ValueError(
            f"y_score has {n_score_classes} columns, but score_classes has {len(score_classes)} labels."
        )

    # Map label -> column index in y_score
    score_col = {lbl: j for j, lbl in enumerate(score_classes)}

    # Make sure every eval class exists in the score columns
    missing = [c for c in eval_classes if c not in score_col]
    if missing:
        raise ValueError(
            f"These classes are in Y_labels but not in score_classes/y_score columns: {missing}. "
            f"Pass score_classes=clf.classes_ (or correct the order) to fix."
        )

    plt.figure(figsize=(5.5, 3))

    # ----- Binary case -----
    if len(eval_classes) == 2:
        # Decide which label is the "positive" class for ROC.
        # Common choice: the higher label; you can change this if needed.
        pos_label = eval_classes[1]
        pos_col = score_col[pos_label]

        # IMPORTANT: pass y_true as {0,1} indicator for pos_label if labels aren't already 0/1
        y_true_bin = (np.array(y_true) == pos_label).astype(int)

        fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, pos_col])
        roc_auc = auc(fpr, tpr)

        # Plot using the positive class color if available, else fallback
        color = cp.get(pos_label, "#003F5C")
        label_txt = f"{label_roc_plot.get(class_task, 'ROC')} (AUC = {roc_auc:.2f})"
        plt.plot(fpr, tpr, color=color, label=label_txt)
        print("AUC =", roc_auc)

        if s_name is not None:
            np.savez(
                s_name.split(".pdf")[0] + ".npz",
                fpr=fpr,
                tpr=tpr,
                color=color,
                roc_auc=roc_auc,
                label=label_txt,
                class_task=class_task,
                pos_label=pos_label,
                score_classes=score_classes,
            )

    # ----- Multiclass case (OvR) -----
    else:
        # Binarize with columns in eval_classes order
        y_bin = label_binarize(y_true, classes=eval_classes)
        MC_save_npz = {}

        # One-vs-rest ROC for each class label c
        for k, c in enumerate(eval_classes):
            col = score_col[c]  # column in y_score for class c
            fpr, tpr, _ = roc_curve(y_bin[:, k], y_score[:, col])
            roc_auc = auc(fpr, tpr)

            # NO FLIPPING. If <0.5, it’s telling you something is misaligned or inverted.
            plt.plot(
                fpr,
                tpr,
                color=cp.get(c, None),
                label=f"{cp_l.get(c, str(c))} (AUC = {roc_auc:.2f})",
            )
            MC_save_npz[cp_l.get(c, str(c))] = (
                fpr,
                tpr,
                cp.get(c, None),
                f"{cp_l.get(c, str(c))} (AUC = {roc_auc:.2f})",
            )

        # Micro-average: use the *same column alignment* as above
        y_score_aligned = np.column_stack(
            [y_score[:, score_col[c]] for c in eval_classes]
        )
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score_aligned.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color="black",
            linestyle="--",
            label=f"Micro-average (AUC = {roc_auc:.2f})",
        )
        print("Micro-average AUC =", roc_auc)

        MC_save_npz["Micro-average"] = (
            fpr,
            tpr,
            "black",
            f"Micro-average (AUC = {roc_auc:.2f})",
        )

        if s_name is not None:
            np.savez(
                s_name.split(".pdf")[0] + ".npz",
                info="(fpr, tpr, color, label)",
                MC_save_npz=MC_save_npz,
                class_task=class_task,
                eval_classes=eval_classes,
                score_classes=score_classes,
            )

    # Plot formatting
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([-0.03, 1.03])
    plt.ylim([0.0, 1.03])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if s_name is not None:
        plt.savefig(s_name, bbox_inches="tight")
    plt.show()
